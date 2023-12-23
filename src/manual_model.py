import logging
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from minigpt4.models.mini_gpt5 import MiniGPT5

class ManualArgs:
    cfg_path = "config/manual.yaml"
    options = []


class ManualMiniGPT5(MiniGPT5):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input_ids, labels, attention_mask, input_images=None, input_img_features=None, output_hidden_states=True):
        batch_size = input_ids.shape[0]
        all_input_embeds, all_attention, all_labels = [], [], []
        for b in range(batch_size):
            if input_img_features is not None:
                wrapped_img_embeds, wrapped_atts_img, wrapped_labels = self.input_warp(input_ids[b:b+1], attention_mask[b:b+1], labels[b:b+1], input_image_feature=input_img_features[b])
            elif input_images is not None:
                wrapped_img_embeds, wrapped_atts_img, wrapped_labels = self.input_warp(input_ids[b:b+1], attention_mask[b:b+1], labels[b:b+1], input_images[b])

            all_input_embeds.append(wrapped_img_embeds)
            all_attention.append(wrapped_atts_img)
            all_labels.append(wrapped_labels)

        #add padding features for batch
        max_len = max([x.shape[1] for x in all_input_embeds])
        for i in range(len(all_input_embeds)):
            if all_input_embeds[i].shape[1] < max_len:
                pad_len = max_len - all_input_embeds[i].shape[1]
                pad_embeds = torch.zeros([all_input_embeds[i].shape[0], pad_len, all_input_embeds[i].shape[2]]).to(all_input_embeds[i].device)
                pad_atts = torch.zeros([all_attention[i].shape[0], pad_len]).to(all_attention[i].device)
                pad_labels = torch.ones([all_labels[i].shape[0], pad_len], dtype=torch.long).to(all_labels[i].device) * -100
                all_input_embeds[i] = torch.cat([all_input_embeds[i], pad_embeds], dim=1)
                all_attention[i] = torch.cat([all_attention[i], pad_atts], dim=1)
                all_labels[i] = torch.cat([all_labels[i], pad_labels], dim=1)
        
        all_input_embeds = torch.cat(all_input_embeds, dim=0)
        all_attention = torch.cat(all_attention, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        past_key_values = None
        if self.using_prefix_tuning:
            device = all_input_embeds.device
            past_key_values = self.get_prompt(batch_size=batch_size, device=device)
            prefix_attention_mask = torch.ones(batch_size, self.peft_config.num_virtual_tokens).to(device)
            all_attention = torch.cat([prefix_attention_mask, all_attention], dim=1)
            # prefix_labels = torch.ones(batch_size, self.peft_config.num_virtual_tokens, dtype=wrapped_labels.dtype).to(device) * -100
            # wrapped_labels = torch.cat([prefix_labels, wrapped_labels], dim=1)

        outputs = self.llama_model(
                inputs_embeds=all_input_embeds,
                attention_mask=all_attention,
                return_dict=True,
                labels=all_labels,
                output_hidden_states=output_hidden_states,
                past_key_values=past_key_values,
            )
        output_token_index = (all_labels == self.output_img_id).nonzero()

        if len(output_token_index):
            addon_index = torch.ones_like(output_token_index)*(-1)
            addon_index[:, 0] = 0
            output_token_index += addon_index
        
        return outputs, output_token_index
    
    def encode_img(self, image, input_ids, attention_mask):
        device = image.device
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")
        with torch.autocast('cuda'):
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,  
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_llama = self.llama_proj(query_output.last_hidden_state)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(device)
        return inputs_llama, atts_llama
    
    def input_warp(self, input_ids, attention_mask, labels=None, input_image=None, input_image_feature=None):
        assert input_ids.shape[0] == 1, "warping each sample individually"

        bos = torch.ones([1, 1],
                        dtype=input_ids.dtype,
                        device=input_ids.device) * self.llama_tokenizer.bos_token_id
        if self.using_lora:
            bos_embeds = self.llama_model.base_model.model.model.embed_tokens(bos)
        else:
            bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = torch.ones([1, 1],dtype=attention_mask.dtype,device=attention_mask.device)
        if labels is not None:
            labels_bos = torch.ones([1, 1],dtype=labels.dtype,device=labels.device) * -100
            wrapped_labels = labels_bos
        else:
            wrapped_labels = None
        
        wrapped_img_embeds, wrapped_atts_img = bos_embeds, atts_bos
        input_img_idx = (input_ids == self.input_img_id).nonzero(as_tuple=True)
        start_idx = 0
        if len(input_img_idx[0]) > 0:
            assert input_image is not None or input_image_feature is not None, 'input_image or input_image_feature should be provided'

            if input_image_feature is not None:
                img_embeds = input_image_feature
                atts_img = torch.ones(img_embeds.size()[:-1], dtype=torch.long).to(img_embeds.device)
            else:
                img_embeds, atts_img = self.encode_img(input_image, input_ids, attention_mask)
            
            if labels is not None:
                img_label = torch.ones_like(atts_img, dtype=torch.long).to(img_embeds.device) * -100

            for i in range(len(input_img_idx[1])):
                p_before = input_ids[:, start_idx:input_img_idx[1][i]]
                p_before_attention_mask = attention_mask[:, start_idx:input_img_idx[1][i]]
                p_before_embeds = self.get_input_embeddings(p_before)
                wrapped_img_embeds = torch.cat([wrapped_img_embeds, p_before_embeds, img_embeds[i:i+1]], dim=1)
                wrapped_atts_img = torch.cat([wrapped_atts_img, p_before_attention_mask, atts_img[i:i+1]], dim=1)
                if labels is not None:
                    p_before_labels = labels[:, start_idx:input_img_idx[1][i]]
                    wrapped_labels = torch.cat([wrapped_labels, p_before_labels, img_label[i:i+1]], dim=1)
                start_idx = input_img_idx[1][i] + 1
            
        p_before = input_ids[:, start_idx:]
        p_before_attention_mask = attention_mask[:, start_idx:]
        p_before_embeds = self.get_input_embeddings(p_before)
        wrapped_img_embeds = torch.cat([wrapped_img_embeds, p_before_embeds], dim=1)
        wrapped_atts_img = torch.cat([wrapped_atts_img, p_before_attention_mask], dim=1)
        if labels is not None:
            p_before_labels = labels[:, start_idx:]
            wrapped_labels = torch.cat([wrapped_labels, p_before_labels], dim=1)
        return wrapped_img_embeds, wrapped_atts_img, wrapped_labels
    


