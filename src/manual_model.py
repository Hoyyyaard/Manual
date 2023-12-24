import logging
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
from lightning.pytorch import LightningModule
from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from minigpt4.models.mini_gpt5 import MiniGPT5, StoppingCriteriaList, StoppingCriteriaSub
from model import MiniGPT5_Model
from minigpt4.common.config import Config
from diffusers import AutoencoderKL, UNet2DConditionModel
import wandb
import torch.nn.functional as F
from utils import plot_images_and_text
from constants import *
from diffusers import StableDiffusionPipeline
from diffusers.models.vae import DiagonalGaussianDistribution
from transformers import get_linear_schedule_with_warmup, CLIPTextModel, CLIPTokenizer, PreTrainedTokenizer


class ManualArgs:
    cfg_path = "config/manual.yaml"
    options = []


class ManualMiniGPT5(MiniGPT5):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input_ids, labels, attention_mask, input_images=None, input_img_features=None, output_hidden_states=True, captions=None):
        '''
            Fork From: models.mini_gpt5.MiniGPT5.forward
            Modify: 
                1. Add captions as input to instruct qformer
        '''
        batch_size = input_ids.shape[0]
        all_input_embeds, all_attention, all_labels = [], [], []
        for b in range(batch_size):
            if input_img_features is not None:
                wrapped_img_embeds, wrapped_atts_img, wrapped_labels = self.input_warp(input_ids[b:b+1], attention_mask[b:b+1], labels[b:b+1], input_image_feature=input_img_features[b])
            elif input_images is not None:
                wrapped_img_embeds, wrapped_atts_img, wrapped_labels = self.input_warp(input_ids[b:b+1], attention_mask[b:b+1], labels[b:b+1], input_images[b], caption=captions[b])

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
        '''
            Fork From: models.mini_gpt5.MiniGPT5.encode_img
            Modify: 
                1. Add caption_inpud_ids as input to instruct qformer
                2. Remove attention mask as input 
                3. Suit to parrallel multiply images qformer forward
        '''
        device = image.device
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")
        with torch.autocast('cuda'):
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)  # [img_num, 257, 1408]
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
                        
            '''
            ImageHere = self.llama_tokenizer.convert_tokens_to_ids('<ImageHere>')
            Img = self.llama_tokenizer.convert_tokens_to_ids('<Img>')
            Img_ = self.llama_tokenizer.convert_tokens_to_ids('</Img>')
            
            query_outputs = []
            for imgbed, imgatt in zip(image_embeds,image_atts):
                # query_tokens = self.query_tokens.expand(imgbed.shape[0], -1, -1)
                query_output = self.Qformer.bert(
                    input_ids=input_ids,
                    # attention_mask=attention_mask,  
                    query_embeds=self.query_tokens,
                    encoder_hidden_states=imgbed.unsqueeze(0),
                    encoder_attention_mask=imgatt.unsqueeze(0),
                    return_dict=True,
                )
                query_outputs.append(query_output)
            '''
            
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            input_ids = input_ids.expand(image_embeds.shape[0], -1) 
            # query_outputs.last_hidden_state [view, 32+len(tokens(instr)), 768]
            query_outputs = self.Qformer.bert(
                input_ids=input_ids,
                # attention_mask=attention_mask,  
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            
            '''
            [bs, 32+len(tokens(instr)) * view_num, 768] 
            query_outputs_last_hidden_state = query_outputs.last_hidden_state.view(1, -1, query_outputs.last_hidden_state.shape[-1])
            Do not do above as view as multiple images input
            '''
            inputs_llama = self.llama_proj(query_outputs.last_hidden_state)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(device)
        return inputs_llama, atts_llama
    
    def input_warp(self, input_ids, attention_mask, labels=None, input_image=None, input_image_feature=None, caption=None):
        '''
            Fork From: models.mini_gpt5.MiniGPT5.input_warp
            Modify: 
                1. Add instruction_input_ids to guide qformer as origin input_ids has special tokens
        '''
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
                # Only input intruction to qformer
                '''
                len(tokens(<Img>)) = 3 len(tokens(</Img>)) = 3 len(tokens(<ImageHere>)) = 1 [IMG0] : 32000
                <Img><ImageHere><ImageHere><ImageHere></Img> : Prefix_index = 3 + len(input_image) + 3
                <Img><ImageHere></Img><Img><ImageHere></Img> : Prefix_index = 7 * len(input_image)
                IMG0_index = (input_ids == 32000).nonzero(as_tuple=True)[1].item()
                '''
                instruction_input_ids = self.llama_tokenizer([caption], return_tensors="pt", add_special_tokens=False).to(input_image.device).input_ids
                img_embeds, atts_img = self.encode_img(input_image, instruction_input_ids, None)
            
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
    
    def predict(self, instruction, input_image, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, task_name=None, output_hidden_states=True, force_generation=False, caption=None):
        '''
            Fork From: models.mini_gpt5.MiniGPT5.predict
            Modify: 
                1. Add captions input to guide qformer 
        '''
        sample_inputs = self.llama_tokenizer(instruction, return_tensors="pt", add_special_tokens = False).to(self.device)
        input_ids = sample_inputs.input_ids
        attention_mask = sample_inputs.attention_mask

        wrapped_img_embeds, wrapped_atts_img, _ = self.input_warp(input_ids, attention_mask, input_image=input_image, caption=caption)

        stop_words_ids = [torch.tensor([835]).to(self.device),
                          torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        # if self.using_prefix_tuning:

        sample_outputs = self.llama_model.generate(
            inputs_embeds=wrapped_img_embeds,
            attention_mask=wrapped_atts_img,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=True,
            force_generation=force_generation
        )

        return sample_outputs


class Manual_MiniGPT5_Model(MiniGPT5_Model, LightningModule):
    def __init__(self, encoder_model_config, **kwargs):
        '''
            Fork From: model.MiniGPT5_Model.__init__
            Modify: 
                1. Modify self.model to ManualMiniGPT5
                2. Do not init MiniGPT5_Model
        '''
        super().__init__(encoder_model_config, **kwargs)
        self.save_hyperparameters(ignore=['encoder_model_config'])
        self.encoder_model_config = encoder_model_config
        self.input_vis_processor = None

        if encoder_model_config.model_type == 'multimodal_encoder':
            print("Use Manual Minigpt5 Model")
            manual_config = Config(ManualArgs)
            self.model = ManualMiniGPT5.from_config(manual_config.model_cfg)
            self.tokenizer = self.model.llama_tokenizer

            hidden_size = self.model.llama_model.config.hidden_size

        sd_model_name = "stabilityai/stable-diffusion-2-1-base"

        self.sd_text_encoder = CLIPTextModel.from_pretrained(sd_model_name, subfolder="text_encoder")
        self.sd_tokenizer = CLIPTokenizer.from_pretrained(sd_model_name, subfolder="tokenizer")
        self.vae = AutoencoderKL.from_pretrained(sd_model_name, subfolder="vae").to(PRECISION)
        
        self.unet = UNet2DConditionModel.from_pretrained(sd_model_name, subfolder="unet").to(PRECISION)
        # Freeze vae and text_encoder
        self.vae.requires_grad_(False)
        self.sd_text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)

        sd_hidden_size = self.sd_text_encoder.config.hidden_size
        self.t2i_decoder_prompt = torch.nn.Parameter(torch.randn((1,77, sd_hidden_size), dtype=TRAINABLE_PRECISION))
        self.llm_to_t2i_mapping = nn.Transformer(batch_first=True, norm_first=True, d_model = sd_hidden_size, num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=sd_hidden_size*4, dropout=0.0, dtype=TRAINABLE_PRECISION)
        
        if len(ALL_IMG_TOKENS):
            self.output_img_id = self.tokenizer.convert_tokens_to_ids(ALL_IMG_TOKENS[0])
        self.img_token_num = IMG_TOKEN_NUM

        self.image_pipeline = StableDiffusionPipeline.from_pretrained(
            sd_model_name,
            vae = self.vae,
            unet = self.unet,
            safety_checker = None,
        )

        self.noise_scheduler = self.image_pipeline.scheduler

        self.fc = nn.Sequential(
                    nn.Linear(hidden_size, sd_hidden_size),
                    nn.GELU(),
                    nn.Linear(sd_hidden_size, sd_hidden_size),
                ).to(TRAINABLE_PRECISION)

        empty_text_feature = self.encode_caption('', self.sd_tokenizer.model_max_length, inference=True)
        self.register_buffer('empty_text_feature', empty_text_feature, persistent=False)

        zero_img_feature = torch.zeros((1, self.img_token_num, hidden_size), dtype=TRAINABLE_PRECISION)
        self.register_buffer('zero_img_feature', zero_img_feature, persistent=False)

        self.sd_text_encoder.to(PRECISION)

        if IS_STAGE2:
            for n, p in self.fc.named_parameters():
                p.requires_grad = False
            # for n, p in self.llm_to_t2i_mapping.named_parameters():
            #     p.requires_grad = False
            self.t2i_decoder_prompt.requires_grad = False
    
    def generate(self, utterance, input_image=None, task_name=None, max_new_tokens=256, force_generation=False, guidance_scale=7.5, caption=None):
        '''
            Fork From: model.MiniGPT5_Model.generate
            Modify: 
                1. Add captions input to guide qformer 
        '''
        self.image_pipeline.to(self.device, PRECISION)
        if input_image is None:
            input_image = torch.zeros((1, 3, 224, 224), dtype=PRECISION).to(self.device)
        if type(utterance) == str:
            utterance = [utterance]
        llm_sample_outputs = self.model.predict(utterance, input_image, max_new_tokens=max_new_tokens, temperature=1.0, repetition_penalty=2.0, task_name=task_name, force_generation=force_generation, caption=caption)
        new_tokens = llm_sample_outputs['sequences'][0]
        pred_out = self.tokenizer.decode(new_tokens)
        print(f'Generated text: {pred_out}')

        last_hidden_state = llm_sample_outputs['hidden_states']
        special_token_index = (new_tokens == self.output_img_id).nonzero()

        predicted_images_ft = None
        if len(special_token_index):
            idx = special_token_index[0,0]
            t2i_input_embedding = last_hidden_state[idx][-1]
            assert t2i_input_embedding.shape[1] == self.img_token_num
            img0_output_feature = last_hidden_state[idx-1][-1][:, -1:]
            t2i_input_embedding = torch.cat([img0_output_feature, t2i_input_embedding[:, :-1]], dim=1)
            t2i_input_embedding = self.fc(t2i_input_embedding)
            mapping_feature = self.llm_to_t2i_mapping(src=t2i_input_embedding, tgt=self.t2i_decoder_prompt)

            if USE_CFG:
                empty_feature = self.fc(self.zero_img_feature)
                empty_feature = self.llm_to_t2i_mapping(src=empty_feature, tgt=self.t2i_decoder_prompt)
                predicted_images_ft = self.image_pipeline(prompt_embeds = mapping_feature, negative_prompt_embeds=empty_feature, guidance_scale=guidance_scale).images[0]
            else:
                predicted_images_ft = self.image_pipeline(prompt_embeds = mapping_feature, guidance_scale=guidance_scale, use_original=True).images[0]
            
        return pred_out, predicted_images_ft
    
    def training_step(self, batch, batch_idx):
        '''
            Fork From: model.MiniGPT5_Model.training_step
            Modify: 
                1. Add captions input to self.generate function
        '''
        for key in batch.keys():
            if type(batch[key]) == list:
                batch[key] = batch[key]
            else:
                batch[key] = batch[key].to(self.device)

        input_ids = batch['input_ids']
        labels = batch['labels']
        attention_mask = batch['attention_mask']
        source_text = batch['source']
        target_text = batch['target']
        captions = batch['caption']
        input_images = batch.get('input_images', None)
        output_image = batch.get('output_image', None)
        input_images_feature = batch.get('input_images_feature', None)
        output_image_feature = batch.get('output_image_feature', None)

        bs = len(source_text)
        loss_dict = self(input_ids, attention_mask, input_images, output_image, labels, captions, input_images_feature, output_image_feature)
        loss = loss_dict['loss']
        log_dict = {f'train_{k}': v for k, v in loss_dict.items()}
        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)
        # check image generation for every 1000 steps
        if (self.global_step+1) % self.encoder_model_config.check_generate_step == 0 and self.global_rank == 0:
            with torch.no_grad():
                self.eval()
                # utterance = "generate image with caption: a man on the sofa."
                utterance = source_text[0]
                gt_text = target_text[0]
                caption = captions[0]
                i_image = None
                if "<ImageHere>" in utterance:
                    i_image = input_images[0]
                text_out, image_out = self.generate(utterance, i_image, caption=caption)
                if image_out is not None:
                    if os.path.exists(tep:=os.path.join(OUTPUT_FOLDER,"train_eval")) == False:
                        os.makedirs(tep)
                    if IS_STAGE2:
                        data = [[self.global_step, utterance, text_out, wandb.Image(image_out), gt_text]]
                        columns = ["step", "input_utterance", "text_out", "img_out", "gt_text"]
                    else:
                        if captions[0] is not None:
                            predicted_images_nl = self.image_pipeline(prompt= captions[0]).images[0]
                            data = [[self.global_step, utterance, text_out, wandb.Image(image_out), captions[0], wandb.Image(predicted_images_nl)]]
                            columns = ["step", "input_utterance", "text_out", "img_out", "caption", "caption_out"]
                            predicted_images_nl.save(os.path.join(OUTPUT_FOLDER, "train_eval", f'{self.global_step}_nl.png'))
                        else:
                            data = [[self.global_step, utterance, text_out, wandb.Image(image_out), gt_text]]
                            columns = ["step", "input_utterance", "text_out", "img_out", "gt_text"]
                    self.logger.log_table(key="sample", data=data, columns=columns)
                    image_out.save(os.path.join(OUTPUT_FOLDER, "train_eval", f'{self.global_step}.png'))
                else:
                    data = [[self.global_step, utterance, text_out, None, gt_text]]
                    columns = ["step", "input_utterance", "text_out", "img_out", "gt_text"]
                    self.logger.log_table(key="sample", data=data, columns=columns)
                self.train()
        return loss
