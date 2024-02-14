#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to fine-tune InstructPix2Pix."""

import argparse
import logging
import math
import os
from pathlib import Path
from typing import Optional
from accelerate import init_empty_weights
import shutil
import accelerate
import datasets
import diffusers
import numpy as np
import PIL
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from diffusers import (AutoencoderKL, DDPMScheduler,
                       StableDiffusionInstructPix2PixPipeline,
                       StableUnCLIPImg2ImgPipeline,
                       StableDiffusionPipeline,
                       UNet2DConditionModel,)
from diffusers.pipelines.stable_diffusion.stable_unclip_image_normalizer import StableUnCLIPImageNormalizer
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.models.embeddings import get_timestep_embedding
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel, AutoProcessor
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers, is_wandb_available


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.15.0.dev0")

logger = get_logger(__name__, log_level="INFO")


WANDB_TABLE_COL_NAMES = ["original_image", "edited_image", "edit_prompt"]
# Abs file dir of this file
current_file_path = os.path.abspath(__file__)
# parent directory of this file
parent_directory = os.path.dirname(current_file_path)
base_dir = os.path.dirname(parent_directory)
# print(base_dir)
import sys
sys.path.append(base_dir)
from minigpt4.models.blip2 import Blip2Base, disabled_train
from transformers import LlamaTokenizer
from accelerate import DistributedDataParallelKwargs


class Latent_Qformer(nn.Module):
    def __init__(self, vocab_size, image_size=512):
        super().__init__()

        self.visual_encoder, self.ln_vision = Blip2Base.init_vision_encoder(
            'eva_clip_g', image_size, 0, False, 'fp16'
        )
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder = self.visual_encoder.eval()
        self.visual_encoder.train = disabled_train
        for name, param in self.ln_vision.named_parameters():
            param.requires_grad = False
        self.ln_vision = self.ln_vision.eval()
        self.ln_vision.train = disabled_train

        # 64 querys & 768 vision fts dim
        self.qformer, self.query_tokens = Blip2Base.init_Qformer(64, 1408, vocab_size=vocab_size)
        self.ln_i2t = nn.Linear(768, 1024)

        del self.qformer.cls.predictions.decoder

    def forward(self, images, input_ids):
        image_embeds = self.ln_vision(self.visual_encoder(images))
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        # query_outputs.last_hidden_state [view, 32+len(tokens(instr)), 768]
        query_outputs = self.qformer.bert(
            input_ids=input_ids,
            # attention_mask=attention_mask,  
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=None,
            return_dict=True,
        )
        # Use only query output
        x = query_outputs.last_hidden_state[:, :query_tokens.shape[1], :]
        x = self.ln_i2t(x)
        return x



class StableDiffusionQformerPipeline(StableDiffusionPipeline):
    
    def prepare_image_latents(
        self, images, batch_size, num_images_per_prompt, dtype, device, do_classifier_free_guidance, generator=None
    ):

        def retrieve_latents(
            encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
        ):
            if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
                return encoder_output.latent_dist.sample(generator)
            elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
                return encoder_output.latent_dist.mode()
            elif hasattr(encoder_output, "latents"):
                return encoder_output.latents
            else:
                raise AttributeError("Could not access latents of provided encoder_output")

        image_latents = []
        for image in images:
            image = image.to(device=device, dtype=dtype)

            batch_size = batch_size * num_images_per_prompt

            image_latent = retrieve_latents(self.vae.encode(image), sample_mode="argmax")
            image_latents.append(image_latent)
        image_latents = torch.stack(image_latents, dim=0)
        image_latents = torch.sum(image_latents, dim=0)

        if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
            # expand image_latents for batch_size
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {image_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
            additional_image_per_prompt = batch_size // image_latents.shape[0]
            image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            image_latents = torch.cat([image_latents], dim=0)

        if do_classifier_free_guidance:
            uncond_image_latents = torch.zeros_like(image_latents)
            image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)

        return image_latents

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        image,
        num_inference_steps: int = 100,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        negative_prompt = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end = None,
        callback_on_step_end_tensor_inputs = ["latents"],
        args = None,
        **kwargs,
    ):


        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        # 0. Check inputs
        self.check_inputs(
            prompt,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )
        self._guidance_scale = guidance_scale
        self._image_guidance_scale = image_guidance_scale

        if image is None:
            raise ValueError("`image` input cannot be undefined.")

        # 1. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # check if scheduler is in sigmas space
        scheduler_is_in_sigma_space = hasattr(self.scheduler, "sigmas")

        # 2. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 3. Preprocess image
        image = [self.image_processor.preprocess(img) for img in image]

        # 4. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare Image latents
    
        image_latents = self.prepare_image_latents(
            image,
            batch_size,
            num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            self.do_classifier_free_guidance,
        )

        if args.zero_base:
            image_latents = torch.zeros_like(image_latents)

        height, width = image_latents.shape[-2:]
        height = height * self.vae_scale_factor
        width = width * self.vae_scale_factor

        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 7. Check that shapes of latents and image match the UNet channels
        num_channels_image = image_latents.shape[1]
        if num_channels_latents + num_channels_image != self.unet.config.in_channels:
            raise ValueError(
                f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                f" `num_channels_image`: {num_channels_image} "
                f" = {num_channels_latents+num_channels_image}. Please verify the config of"
                " `pipeline.unet` or your `image` input."
            )

        # 8. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 9. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Expand the latents if we are doing classifier free guidance.
                # The latents are expanded 3 times because for pix2pix the guidance\
                # is applied for both the text and the input image.
                latent_model_input = torch.cat([latents] * 3) if self.do_classifier_free_guidance else latents

                # concat latents, image_latents in the channel dimension
                scaled_latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                scaled_latent_model_input = torch.cat([scaled_latent_model_input, image_latents], dim=1)

                # predict the noise residual
                noise_pred = self.unet(
                    scaled_latent_model_input, t, encoder_hidden_states=prompt_embeds, return_dict=False
                )[0]

                # Hack:
                # For karras style schedulers the model does classifer free guidance using the
                # predicted_original_sample instead of the noise_pred. So we need to compute the
                # predicted_original_sample here if we are using a karras style scheduler.
                if scheduler_is_in_sigma_space:
                    step_index = (self.scheduler.timesteps == t).nonzero()[0].item()
                    sigma = self.scheduler.sigmas[step_index]
                    noise_pred = latent_model_input - sigma * noise_pred

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                    noise_pred = (
                        noise_pred_uncond
                        + self.guidance_scale * (noise_pred_text - noise_pred_image)
                        + self.image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                    )

                # Hack:
                # For karras style schedulers the model does classifer free guidance using the
                # predicted_original_sample instead of the noise_pred. But the scheduler.step function
                # expects the noise_pred and computes the predicted_original_sample internally. So we
                # need to overwrite the noise_pred here such that the value of the computed
                # predicted_original_sample is correct.
                if scheduler_is_in_sigma_space:
                    noise_pred = (noise_pred - latents) / (-sigma)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    image_latents = callback_outputs.pop("image_latents", image_latents)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
            has_nsfw_concept = None
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple example of a training script for InstructPix2Pix."
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default='insp2p',
        required=True,
        choices=["insp2p", "unclip", "qformer"],
    )
    parser.add_argument(
        "--lora",
        action="store_true",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--original_image_column",
        type=str,
        default="original_image",
        help="The column of the dataset containing the original image on which edits where made.",
    )
    parser.add_argument(
        "--edited_image_column",
        type=str,
        default="cartoonized_image",
        help="The column of the dataset containing the edited image.",
    )
    parser.add_argument(
        "--edit_prompt_column",
        type=str,
        default="edit_prompt",
        help="The column of the dataset containing the edit instruction.",
    )
    parser.add_argument(
        "--val_image_url",
        type=str,
        default=None,
        help="URL to the original image that you would like to edit (used during inference for debugging purposes).",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is sampled during training for inference.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="instruct-pix2pix-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--val",
        action="store_true",
        help="Whether or not to run validation during training.",
    )
    parser.add_argument(
        "--only_val",
        action="store_true",
        help="Whether or not to run validation during training.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=None,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--use_exo",
        action="store_true",
    )
    parser.add_argument(
        "--avg_exo",
        action="store_true",
    )
    parser.add_argument(
        "--from_scratch",
        action="store_true",
    )
    parser.add_argument(
        "--zero_base",
        action="store_true",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--use_ema", action="store_true", help="Whether to use EMA model."
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=2,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    # if args.dataset_name is None and args.train_data_dir is None:
    #     raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision
    os.environ["WANDB_DIR"] = args.output_dir
    return args

def convert_to_np(image, resolution):
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)

def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

def insp2p_forward(unet, batch, args, vae, weight_dtype, bsz, latents, generator, text_encoder, tokenize_captions, accelerator, noisy_latents, timesteps, encoder_hidden_states, feature_extractor, image_encoder, image_normalizer, image_noising_scheduler, qformer):
    # Add the exo pixel values as base image
    original_image_embeds = vae.encode(
        batch["original_pixel_values"].to(weight_dtype)
    ).latent_dist.mode()

    if args.zero_base:
        original_image_embeds = torch.zeros_like(original_image_embeds)

    # original_image_embeds = latent_qformer(batch['exo_pixel_values'], batch['input_ids'])

    # Conditioning dropout to support classifier-free guidance during inference. For more details
    # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
    if args.conditioning_dropout_prob is not None:
        random_p = torch.rand(
            bsz, device=latents.device, generator=generator
        )
        # Sample masks for the edit prompts.
        prompt_mask = random_p < 2 * args.conditioning_dropout_prob
        prompt_mask = prompt_mask.reshape(bsz, 1, 1)
        # Final text conditioning.
        null_conditioning = text_encoder(
            tokenize_captions([""]).to(accelerator.device)
        )[0]
        encoder_hidden_states = torch.where(
            prompt_mask, null_conditioning, encoder_hidden_states
        )

        # Sample masks for the original images.
        image_mask_dtype = original_image_embeds.dtype
        image_mask = 1 - (
            (random_p >= args.conditioning_dropout_prob).to(
                image_mask_dtype
            )
            * (random_p < 3 * args.conditioning_dropout_prob).to(
                image_mask_dtype
            )
        )
        image_mask = image_mask.reshape(bsz, 1, 1, 1)
        # Final image conditioning.
        original_image_embeds = image_mask * original_image_embeds

    # Concatenate the `original_image_embeds` with the `noisy_latents`.
    concatenated_noisy_latents = torch.cat(
        [noisy_latents, original_image_embeds], dim=1
    )

    # Predict the noise residual and compute loss
    model_pred = unet(
        concatenated_noisy_latents, timesteps, encoder_hidden_states
    ).sample

    return model_pred

def insp2p_eval(args, batch, pipeline, generator, bn, qformer):

    # original_image = batch['original_image'][bn].resize((args.resolution, args.resolution))
    edited_image = (
        pipeline(
            batch['text'][bn],
            image=batch['original_image'][bn],
            num_inference_steps=20,
            image_guidance_scale=1.5,
            guidance_scale=7,
            generator=generator,
            args=args
        ).images[0]
    )

    return edited_image, batch['original_image'][bn]

def unclip_forward(unet, batch, args, vae, weight_dtype, bsz, latents, generator, text_encoder, tokenize_captions, accelerator, noisy_latents, timesteps, encoder_hidden_states, feature_extractor, image_encoder, image_normalizer, image_noising_scheduler, qformer):
    
    def noise_image_embeddings(
        image_embeds: torch.Tensor,
        noise_level: int,
        noise: Optional[torch.FloatTensor] = None,
        generator: Optional[torch.Generator] = None,
    ):
        """
        Add noise to the image embeddings. The amount of noise is controlled by a `noise_level` input. A higher
        `noise_level` increases the variance in the final un-noised images.

        The noise is applied in two ways:
        1. A noise schedule is applied directly to the embeddings.
        2. A vector of sinusoidal time embeddings are appended to the output.

        In both cases, the amount of noise is controlled by the same `noise_level`.

        The embeddings are normalized before the noise is applied and un-normalized after the noise is applied.
        """
        if noise is None:
            noise = randn_tensor(
                image_embeds.shape, generator=generator, device=image_embeds.device, dtype=image_embeds.dtype
            )

        noise_level = torch.tensor([noise_level] * image_embeds.shape[0], device=image_embeds.device)

        image_normalizer.to(image_embeds.device)
        image_embeds = image_normalizer.scale(image_embeds)

        image_embeds = image_noising_scheduler.add_noise(image_embeds, timesteps=noise_level, noise=noise)

        image_embeds = image_normalizer.unscale(image_embeds)

        noise_level = get_timestep_embedding(
            timesteps=noise_level, embedding_dim=image_embeds.shape[-1], flip_sin_to_cos=True, downscale_freq_shift=0
        )

        # `get_timestep_embeddings` does not contain any weights and will always return f32 tensors,
        # but we might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        noise_level = noise_level.to(image_embeds.dtype)

        image_embeds = torch.cat((image_embeds, noise_level), 1)

        return image_embeds
    
    def _encode_image(
        image,
        device,
        batch_size,
        num_images_per_prompt,
        # do_classifier_free_guidance,
        noise_level,
        generator,
        image_embeds,
    ):
        dtype = weight_dtype
        if isinstance(image, PIL.Image.Image):
            # the image embedding should repeated so it matches the total batch size of the prompt
            repeat_by = batch_size
        else:
            # assume the image input is already properly batched and just needs to be repeated so
            # it matches the num_images_per_prompt.
            #
            # NOTE(will) this is probably missing a few number of side cases. I.e. batched/non-batched
            # `image_embeds`. If those happen to be common use cases, let's think harder about
            # what the expected dimensions of inputs should be and how we handle the encoding.
            repeat_by = num_images_per_prompt

        if image_embeds is None:
            if not isinstance(image, torch.Tensor):
                image = feature_extractor(images=image, return_tensors="pt").pixel_values

            image = image.to(device=device, dtype=dtype)
            image_embeds = image_encoder(image).image_embeds

        image_embeds = noise_image_embeddings(
            image_embeds=image_embeds,
            noise_level=noise_level,
            generator=generator,
        )

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        image_embeds = image_embeds.unsqueeze(1)
        bs_embed, seq_len, _ = image_embeds.shape
        image_embeds = image_embeds.repeat(1, repeat_by, 1)
        image_embeds = image_embeds.view(bs_embed * repeat_by, seq_len, -1)
        image_embeds = image_embeds.squeeze(1)

        # if do_classifier_free_guidance:
        #     negative_prompt_embeds = torch.zeros_like(image_embeds)

        #     # For classifier free guidance, we need to do two forward passes.
        #     # Here we concatenate the unconditional and text embeddings into a single batch
        #     # to avoid doing two forward passes
        #     image_embeds = torch.cat([negative_prompt_embeds, image_embeds])

        return image_embeds
    
    
    device = latents.device
    noise_level = 0
    noise_level = torch.tensor([noise_level], device=device)
    original_image_embeds = _encode_image(
        image=batch['original_image'],
        device=device,
        batch_size=bsz,
        num_images_per_prompt=1,
        # do_classifier_free_guidance=do_classifier_free_guidance,
        noise_level=noise_level,
        generator=generator,
        image_embeds=None,
    )

    # Conditioning dropout to support classifier-free guidance during inference. For more details
    # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
    if args.conditioning_dropout_prob is not None:
        random_p = torch.rand(
            bsz, device=latents.device, generator=generator
        )
        # Sample masks for the edit prompts.
        prompt_mask = random_p < 2 * args.conditioning_dropout_prob
        prompt_mask = prompt_mask.reshape(bsz, 1, 1)
        # Final text conditioning.
        null_conditioning = text_encoder(
            tokenize_captions([""]).to(accelerator.device)
        )[0]
        encoder_hidden_states = torch.where(
            prompt_mask, null_conditioning, encoder_hidden_states
        )

        # Sample masks for the original images.
        image_mask_dtype = original_image_embeds.dtype
        image_mask = 1 - (
            (random_p >= args.conditioning_dropout_prob).to(
                image_mask_dtype
            )
            * (random_p < 3 * args.conditioning_dropout_prob).to(
                image_mask_dtype
            )
        )
        image_mask = image_mask.reshape(bsz, 1, 1, 1)
        # Final image conditioning.
        original_image_embeds = image_mask * original_image_embeds

    # Predict the noise residual and compute loss
    model_pred = unet(
        noisy_latents, timesteps, encoder_hidden_states, class_labels=original_image_embeds,
    ).sample

    return model_pred

def unclip_eval(args, batch, pipeline, generator, bn, qformer):
    # original_image = batch['original_image'][bn].resize((args.resolution, args.resolution))
    edited_image = (
        pipeline(
            batch['original_image'][bn],
            prompt=batch['text'][bn],
        ).images[0]
    )

    return edited_image, batch['original_image'][bn]

def qformer_forward(unet, batch, args, vae, weight_dtype, bsz, latents, generator, text_encoder, tokenize_captions, accelerator, noisy_latents, timesteps, encoder_hidden_states, feature_extractor, image_encoder, image_normalizer, image_noising_scheduler, qformer):

    encoder_hidden_states = qformer(batch["original_pixel_values"].to(weight_dtype), batch["input_ids"])

    # Predict the noise residual and compute loss
    model_pred = unet(
        noisy_latents, timesteps, encoder_hidden_states
    ).sample

    return model_pred

def qformer_eval(args, batch, pipeline, generator, bn, qformer):
    # original_image = batch['original_image'][bn].resize((args.resolution, args.resolution))
    prompt_embeds = qformer(batch["original_pixel_values"], batch['input_ids'])
    edited_image = (
        pipeline(
            prompt_embeds=prompt_embeds,
        ).images[0]
    )

    return edited_image, batch['original_image'][bn]

def main():
    args = parse_args()

    PIPELINE_POOL = {
                'insp2p' : {'class': StableDiffusionInstructPix2PixPipeline, 
                            'hf':'timbrooks/instruct-pix2pix', 
                            'method': insp2p_forward,
                            'eval_method': insp2p_eval}, 
                'unclip' : {'class': StableUnCLIPImg2ImgPipeline, 
                            'hf':'stabilityai/stable-diffusion-2-1-unclip', 
                            'method': unclip_forward,
                            'eval_method': unclip_eval},
                'qformer' : {'class': StableDiffusionPipeline,
                            'hf':'stabilityai/stable-diffusion-2-1', 
                            'method': qformer_forward,
                            'eval_method': qformer_eval}
                }


    pretrained_model_name_or_path = PIPELINE_POOL[args.pipeline]['hf']
    pipeline_class = PIPELINE_POOL[args.pipeline]['class']
    pipeline_method = PIPELINE_POOL[args.pipeline]['method']
    pipeline_eval_method = PIPELINE_POOL[args.pipeline]['eval_method']

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )
        import wandb    
    
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit,
        project_dir=logging_dir,
    )
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs]
    )

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)


    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(
        pretrained_model_name_or_path, subfolder="scheduler"
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )

    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae", revision=args.revision
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if args.pipeline == 'unclip':
        feature_extractor = CLIPImageProcessor.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="feature_extractor",
            revision=args.revision,
        )
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="image_encoder",
            revision=args.revision,
        )
        image_normalizer = StableUnCLIPImageNormalizer.from_pretrained(     
            pretrained_model_name_or_path,
            subfolder="image_normalizer",
            revision=args.revision,
        )
        image_noising_scheduler = DDPMScheduler.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="image_noising_scheduler",
            revision=args.revision,
        )
        image_encoder.requires_grad_(False)
        image_encoder.to(accelerator.device, dtype=weight_dtype)
    else:
        feature_extractor = None
        image_encoder = None
        image_normalizer = None
        image_noising_scheduler = None
    latent_qformer = None

    if args.from_scratch:
        with init_empty_weights():
            unet = UNet2DConditionModel.from_pretrained(
                    pretrained_model_name_or_path,
                    subfolder="unet",
                    revision=args.non_ema_revision,
                )
    else:
        unet = UNet2DConditionModel.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="unet",
                revision=args.non_ema_revision,
            )

    if args.pipeline == 'qformer':
        latent_qformer = Latent_Qformer(vocab_size=tokenizer.vocab_size, image_size=args.resolution)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    if args.lora:
    # Add adapter and make sure the trainable params are in float32.
        print('----------------Use lora finetune----------------')
        for param in unet.parameters():
            param.requires_grad_(False)

        unet_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        unet.add_adapter(unet_lora_config)
        if args.mixed_precision == "fp16":
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(unet, dtype=torch.float32)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = EMAModel(
            unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config
        )

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    # `accelerate` 0.16.0 will have better support for customized saving
    # if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
    #     # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    #     def save_model_hook(models, weights, output_dir):
    #         if args.use_ema:
    #             ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

    #         for i, model in enumerate(models):
    #             model.save_pretrained(os.path.join(output_dir, "unet"))

    #             # make sure to pop weight so that corresponding model is not saved again
    #             weights.pop()

    #     def load_model_hook(models, input_dir):
    #         if args.use_ema:
    #             load_model = EMAModel.from_pretrained(
    #                 os.path.join(input_dir, "unet_ema"), UNet2DConditionModel
    #             )
    #             ema_unet.load_state_dict(load_model.state_dict())
    #             ema_unet.to(accelerator.device)
    #             del load_model

    #         for i in range(len(models)):
    #             # pop models so that they are not loaded again
    #             model = models.pop()

    #             # load diffusers style into model
    #             load_model = UNet2DConditionModel.from_pretrained(
    #                 input_dir, subfolder="unet"
    #             )
    #             model.register_to_config(**load_model.config)

    #             model.load_state_dict(load_model.state_dict())
    #             del load_model

    #     accelerator.register_save_state_pre_hook(save_model_hook)
    #     accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer_parameters = filter(lambda p: p.requires_grad, unet.parameters()) if args.lora else unet.parameters()

    if args.pipeline == 'qformer':
        optimizer_parameters = list(optimizer_parameters) + list(latent_qformer.parameters())

    optimizer = optimizer_cls(
        optimizer_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(captions):
        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids

    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def collate_fn(examples):
        exo_pixel_values = None
        if args.use_exo:
            exo_pixel_values = torch.stack([example["exo_pixel_values"] for example in examples])
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        original_pixel_values = torch.stack([example["original_pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.cat([example["input_ids"] for example in examples])
        text = [example['text'] for example in examples]
        image = [example['image'] for example in examples]
        origin_image = [example['original_image'] for example in examples]
        return {"edited_pixel_values": pixel_values, "input_ids": input_ids, 'image':image, 'original_image':origin_image, 'text':text, 'exo_pixel_values':exo_pixel_values, 'original_pixel_values':original_pixel_values}
    
    def preprocess_func(image, text):
        return train_transforms(image), tokenize_captions(text)
    print("Loading dataset...")
    from dataset import Diffusion_Finetune_Dataset
    train_dataset = Diffusion_Finetune_Dataset(preprocess_func=preprocess_func, use_exo=args.use_exo, avg_exo=args.avg_exo, split='train', res=args.resolution)
    val_dataset = Diffusion_Finetune_Dataset(preprocess_func=preprocess_func, use_exo=args.use_exo, avg_exo=args.avg_exo, split='val', res=args.resolution)

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    if args.pipeline == 'qformer':
        unet, latent_qformer, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
            unet, latent_qformer, optimizer, train_dataloader,val_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader,val_dataloader, lr_scheduler
        )
    
    if args.use_ema:
        ema_unet.to(accelerator.device)

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("image2image", config=vars(args))

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            # dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps
            )

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if (
                args.resume_from_checkpoint
                and epoch == first_epoch
                and step < resume_step
            ):
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # We want to learn the denoising process w.r.t the edited images which
                # are conditioned on the original image (which was edited) and the edit instruction.
                # So, first, convert images to latent space.
                # [bs ,4, 64, 64]
                latents = vae.encode(
                    batch["edited_pixel_values"].to(weight_dtype)
                ).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning.
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Get the additional image embedding for conditioning.
                # Instead of getting a diagonal Gaussian here, we simply take the mode.
                
                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )
                
                model_pred = pipeline_method(unet, batch, args, vae, weight_dtype, bsz, latents, generator, text_encoder, tokenize_captions, accelerator, noisy_latents, timesteps, encoder_hidden_states, feature_extractor, image_encoder, image_normalizer, image_noising_scheduler, latent_qformer)

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    # if accelerator.num_processes == 1:
                    #     clip_param = filter(lambda p: p.requires_grad, unet.parameters()) if args.lora else unet.parameters()
                    #     accelerator.clip_grad_norm_(clip_param, args.max_grad_norm)
                    # else:
                    #     clip_param = filter(lambda p: p.requires_grad, unet.module.parameters()) if args.lora else unet.module.parameters()
                    #     accelerator.clip_grad_norm_(clip_param, args.max_grad_norm)
                    clip_param = filter(lambda p: p.requires_grad, unet.parameters()) if args.lora else unet.parameters()
                    accelerator.clip_grad_norm_(clip_param, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "epoch": epoch
            }
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
        
        if epoch % args.checkpointing_steps == 0:
            if accelerator.is_main_process:
                # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                if args.checkpoints_total_limit is not None:
                    checkpoints = os.listdir(args.output_dir)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                    if len(checkpoints) >= args.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]

                        logger.info(
                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                        )
                        logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)

                save_path = os.path.join(
                    args.output_dir, f"checkpoint-{global_step}-{epoch}"
                )
                accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")

                if args.lora:
                    unwrapped_unet = accelerator.unwrap_model(unet)
                    unet_lora_state_dict = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(unwrapped_unet)
                    )

                    pipeline_class.save_lora_weights(
                        save_directory=save_path,
                        unet_lora_layers=unet_lora_state_dict,
                        safe_serialization=True,
                    )
        
        if accelerator.is_main_process:
            if (
                epoch % args.validation_epochs == 0 and args.val
            ):
                logger.info(
                    f"Running validation... \n "
                )
                unet = accelerator.unwrap_model(unet)
                # create pipeline
                if args.use_ema:
                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                    ema_unet.store(unet.parameters())
                    ema_unet.copy_to(unet.parameters())

                pipeline = pipeline_class.from_pretrained(
                    pretrained_model_name_or_path,
                    unet=unet ,
                    revision=args.revision,
                    torch_dtype=weight_dtype,
                )
                # Deal with the issue oom
                device = accelerator.device
                pipeline = pipeline.to(device)
                pipeline.set_progress_bar_config(disable=True)

                # run inference
                # latent_qformer.eval()
                edited_images = []
                texts = []
                with torch.autocast(
                    'cuda',
                    enabled=accelerator.mixed_precision == "fp16",
                ):
                    with torch.no_grad():
                        for bn in tqdm(range(len(batch['text'][:10])), desc="Generating train images"):
                            
                            edited_image, original_image = pipeline_eval_method(args, batch, pipeline, generator, bn, latent_qformer)

                            h_concat = PIL.Image.new('RGB', (edited_image.width * 3, edited_image.height))
                            h_concat.paste(original_image, (0, 0))
                            h_concat.paste(edited_image, (edited_image.width, 0))
                            h_concat.paste(batch['image'][bn].resize((args.resolution, args.resolution)), (edited_image.width*2, 0))
                            edited_images.append(h_concat)
                            texts.append(batch['text'][bn])
                #  Log images to disk
                for img, prompt in zip(edited_images, texts):
                    os.makedirs(os.path.join(args.output_dir, 'vis', f'train_epoch[{epoch}]_step[{global_step}]'), exist_ok=True)
                    img.save(os.path.join(args.output_dir, 'vis', f'train_epoch[{epoch}]_step[{global_step}]', f"{prompt.replace(' ', '_')[:-1]}.png"))


                edited_images = []
                texts = []
                with torch.autocast(
                    'cuda',
                    enabled=accelerator.mixed_precision == "fp16",
                ):
                    for vbatch in (val_dataloader):
                        with torch.no_grad():
                            for bn in tqdm(range(len(vbatch['text'][:10])), desc="Generating val images"):

                                edited_image, original_image = pipeline_eval_method(args, vbatch, pipeline, generator, bn, latent_qformer)
                                
                                h_concat = PIL.Image.new('RGB', (edited_image.width * 3, edited_image.height))
                                h_concat.paste(original_image, (0, 0))
                                h_concat.paste(edited_image, (edited_image.width, 0))
                                h_concat.paste(vbatch['image'][bn].resize((args.resolution, args.resolution)), (edited_image.width*2, 0))
                                edited_images.append(h_concat)
                                texts.append(vbatch['text'][bn])
                            break
                #  Log images to disk
                for img, prompt in zip(edited_images, texts):
                    os.makedirs(os.path.join(args.output_dir, 'vis', f'val_epoch[{epoch}]_step[{global_step}]'), exist_ok=True)
                    img.save(os.path.join(args.output_dir, 'vis', f'val_epoch[{epoch}]_step[{global_step}]', f"{prompt.replace(' ', '_')[:-1]}.png"))            

                for tracker in accelerator.trackers:
                    if tracker.name == "wandb":
                        wandb_table = wandb.Table(columns=WANDB_TABLE_COL_NAMES)
                        for edited_image in edited_images:
                            wandb_table.add_data(
                                wandb.Image(original_image),
                                wandb.Image(edited_image),
                                args.validation_prompt,
                            )
                        tracker.log({"validation": wandb_table})
                if args.use_ema:
                    # Switch back to the original UNet parameters.
                    ema_unet.restore(unet.parameters())


                del pipeline
                torch.cuda.empty_cache()

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())


        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet ,
            revision=args.revision,
        )
        pipeline.save_pretrained(args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()

