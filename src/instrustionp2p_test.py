import argparse
import logging
import math
import os
from pathlib import Path
from typing import Optional

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
from diffusers import (AutoencoderKL, DDPMScheduler,
                       StableDiffusionInstructPix2PixPipeline,
                       UNet2DConditionModel)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel, AutoProcessor
import os 
from accelerate import dispatch_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        required=True,
    )
    args = parser.parse_args()
    return args

args = parse_args()

tokenizer = CLIPTokenizer.from_pretrained(
    'timbrooks/instruct-pix2pix',
    subfolder="tokenizer",
)
from dataset import Diffusion_Finetune_Dataset
train_transforms = transforms.Compose(
    [
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(512) ,
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)
def collate_fn(examples):
    exo_pixel_values = None
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
def tokenize_captions(captions):
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs.input_ids
train_dataset = Diffusion_Finetune_Dataset(preprocess_func=preprocess_func, use_exo=True)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    shuffle=True,
    collate_fn=collate_fn,
    batch_size=20,
    num_workers=0,
)

# unet = UNet2DConditionModel.from_pretrained(
#     'timbrooks/instruct-pix2pix',
#     subfolder="unet",
# )
# accelerator = Accelerator(mixed_precision='fp16')

# def load_model_hook(models, input_dir):

#     for i in range(len(models)):
#         # pop models so that they are not loaded again
#         model = models.pop()

#         # load diffusers style into model
#         load_model = UNet2DConditionModel.from_pretrained(
#             input_dir, subfolder="unet"
#         )
#         model.register_to_config(**load_model.config)

#         model.load_state_dict(load_model.state_dict())
#         del load_model

# accelerator.register_load_state_pre_hook(load_model_hook)

# unet = accelerator.prepare(unet)

# Get the most recent checkpoint
dirs = os.listdir(args.checkpoint_dir)
dirs = [d for d in dirs if d.startswith("checkpoint")]
dirs = sorted(dirs, key=lambda x: int(x.split('/')[-1].split("-")[1]))
path = dirs[-1] if len(dirs) > 0 else None
unet = UNet2DConditionModel.from_pretrained(
    os.path.join(args.checkpoint_dir, path),
    subfolder="unet",
)
# accelerator.print(f"Resuming from checkpoint {os.path.join(args.checkpoint_dir, path)}")
# accelerator.load_state(os.path.join(args.checkpoint_dir, path))

global_step = int(path.split('/')[-1].split("-")[1])
epoch = global_step = int(path.split('/')[-1].split("-")[2])

pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                'timbrooks/instruct-pix2pix',
                unet=unet.to(torch.float16) ,
                torch_dtype=torch.float16,
                device_map="auto"
            )
del unet
# pipeline = pipeline.to('cuda')


for batch in train_dataloader:
    with torch.no_grad():
        edited_images = []
        texts = []
        
        for bn in tqdm(range(len(batch['text'][:10])), desc="Generating val images"):
            
            original_image = batch['original_image'][bn]
            edited_image = (
                pipeline(
                    batch['text'][bn],
                    image=original_image,
                    num_inference_steps=20,
                    image_guidance_scale=1.5,
                    guidance_scale=7,
                ).images[0]
            )
            h_concat = PIL.Image.new('RGB', (edited_image.width * 2, edited_image.height))
            h_concat.paste(original_image, (0, 0))
            h_concat.paste(edited_image, (edited_image.width, 0))
            edited_images.append(h_concat)
            texts.append(batch['text'][bn])
    #  Log images to disk
    output_dir = args.checkpoint_dir
    for img, prompt in zip(edited_images, texts):
        os.makedirs(os.path.join(output_dir, 'vis', f'epoch{epoch}_step[{global_step}]'), exist_ok=True)
        img.save(os.path.join(output_dir, 'vis', f'epoch{epoch}_step[{global_step}]', f"{prompt.replace(' ', '_')}.png"))            
    
    
    break