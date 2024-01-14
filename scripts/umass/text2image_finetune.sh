export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"

accelerate launch src/finetune_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=48 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir=results/diffusion/text2image --report_to=tensorboard --validation_prompts="Make milk tea." --tracker_project_name="text2image-ego"       