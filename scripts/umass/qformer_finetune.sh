source activate minigpt5
python -m accelerate.commands.launch src/finetune_image_to_image.py \
  --pipeline qformer \
  --resolution=768 --center_crop \
  --from_scratch \
  --train_batch_size=16 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --use_exo \
  --checkpointing_steps 10 \
  --validation_epochs 10 \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir=results/diffusion/qformer_res758_train_woema_fromscratch \
  --report_to=tensorboard \
  --val \
  --resume_from_checkpoint latest \
   
