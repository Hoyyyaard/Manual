export MODEL_NAME="timbrooks/instruct-pix2pix"
source activate minigpt5
python -m accelerate.commands.launch src/finetune_instrp2p.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --use_ema \
  --resolution=256 --center_crop \
  --train_batch_size=32 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --use_exo \
  --avg_exo \
  --checkpointing_steps 10 \
  --validation_epochs 10 \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir=results/diffusion/instrp2p_res256_train_avgexo \
  --report_to=tensorboard \
  --val \
  --resume_from_checkpoint latest \
   
