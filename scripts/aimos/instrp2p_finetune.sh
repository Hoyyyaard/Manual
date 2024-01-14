export MODEL_NAME="timbrooks/instruct-pix2pix"
ulimit -n 102400
~/x64/anaconda3/envs/manual/bin/python -m accelerate.commands.launch src/finetune_instrp2p.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --use_exo \
  --checkpointing_steps 5 \
  --validation_epochs 5 \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir=$1 \
  --report_to=tensorboard \
  --val \
  --resume_from_checkpoint latest \
   