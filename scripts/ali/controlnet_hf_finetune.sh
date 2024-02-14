source activate minigpt5
python -m accelerate.commands.launch src/finetune_controlnet.py \
 --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
 --output_dir="results/controlnet_hf" \
 --resolution=256 \
 --learning_rate=1e-5 \
 --train_batch_size=32 \
 --num_train_epochs=100 \
 --tracker_project_name="controlnet" \
 --enable_xformers_memory_efficient_attention \
 --checkpointing_steps=10 \
 --validation_steps=10 \
 --report_to tensorboard \
 --resume_from_checkpoint latest \
 --checkpoints_total_limit 5 \