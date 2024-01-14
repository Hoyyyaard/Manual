export IS_STAGE2=False
export WEIGHTFOLDER=ckpts
export DATAFOLDER=datasets/EgoExo4d
export IS_MANUAL=True
export OUTPUT_FOLDER=results/stage1_manual/$2
export ONLY_TRAIN=True
source activate minigpt5
/project/pi_chuangg_umass_edu/chenpeihao/miniconda3/envs/minigpt5/bin/python train_eval.py --is_training True \
                        --train_data_path '' \
                        --val_data_path '' \
                        --check_generate_step 100 \
                        --model_save_name stage1_manual_${2}_{epoch}-{step} \
                        --gpus $1 \
                        --num_train_epochs 1000 \
                        --per_device_train_batch_size 4 \
                        --output_dir results/stage1_manual/$2 \
                        --stage1_weight stage1_cc3m.ckpt \
                        --sd_pipeline ckpts/huggingface/stable-diffusion-2-1-base \
                        # --real_batch_size $2 \
                        # --per_device_train_batch_size $3 --num_train_epochs $4 