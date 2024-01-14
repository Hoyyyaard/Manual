export IS_STAGE2=False
export WEIGHTFOLDER=ckpts
export DATAFOLDER=datasets/CC3M
source activate minigpt5
/project/pi_chuangg_umass_edu/chenpeihao/miniconda3/envs/minigpt5/bin/python train_eval.py --is_training True \
                        --train_data_path cc3m_val.tsv \
                        --val_data_path cc3m_val.tsv \
                        --model_save_name stage1_cc3m_{epoch}-{step} \
                        --gpus $1 \
                        --output_dir results/official/$3 \
                        --per_device_train_batch_size $2 \
                        --num_train_epochs $4\