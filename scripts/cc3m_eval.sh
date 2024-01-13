export IS_STAGE2=False
export WEIGHTFOLDER=ckpts
export DATAFOLDER=datasets/CC3M
export OUTPUT_FOLDER=results/CC3M
source activate minigpt5
/project/pi_chuangg_umass_edu/chenpeihao/miniconda3/envs/minigpt5/bin/python train_eval.py --test_data_path cc3m_val.tsv  --test_weight stage1_cc3m.ckpt --gpus 0 