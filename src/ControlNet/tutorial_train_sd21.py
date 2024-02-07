from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import os
import sys
# Abs file dir of this file
current_file_path = os.path.abspath(__file__)
# parent directory of this file
super_parent_directory = os.path.dirname(current_file_path)
parent_directory = os.path.dirname(super_parent_directory)
base_dir = os.path.dirname(parent_directory)
# print(base_dir)
sys.path.append(base_dir)
from src.dataset import ControlNet_Finetune_Dataset
NP = os.getenv("NP", 1)
NN = int(os.getenv("NN", 1))

opd = 'results/ControlNet/res256'
import torch
# Configs

batch_size = 1
logger_freq = 200
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('src/ControlNet/models/cldm_v21.yaml').cpu()

if torch.cuda.current_device() == 0:
    if os.path.exists(opd):
        versions = os.listdir(os.path.join(opd, 'lightning_logs'))
        versions.sort(key=lambda x: int(x.split('_')[-1]), reverse=True)
        ckpts = os.listdir(os.path.join(opd, 'lightning_logs', versions[0], 'checkpoints'))
        ckpts.sort(key=lambda x: int(x.split('-')[-1].split('=')[-1].split('.')[0]), reverse=True)
        try:
            resume_path = os.path.join(opd, 'lightning_logs', versions[0], 'checkpoints', ckpts[0])
        except:
            resume_path = None
    else:
        resume_path = None

    print('Resume from: ', resume_path)

    model.load_state_dict(load_state_dict('src/ControlNet/models/control_sd21_ini.ckpt', location='cpu'))

model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = ControlNet_Finetune_Dataset()
dataloader = DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=False)
val_dataset = ControlNet_Finetune_Dataset(split='val')
val_dataloader = DataLoader(val_dataset, num_workers=4, batch_size=batch_size, shuffle=False)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=NP, num_nodes=NN, resume_from_checkpoint=resume_path, precision=16, callbacks=[logger, pl.callbacks.ModelCheckpoint(every_n_train_steps=10000, save_top_k=-1)], enable_checkpointing=True, accumulate_grad_batches=1, default_root_dir=opd, strategy="ddp")
try:
    trainer.global_step = int(resume_path.split('/')[-1].split('-')[-1].split('=')[-1].split('.')[0])
except Exception as e:
    print(e)
    


# Train!
trainer.fit(model, dataloader, val_dataloader)
