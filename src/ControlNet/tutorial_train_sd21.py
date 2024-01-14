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


# Configs
resume_path = 'src/ControlNet/models/control_sd21_ini.ckpt'
batch_size = 4
logger_freq = 200
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('src/ControlNet/models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = ControlNet_Finetune_Dataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=False)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=16, callbacks=[logger], enable_checkpointing=True, accumulate_grad_batches=4, default_root_dir='results/ControlNet/finetune_sd21')


# Train!
trainer.fit(model, dataloader)
