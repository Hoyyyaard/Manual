import os
import json
from tqdm import tqdm
import cv2
from torch.utils.data import Dataset
from typing import Optional, Dict, Sequence
from pathlib import Path
import imageio
from PIL import Image
from constants import *
import random
import numpy as np
import re

class EgoExo4d_Prerain_Dataset(Dataset):
    '''
        Each episode is a pairs of action description and 
        corresponding exocentric views of a take.
    '''
    
    def __init__(self, split='val', data_path='datasets/EgoExo4d', cooking_only=True, preprocess=False, input_processor=None, output_vis_processor=None, test=False) -> None:
        self.episodes = []
        self._data_root_dir = data_path
        self._split = split
        self._cooking_only = cooking_only
        
        if preprocess:
            self._load_neccesary_data()
            print('INFO: [ Preprocess episodes and save ]')
            self._preprocess_episodes_and_save()
            
        # print('INFO: [ Load preprocess episodes ]')
        # self._load_episodes()
        
        # Suit to Minigpt5 dataset format
        self.test = test
        self.input_processor = input_processor
        self.output_vis_processor = output_vis_processor
        self.output_img_id = input_processor.tokenizer.convert_tokens_to_ids(ALL_IMG_TOKENS[0])
        
        saved_data_path = os.path.join(self._data_root_dir, 'stage1.pkl')
        if os.path.exists(saved_data_path):
            print("Loading saved data...")
            self.recover_data(saved_data_path)
            print("Loaded saved data for EgoExo4d!")
        else:
            self.sources, self.targets, self.input_image_path, self.output_image_path = [], [], [], []
            self.caption, self.task_names = [], []

            system_prompt="You will be able to generate image according to command."
            generation_prompts = [
                "generate image with caption:",
                "can you give me the image with caption:",
                "help me to generate this image:",
                "generate image with according to caption:",
                "according to caption, generate image:",
                "an image with caption:",
                "can you visualize this caption:",
            ]

            epi_save_dir = os.path.join(self._data_root_dir, 'preprocessed_episodes', self._split)
            for i,take_name in tqdm(enumerate(os.listdir(epi_save_dir)), desc='Loading takes'):
                if self._cooking_only and 'cooking' not in take_name:
                    continue
                take_p = os.path.join(epi_save_dir, take_name)
                for frame in tqdm(os.listdir(take_p), desc='Loading episode within a take'):
                    input_image_path = []
                    for cam in os.listdir(os.path.join(take_p, frame)):
                        if 'cam' in cam:
                            input_image_path.append(os.path.join(take_p, frame, cam))
                        elif 'ego_rgb' in cam:
                            self.output_image_path.append(os.path.join(take_p, frame, cam))
                        elif 'caption' in cam:
                            with open(os.path.join(take_p, frame, cam), 'r') as f:
                                data = json.load(f)
                                step_caption = data['caption']

                    step_caption = self.pre_caption(step_caption)
                    caption_source = f"{step_caption}"
                    caption_target = f'{ALL_IMG_TOKENS_STR} ###'
                    self.sources.append(caption_source)
                    self.targets.append(caption_target)
                    self.caption.append(step_caption)
                    task = data['task']
                    self.task_names.append(f'{take_name}_{frame}_{task}')
                    self.input_image_path.append(input_image_path)
                    
                    # if i%100 == 0 and not test:
                    #     caption_source = f"###Human: {random.choice(generation_prompts)} {step_caption} ###Assistant:"
                    #     caption_source = system_prompt + caption_source
                    #     caption_target = f'{ALL_IMG_TOKENS_STR} ###'
                    #     self.sources.append(caption_source)
                    #     self.targets.append(caption_target)
                    #     self.caption.append(step_caption)
                    #     self.task_names.append(f'cc3m_{i}_instruction')

                        
            self.valid_idx = list(range(len(self.sources)))
            print("Saving data...")
            self.save_process_data(saved_data_path)
            print("Saved data for EgoExo4d!")
        
    def _load_episodes(self):
        epi_save_dir = os.path.join(self._data_root_dir, 'preprocessed_episodes', self._split)
        for take_name in tqdm(os.listdir(epi_save_dir), desc='Loading takes'):
            if self._cooking_only and 'cooking' not in take_name:
                continue
            take_p = os.path.join(epi_save_dir, take_name)
            for frame in tqdm(os.listdir(take_p), desc='Loading episode within a take'):
                epi = {'take':take_name}
                epi['frame'] = int(frame)
                for cam in os.listdir(os.path.join(take_p, frame)):
                    if 'cam' in cam:
                        # epi[cam] = cv2.imread(os.path.join(take_p, frame, cam))
                        epi[cam] = Image.open(os.path.join(take_p, frame, cam))
                    elif 'ego_rgb' in cam:
                        # epi[cam] = cv2.imread(os.path.join(take_p, frame, cam))
                        epi[cam] = Image.open(os.path.join(take_p, frame, cam))
                    elif 'caption' in cam:
                        with open(os.path.join(take_p, frame, cam), 'r') as f:
                            data = json.load(f)
                            epi['caption'] = data['caption']
                            epi['task'] = data['task']
                self.episodes.append(epi)
    
    def _load_neccesary_data(self):
        with open(os.path.join(self._data_root_dir, 'takes.json'), 'r') as f:
            takes = json.load(f)
            self.taskname_uid = {}
            self.take_task = {}
            for tak in takes:
                self.taskname_uid[tak['take_name']] = tak['take_uid']
                self.take_task[tak['take_name']] = tak['task_name']

        with open(os.path.join(self._data_root_dir, 'annotations', f'atomic_descriptions_{self._split}.json'), 'r') as f:
            self.narrations = json.load(f)['annotations']
            for k,v in self.narrations.items():
                self.narrations[k] = v[0]['descriptions']
    
    def _preprocess_episodes_and_save(self):
        
        epi_save_dir = os.path.join(self._data_root_dir, 'preprocessed_episodes', self._split)
        
        for take_name in tqdm(os.listdir(os.path.join(self._data_root_dir, 'takes')), desc='Loading takes'):
            if self._cooking_only and 'cooking' not in take_name:
                continue
            take_p = os.path.join(self._data_root_dir, 'takes', take_name)
            tak_save_dir = os.path.join(epi_save_dir, take_name)
            take_task = self.take_task[take_name]
            if os.path.exists(tak_save_dir):
                continue
            # os.makedirs(tak_save_dir, exist_ok=True)
            self.save_episodes = []
            
            # Load video captures
            frame_aligned_videos_p = os.path.join(take_p, 'frame_aligned_videos')
            ## Filter slam left/right and et mp4
            filters = ['aria01_211-1.mp4', 'aria01_1201-1.mp4', 'aria01_1201-2.mp4']
            video_captures = {}
            for mp4 in os.listdir(frame_aligned_videos_p):
                if mp4 in filters:
                    continue
                mp4_p = os.path.join(frame_aligned_videos_p, mp4)
                video_name = mp4.split('.')[0] if not 'aria' in mp4 else 'ego_rgb'
                # Cv2 capture or imageio reader
                video_captures.update({video_name: cv2.VideoCapture(mp4_p)})
                # video_captures.update({video_name: imageio.get_reader(mp4_p,'ffmpeg')})
            
            # Load pretrain data : narrations and exocentric images
            if not self.taskname_uid[take_name] in self.narrations:
                continue
            narrations = self.narrations[self.taskname_uid[take_name]]
            for na in tqdm(narrations,desc='Loading episode within a take'):
                epi = {}
                ## Remove 'C ' in 'C picks a pot in the kitchen steel rack with his right hand.'
                ## and upper the first letter of the sentence
                epi['caption'] = na['text'].replace(na['text'][:3], na['text'][2].upper())
                for k,v in video_captures.items():
                    FPS = v.get(cv2.CAP_PROP_FPS)
                    frame_num = int(na['timestamp'] * FPS)
                    v.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, frame = v.read()
                    assert ret
                    epi[k] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # FPS = v.get_meta_data()['fps']
                    # frame_num = int(na['timestamp'] * FPS)
                    # datas = v.get_data(frame_num)
                    # frame = Image.fromarray(datas)
                    
                    epi[k] = frame
                    epi['frame'] = frame_num
                self.save_episodes.append(epi)   
            
            for _, cap in video_captures.items():
                cap.release()
            
            for epi in tqdm(self.save_episodes, desc='Saving episode within a take'):
                sp = os.path.join(tak_save_dir, str(epi['frame']))
                os.makedirs(sp, exist_ok=True)
                cv2.imwrite(os.path.join(sp, 'ego_rgb.png'), epi['ego_rgb'])
                # epi['ego_rgb'].save(os.path.join(sp, 'ego_rgb.png'))    
                for k,v in epi.items():
                    if 'cam' in k:
                        cv2.imwrite(os.path.join(sp, f'{k}.png'), v)
                        # v.sava(os.path.join(sp, f'{k}.png'))
                with open(os.path.join(sp, 'caption.json'), 'w') as f:
                    json.dump({'task':take_task,'caption':epi['caption']}, f)

    def pre_caption(self, caption):
        
        caption = re.sub(
            r"([.!\"()*#:;~])",
            " ",
            caption.lower(),
        )
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # truncate caption
        # max_words = 100
        # caption_words = caption.split(" ")
        # if len(caption_words) > max_words:
        #     caption = " ".join(caption_words[: max_words])

        return caption

    def save_process_data(self, saved_file):
        all_data = {'sources': self.sources,
                    'targets': self.targets,
                    'input_image_path': self.input_image_path,
                    'output_image_path': self.output_image_path,
                    'caption': self.caption,
                    'task_names': self.task_names,
                    }
        torch.save(all_data, saved_file)
                
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        input_image_path = self.input_image_path[i]
        output_image_path = self.output_image_path[i]
        input_text = self.sources[i]
        output_text = self.targets[i]
        if self.load_preprocessed_image_features and PREPROCESS_FEATURE_FOLDER is not None and os.path.isdir(PREPROCESS_FEATURE_FOLDER):
            if output_image_path is not None:
                output_feature_name = Path(output_image_path).name
                output_feature_name = output_feature_name.replace('.jpg', '_output.pt')
                if 'val' in output_image_path:
                    output_feature_path = Path(PREPROCESS_FEATURE_FOLDER).joinpath('val', output_feature_name)
                elif 'train' in output_image_path:
                    output_feature_path = Path(PREPROCESS_FEATURE_FOLDER).joinpath('train', output_feature_name)
                output_image_feature = torch.load(output_feature_path).unsqueeze(0)
            else:
                output_image_path = 'none'
                output_image_feature = torch.zeros((1, 8, 64, 64))
            
            input_images_feature = []
            for in_img_path in input_image_path:
                if in_img_path is not None:
                    input_feature_name = Path(in_img_path).name
                    input_feature_name = input_feature_name.replace('.jpg', '_input.pt')
                    if 'val' in in_img_path:
                        input_feature_path = Path(PREPROCESS_FEATURE_FOLDER).joinpath('val', input_feature_name)
                    elif 'train' in in_img_path:
                        input_feature_path = Path(PREPROCESS_FEATURE_FOLDER).joinpath('train', input_feature_name)
                    input_image_feature = torch.load(input_feature_path).unsqueeze(0)
                else:
                    input_image_feature = torch.zeros((1, 32, 4096))
                input_images_feature.append(input_image_feature)
            input_images_feature = torch.cat(input_images_feature, dim=0)
            input_dict = self.input_processor(text = input_text, add_special_tokens=False)
            input_dict['input_images_feature'] = input_images_feature
            input_dict['output_image_feature'] = output_image_feature
        else:
            input_images = []
            for in_img_path in input_image_path:
                if in_img_path is not None:
                    input_image = Image.open(in_img_path).convert("RGB")
                else:
                    input_image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
                input_images.append(input_image)
            input_dict = self.input_processor(text = input_text, images = input_images, add_special_tokens=False)
            input_dict['original_images'] = input_images
            
            if output_image_path is not None:
                output_image = Image.open(output_image_path).convert("RGB")
                output_image = self.expand2square(output_image, (255, 255, 255))
                output_image = self.output_vis_processor(output_image)
                output_image = output_image.unsqueeze(0)
            else:
                output_image_path = 'none'
                output_image = torch.zeros((1, 3, 512, 512))
            input_dict["output_image"] = output_image

        input_dict["caption"] = self.caption[i]
        input_dict["task_name"] = self.task_names[i]
        target_ids = self.input_processor(text = output_text, add_special_tokens=False)['input_ids']
        label = torch.ones_like(input_dict["input_ids"])*-100
        label = torch.cat((label, target_ids), dim=1)
        index = torch.nonzero(label == self.output_img_id)
        if len(index):
            index = index[0,1]
            label[:, index+1:index+IMG_TOKEN_NUM-1] = -100
        input_dict["labels"] = label
        input_dict["input_ids"] = torch.cat((input_dict["input_ids"], target_ids), dim=1)
        input_dict["attention_mask"] = torch.cat((input_dict["attention_mask"], torch.ones_like(target_ids)), dim=1)
        input_dict["source"] = input_text
        input_dict["target"] = output_text

        return input_dict
    
    def __len__(self):
        return len(self.episodes)   
    
    
if __name__ == '__main__':
    pretrain = EgoExo4d_Prerain_Dataset(split='val', preprocess=True)
    
    