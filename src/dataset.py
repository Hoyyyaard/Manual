import os
import json
from tqdm import tqdm
import cv2
from torch.utils.data import Dataset
from typing import Optional, Dict, Sequence
from pathlib import Path
import imageio
from PIL import Image
import sys
# Abs file dir of this file
current_file_path = os.path.abspath(__file__)
# parent directory of this file
parent_directory = os.path.dirname(current_file_path)
base_dir = os.path.dirname(parent_directory)
# print(base_dir)
sys.path.append(base_dir)
from constants import *
import random
import numpy as np
import re
import pandas as pd
import clip
import argparse
from decord import VideoReader
from decord import cpu, gpu
import math

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess EgoExo4d dataset')
    parser.add_argument('--data_path', type=str, default='datasets/EgoExo4d', help='Path to the dataset')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'], help='Split of the dataset')
    parser.add_argument('--cooking_only', action='store_true', help='Only use cooking tasks')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess dataset')
    parser.add_argument('--chunk', type=int, default=1, help='Chunk size for preprocessed dataset')
    parser.add_argument('--chunk_idx', type=int, default=0, help='Chunk index for preprocessed dataset')
    parser.add_argument('--filter_frame',action='store_true', help='Use clip to filter frames')
    args = parser.parse_args()
    return args

class Diffusion_Finetune_Dataset(Dataset):
    '''
        This dataset will leverage different preprocessed dataset:
            1. EgoExo4d_Finetune_Dataset
    '''
    def __init__(self, split='train', preprocess_func=None, dataset_list= ['epic']):
        self.EgoExo4d_Finetune_dataset_path = os.path.join('datasets', 'EgoExo4d', 'preprocessed_episodes_finetune', split)
        self.Epic_Kitchen_Text_Image_Pairs_dataset_path = os.path.join('datasets', 'epic-kitchen', 'text_image_pairs', split)
        self.preprocess_func = preprocess_func
        self.episodes = []
        
        if 'egoexo' in dataset_list:
            print("Loading EgoExo4d_Finetune_Dataset..")
            self.load_from_EgoExo4d_Finetune_Dataset()
        if 'epic' in dataset_list:
            saved_data_path = os.path.join(self.Epic_Kitchen_Text_Image_Pairs_dataset_path, 'image_text_pairs.pkl')
            if os.path.exists(saved_data_path):
                print("Loading saved data...")
                self.recover_data(saved_data_path)
                print("Loaded saved data for Epic_Kitchen_Text_Image_Pairs_Dataset!")
            else:
                print("Loading Epic_Kitchen_Text_Image_Pairs_Dataset..")
                self.load_from_Epic_Kitchen_Text_Image_Pairs_Dataset()
                self.save_process_data(saved_data_path)
    
    def recover_data(self, saved_file):
        all_data = torch.load(saved_file)
        self.episodes = all_data['episodes']
        del all_data
        
    def save_process_data(self, saved_file):
        all_data = {'episodes': self.episodes}
        torch.save(all_data, saved_file)
       
    def load_from_EgoExo4d_Finetune_Dataset(self, ):
        # Each episode will be in format {'image_path', 'caption'}
        for task_name in tqdm(os.listdir(self.EgoExo4d_Finetune_dataset_path), desc='Loading EgoExo4d_Finetune_Dataset'):
            take_path = os.path.join(self._dataset_path, task_name)
            with open(os.path.join(take_path, 'caption.json'), 'r') as f:
                data = json.load(f)
                captions = data['captions'] 
            image_paths = os.listdir(os.path.join(take_path, 'egocentric_images'))
            assert len(captions) == len(image_paths)
            image_paths = [os.path.join(os.path.join(take_path, 'egocentric_images'), p) for p in image_paths]
            for pa, ca in zip(image_paths, captions):
                self.episodes.append({'image_path':pa, 'caption':ca})
                
    def load_from_Epic_Kitchen_Text_Image_Pairs_Dataset(self, ):
        for video_index in tqdm(os.listdir(self.Epic_Kitchen_Text_Image_Pairs_dataset_path), desc='Loading Epic_Kitchen_Text_Image_Pairs_Dataset'):
            for index in (os.listdir(os.path.join(self.Epic_Kitchen_Text_Image_Pairs_dataset_path, video_index))):
                path = os.path.join(self.Epic_Kitchen_Text_Image_Pairs_dataset_path, video_index, index)
                with open(os.path.join(path, 'caption.txt'), 'r') as f:
                    caption = f.read()
                image_path = os.path.join(path, 'image.png')
                self.episodes.append({'image_path':image_path, 'caption':caption})
        
    
    def __getitem__(self, i):
        image_p, text = self.episodes[i]['image_path'], self.episodes[i]['caption']
        image = Image.open(image_p).convert("RGB")
        pixel_values, input_ids = self.preprocess_func(image, text)
        
        return {'pixel_values':pixel_values, 
                'input_ids':input_ids,
                'image':image,
                'text':text}
        
    def __len__(self):
        return len(self.episodes)   
        

class Epic_Kitchen_Text_Image_Pairs_Dataset(Dataset):
    def __init__(self, split='train', preprocess=False, chunk=None, chunk_idx=None, filter_frame=False) -> None:
        self._annotations_dir = 'datasets/epic-kitchen/epic-kitchens-100-annotations'
        self._raw_video_dir = 'datasets/epic-kitchen/EK100_256p'
        self._split = split
        self._save_dir = 'datasets/epic-kitchen/text_image_pairs'
        self._filter_frame = filter_frame
        
        if preprocess:
            self._device = "cuda" 
            self.model, self.preprocess = clip.load("ViT-B/32", device=self._device)
            self._preprocess_episodes_and_save()
    
    def _calculate_sharp_score(self, images):
        scores = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            blurred_image = cv2.GaussianBlur(img, (3, 3), 0)
            # 计算拉普拉斯算子
            laplacian_image = cv2.Laplacian(blurred_image, cv2.CV_64F)
            sharpness = np.var(laplacian_image)
            scores.append(sharpness)
        return np.array(scores)

    def _preprocess_episodes_and_save(self):
        def np_softmax(x):
            e_x = np.exp(x - np.max(x))  
            return e_x / e_x.sum(axis=0)
        
        SAMPLE_FRAME_NUM_PER_VIDEO_CLIP = 30
        
        df = pd.read_csv(os.path.join(self._annotations_dir, f'EPIC_100_{self._split}.csv'))
        self.anno_datas = {}
        total_pairs = 0
        # Load in format {'video_id': [[..], [..]]}
        for index, row in df.iterrows():
            video_id = row['video_id']
            narrations = row['narration']
            start_frame = row['start_frame']
            stop_frame = row['stop_frame']
            middle_frame = int((start_frame + stop_frame) // 2)
            if not video_id in self.anno_datas:
                self.anno_datas[video_id] = [{'narrations':narrations, 'start_frame':start_frame, 'stop_frame':stop_frame, 'middle_frame':middle_frame}] 
                # self.anno_datas[video_id] = []
            else:   
                self.anno_datas[video_id].append({'narrations':narrations, 'start_frame':start_frame, 'stop_frame':stop_frame, 'middle_frame':middle_frame})
        for video_id, v in tqdm(self.anno_datas.items(), desc='Loading video'):
            if os.path.exists(os.path.join(self._save_dir, self._split, video_id)):
                continue
            self.text_image_pairs = []
            base_video_name = video_id.split('_')[0]
            video_path = os.path.join(self._raw_video_dir, base_video_name, f'{video_id}.MP4')
            with open(video_path, 'rb') as f:
                vr = VideoReader(f, ctx=cpu(0))
            
            if self._filter_frame:
                try:
                    for narra in tqdm(v, desc='Loading video narrations'):
                        start_frame = narra['start_frame']
                        stop_frame = narra['stop_frame']
                        middle_frame = narra['middle_frame']
                        text = narra['narrations']
                        # FPS = vr.get_avg_fps() 
                        
                        sample_interval_to_frame = math.ceil(((stop_frame - start_frame)) / SAMPLE_FRAME_NUM_PER_VIDEO_CLIP)
                        images = vr.get_batch(range(start_frame, stop_frame, sample_interval_to_frame)).asnumpy()
                        # range_low = max(start_frame, middle_frame-int(SAMPLE_FRAME_NUM_PER_VIDEO_CLIP/2))
                        # range_high = min(stop_frame, middle_frame+int(SAMPLE_FRAME_NUM_PER_VIDEO_CLIP/2))
                        # images = vr.get_batch(range(range_low , range_high)).asnumpy()
                        
                        # Do sharpness filter
                        sharp_scores = self._calculate_sharp_score(images)
                        sharp_scores = np_softmax(sharp_scores)
                        # print(sharp_scores)
                        
                        images = [Image.fromarray((img)) for img in images]
                        
                        # save images and named as score
                        # for ii,im in enumerate(images):
                        #     im.save(os.path.join('results', f'{sharp_scores[ii]}.png'))
                        
                        # assert False
                        meaningful_image_test = images[sharp_scores.argmax()]  
                        
                        # Do clip similar selection
                        c_images = [self.preprocess(img) for img in images]
                        c_images = torch.stack(c_images).to(self._device)
                        c_text = clip.tokenize(text).to(self._device)
                        with torch.no_grad():
                            logits_per_image, logits_per_text = self.model(c_images, c_text)
                            probs = logits_per_text.softmax(dim=-1).cpu().numpy()
                            
                        # Weight the two scores
                        meaningful_image = images[(probs+sharp_scores).argmax()]   
                        # h_concat = Image.new('RGB', (meaningful_image.width * 2, meaningful_image.height))
                        # h_concat.paste(meaningful_image, (0, 0))
                        # h_concat.paste(meaningful_image_test, (meaningful_image.width, 0))
                        self.text_image_pairs.append({'text':text, 'image':meaningful_image})
                except Exception as e:
                    print(e)
                    continue

            else:
                middle_frames = [narra['middle_frame'] for narra in v]
                texts = [narra['narrations'] for narra in v]
                images = vr.get_batch(middle_frames).asnumpy()
                images = [Image.fromarray((img)) for img in images]
                for t, im in zip(texts, images):
                    self.text_image_pairs.append({'text':t, 'image':im})

            total_pairs += len(self.text_image_pairs)
            print('Total Text-Image Pairs num:', total_pairs)
            sbar = tqdm(total=len(self.text_image_pairs), desc='Saving text_image_pairs')
            for id, tip in enumerate(self.text_image_pairs):
                os.makedirs(sp := os.path.join(self._save_dir, self._split, video_id, str(id)), exist_ok=True)
                tip['image'].save(os.path.join(sp, 'image.png'))
                with open(os.path.join(sp, 'caption.txt'), 'w') as f:
                    f.write(tip['text'])
                    # f.write('\n')
                    # f.write(str(tip[f'score2{SAMPLE_FRAME_NUM_PER_VIDEO_CLIP}']))
                sbar.update(1)

class EgoExo4d_Finetune_Dataset(Dataset):
    def __init__(self, split='val', data_path='datasets/EgoExo4d', cooking_only=True, preprocess=False, input_processor=None, output_vis_processor=None, test=False, chunk=None, chunk_idx=None) -> None:
        self.episodes = []
        self._data_root_dir = data_path
        self._split = split
        self._cooking_only = cooking_only
        self.image_placehold = '<Img><ImageHere></Img>'
        self._chunk = chunk
        self._chunk_idx = chunk_idx
        
        
        if preprocess:
            self._device = "cuda" 
            self.model, self.preprocess = clip.load("ViT-B/32", device=self._device)
            self._load_neccesary_data()
            print('INFO: [ Preprocess episodes and save ]')
            self._preprocess_episodes_and_save()
            
    def _load_neccesary_data(self):
        with open(os.path.join(self._data_root_dir, 'takes.json'), 'r') as f:
            takes = json.load(f)
            self.taskname_uid = {}
            self.take_task = {}
            for tak in takes:
                self.taskname_uid[tak['take_name']] = tak['take_uid']
                self.take_task[tak['take_name']] = tak['task_name']

        with open(os.path.join(self._data_root_dir, 'annotations', f'keystep_{self._split}.json'), 'r') as f:
            key_frame_annotations_raw = json.load(f)['annotations']
            self.key_frame_annotations = {}
            for _,v in key_frame_annotations_raw.items():
                self.key_frame_annotations[v['take_uid']] = {
                    "take_name": v['take_name'],
                    # filter out non-essential segments
                    "key_frames" : [seg for seg in v['segments'] if seg['is_essential']]
                }
                
                
    def _preprocess_episodes_and_save(self):
        SAMPLE_FRAME_NUM_PER_SECOND = 10
        
        epi_save_dir = os.path.join(self._data_root_dir, 'preprocessed_episodes_finetune', self._split)
        total_takes = os.listdir(os.path.join(self._data_root_dir, 'takes'))
        total_takes.sort()
        tasks_num_per_chunk = math.ceil(len(total_takes)/self._chunk)
        total_takes = total_takes[self._chunk_idx*tasks_num_per_chunk:(self._chunk_idx+1)*tasks_num_per_chunk]
        print(f'INFO: [ Chunk num: {self._chunk}, Chunk idx: {self._chunk_idx}, Process tasks num from {self._chunk_idx*tasks_num_per_chunk} to {(self._chunk_idx+1)*tasks_num_per_chunk} ]')
        
        for take_name in tqdm(total_takes, desc='Loading takes'):
            if self._cooking_only and 'cooking' not in take_name:
                continue
            take_p = os.path.join(self._data_root_dir, 'takes', take_name)
            tak_save_dir = os.path.join(epi_save_dir, take_name)
            take_task = self.take_task[take_name]
            if os.path.exists(tak_save_dir):
                continue
            # os.makedirs(tak_save_dir, exist_ok=True)
            
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
                
            epi = {}
            epi['exocentric_images'] = []
            epi['egocentric_images'] = []
            epi['captions'] = []
            # Sample exo-centirc images only one time
            for k,v in video_captures.items():
                if 'ego' in k:
                    continue
                # 100 to avoid some shelter from first few frames
                v.set(cv2.CAP_PROP_POS_FRAMES, 100)
                ret, frame = v.read()
                assert ret
                epi['exocentric_images'].append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))) 
                
            # Load pretrain data : narrations and exocentric images
            if not self.taskname_uid[take_name] in self.key_frame_annotations:
                continue
            key_frame_annotations = self.key_frame_annotations[self.taskname_uid[take_name]]['key_frames']
            for anno in tqdm(key_frame_annotations,desc='Loading episode within a take'):
                ego_video_capture = video_captures['ego_rgb']
                images = []
                text = anno['step_description']
                
                # There will be caption duplication in one take
                if text in epi['captions']:
                    continue
                
                FPS = ego_video_capture.get(cv2.CAP_PROP_FPS)
                delta_time = math.ceil(anno['end_time'] - anno['start_time'])
                sample_total_frame = int(delta_time * SAMPLE_FRAME_NUM_PER_SECOND)  
                sample_interval_to_frame = int((delta_time * FPS) / sample_total_frame)
                for fra in range(int(FPS*anno['start_time']), int(FPS*anno['end_time']), sample_interval_to_frame):
                    ego_video_capture.set(cv2.CAP_PROP_POS_FRAMES, fra)
                    ret, frame = ego_video_capture.read()
                    assert ret
                    images.append((Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
                    
                # Do clip similar selection
                c_images = [self.preprocess(img) for img in images]
                c_images = torch.stack(c_images).to(self._device)
                c_text = clip.tokenize(text).to(self._device)
                with torch.no_grad():
                    logits_per_image, logits_per_text = self.model(c_images, c_text)
                    probs = logits_per_text.softmax(dim=-1).cpu().numpy()
                    meaningful_image = images[probs.argmax()]
                epi['egocentric_images'].append(meaningful_image)
                epi['captions'].append(text)
            
            for _, cap in video_captures.items():
                cap.release()
            
            ego_sp = os.path.join(tak_save_dir, 'egocentric_images')
            os.makedirs(ego_sp, exist_ok=True)
            for imgi in range(len(epi['egocentric_images'])):
                epi['egocentric_images'][imgi].save(os.path.join(ego_sp, f'{imgi}.png'))
            exo_sp = os.path.join(tak_save_dir, 'exocentric_images')
            os.makedirs(exo_sp, exist_ok=True)
            for imgi in range(len(epi['exocentric_images'])):
                epi['exocentric_images'][imgi].save(os.path.join(exo_sp, f'{imgi}.png'))
            with open(os.path.join(tak_save_dir, 'caption.json'), 'w') as f:
                json.dump({'task':take_task,'captions':epi['captions']}, f)
        
    
class EgoExo4d_Prerain_Dataset(Dataset):
    '''
        Each episode is a pairs of action description and 
        corresponding exocentric views of a take.
        
        Dataset Structure:
            - dataset_root_dir
                - split
                    - take_name
                        - frame_number
                            - caption.json
                                - task
                                - caption
                            - cam01.png
                            - cam02.png
                            - cam03.png
                            - cam04.png
                            - ego_rgb.png
    '''
    
    def __init__(self, split='val', data_path='datasets/EgoExo4d', cooking_only=True, preprocess=False, input_processor=None, output_vis_processor=None, test=False) -> None:
        self.episodes = []
        self._data_root_dir = data_path
        self._split = split
        self._cooking_only = cooking_only
        self.image_placehold = '<Img><ImageHere></Img>'
        
        if preprocess:
            self._load_neccesary_data()
            print('INFO: [ Preprocess episodes and save ]')
            self._preprocess_episodes_and_save()
            
        # print('INFO: [ Load preprocess episodes ]')
        # self._load_episodes()
        
        # Suit to Minigpt5 dataset format
        self.test = test
        self.load_preprocessed_image_features = False
        self.input_processor = input_processor
        self.output_vis_processor = output_vis_processor
        self.output_img_id = input_processor.tokenizer.convert_tokens_to_ids(ALL_IMG_TOKENS[0])
        
        saved_data_path = os.path.join(self._data_root_dir, 'stage1_pretrain_data.pkl')
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
                            output_image_path = (os.path.join(take_p, frame, cam))
                        elif 'caption' in cam:
                            with open(os.path.join(take_p, frame, cam), 'r') as f:
                                data = json.load(f)
                                step_caption = data['caption']

                    step_caption = self.pre_caption(step_caption)
                    # this_take_image_placehold = '<Img>' + self.image_placehold*len(input_image_path) + '</Img>'
                    this_take_image_placehold = self.image_placehold*len(input_image_path)
                    caption_source = f"{this_take_image_placehold}{step_caption}"
                    caption_target = f'{ALL_IMG_TOKENS_STR} ###'
                    self.sources.append(caption_source)
                    self.targets.append(caption_target)
                    self.caption.append(step_caption)
                    task = data['task']
                    self.task_names.append(f'{take_name}_{frame}_{task}')
                    self.input_image_path.append(input_image_path)
                    self.output_image_path.append(output_image_path)
                    
                    if i%100 == 0 and not test:
                        caption_source = f"###Human: {random.choice(generation_prompts)} {step_caption} ###Assistant:"
                        caption_source = system_prompt + caption_source
                        caption_target = f'{ALL_IMG_TOKENS_STR} ###'
                        self.sources.append(caption_source)
                        self.targets.append(caption_target)
                        self.caption.append(step_caption)
                        self.task_names.append(f'{take_name}_{frame}_{task}_instruction')
                        self.input_image_path.append(input_image_path)
                        self.output_image_path.append(output_image_path)
                    
            self.valid_idx = list(range(len(self.sources)))
            print("Saving data...")
            self.save_process_data(saved_data_path)
            print("Saved data for EgoExo4d!")
        if test:
            self.targets = self.caption
        
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
                # Cv2 capture or  decord reader
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

    def recover_data(self, saved_file):
        all_data = torch.load(saved_file)
        self.sources = all_data['sources']  # caption
        self.targets = all_data['targets']  # [IMG0][IMG1][IMG2][IMG3][IMG4][IMG5][IMG6][IMG7] ###
        self.input_image_path = all_data['input_image_path']   # [None]
        self.output_image_path = all_data['output_image_path'] # image path
        self.caption = all_data['caption']  # caption, the same as sources
        self.task_names = all_data['task_names']
        del all_data
        if self.test:
            self.valid_idx = []
            for i in range(len(self.targets)):
                if self.output_image_path[i] is not None:
                    self.valid_idx.append(i)
                
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if self.test:
            i = self.valid_idx[i]
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
        if self.test:
            return len(self.valid_idx)
        return len(self.sources)  
    
    @staticmethod
    def expand2square(pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result
    
if __name__ == '__main__':
    # pretrain = EgoExo4d_Prerain_Dataset(split='val', preprocess=True)
    args = parse_args()
    finetune = Epic_Kitchen_Text_Image_Pairs_Dataset(split=args.split, preprocess=args.preprocess, chunk=args.chunk, chunk_idx=args.chunk_idx, filter_frame=args.filter_frame)
