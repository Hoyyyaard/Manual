import torch
import clip
from PIL import Image
import argparse
import os 
import json
from tqdm import tqdm
import numpy as np
import math

def parse_args():
    parser = argparse.ArgumentParser(description='Filter out the meaningful frames from the video clip')
    parser.add_argument('--output_path', type=str, help='Path to the output directory')
    parser.add_argument('--chunk', type=int, required=True, help='Processes to run in parallel')
    parser.add_argument('--chunk_id', type=int, required=True, help='Process id')
    parser.add_argument('--split', default='val', type=str, choices=['train', 'val', 'test'])
    parser.add_argument('--dataset_name', default='egoexo4d_pretrain', type=str, choices=['egoexo4d_pretrain', 'egoexo4d_finetune', 'epic_kitchens_finetune'])
    parser.add_argument('--data_root_dir', default='datasets/EgoExo4d', type=str)
    parser.add_argument('--cooking_only', action='store_true', default=True, help='Only process the cooking episodes')
    args = parser.parse_args()
    return args


class Filter:
    '''
        As always a narration corresponds to a video clip in EgoExo4d or Epic-Kitchens
        This class is used to filter out the meaningful frames from the video clip.
        
        This class will load from the preprocess frame from datasets.
        
        There are three requirements:
        1. [EgoExo4d Pretrain dataset] 
            A. use the narrations, ego-frame and exo-frames
                but sometimes ego-frame will be meaningless, so we need to filter out the 
                meaningless frames (TODO:how)
            B. output the filter signal to the caption.json file : high_quality: True/False
        2. [EgoExo4d Fine-tune dataset] use the key-frames, but one key-annotation 
        corresponds to a short video clip, so we need to filter out the meaningful frame
        use clip text-image similarity score.
        3. [Epic-kitchen Fine-tune dataset] use the key-frames, but one key-annotation 
        corresponds to a short video clip, so we need to filter out the meaningful frame
        use clip text-image similarity score.
    '''
    
    def __init__(self, args):
        self._device = "cuda" 
        self.model, self.preprocess = clip.load("ViT-B/32", device=self._device)
        self._args = args
        self._split = args.split
        self._dataset_name = args.dataset_name  
        self._data_root_dir = args.data_root_dir 
        self._cooking_only = args.cooking_only
        
    def filter_dataset(self):
        return getattr(self, f'_filter_{self._dataset_name}_dataset')()
    
    def _filter_egoexo4d_pretrain_dataset(self):
        # How many frames and texts will construct to a batch feed to the clip model
        PROCESS_FRAME_PER_CLIP_TURN = 2000
        HIGH_QUALITY_THRESHOLD = 0.7
        epi_save_dir = os.path.join(self._data_root_dir, 'preprocessed_episodes', self._split)
        
        # Do chunk filter
        num_per_chunk = int(len(total_epis:=os.listdir(epi_save_dir)) // self._args.chunk)
        total_epis = total_epis[self._args.chunk_id*num_per_chunk:(self._args.chunk_id+1)*num_per_chunk]
        
        for take_name in tqdm(total_epis, desc='filter takes'):
            take_episodes = []
            hight_quality_index = []
            if self._cooking_only and 'cooking' not in take_name:
                continue
            take_p = os.path.join(epi_save_dir, take_name)
            # for frame in tqdm(os.listdir(take_p), desc='filter episode within a take'):
            for frame in os.listdir(take_p):
                ego_rgb = os.path.join(take_p, frame, 'ego_rgb.png')
                with open(os.path.join(take_p, frame, 'caption.json'), 'r') as f:
                    step_caption = json.load(f)['caption']
                take_episodes.append({'image':ego_rgb, 'text':step_caption, 'frame_num':frame})
            # Do filter in batch
            process_rounds = math.ceil(len(take_episodes) / PROCESS_FRAME_PER_CLIP_TURN)
            for r in range(process_rounds):
                chunk_take_epi = take_episodes[r*PROCESS_FRAME_PER_CLIP_TURN:(r+1)*PROCESS_FRAME_PER_CLIP_TURN]
                images = [self.preprocess(Image.open(te['image'])) for te in chunk_take_epi]
                images = torch.stack(images).to(self._device)
                texts = [te['text'] for te in chunk_take_epi]
                texts = clip.tokenize(texts).to(self._device)
                with torch.no_grad():
                    image_features = self.model.encode_image(images)
                    text_features = self.model.encode_text(texts)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    # FIXME: May be similarity score cannot filter out
                    # if this frame is meaningful or not
                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu()
                    scores = torch.diag(similarity)
                    for sc in scores:
                        if sc > HIGH_QUALITY_THRESHOLD:
                            hight_quality_index.append(True)
                        else:
                            hight_quality_index.append(False)
            assert len(hight_quality_index) == len(take_episodes)
            # Save the filter signal to the caption.json file
            for i,te in enumerate(chunk_take_epi):
                sp = os.path.join(take_p, te['frame_num'], 'caption.json')
                with open(sp, 'r') as f:
                    data = json.load(f)
                data['high_quality'] = hight_quality_index[i]
                with open(sp, 'w') as f:
                    json.dump(data, f)
                
    def _filter_egoexo4d_finetune_dataset(self):
        pass
    
    def _filter_epic_kitchens_finetune_dataset(self):
        pass
    
if __name__ == '__main__':
    args = parse_args()
    assert args.chunk_id < args.chunk
    print(args)
    filter = Filter(args)
    filter.filter_dataset()