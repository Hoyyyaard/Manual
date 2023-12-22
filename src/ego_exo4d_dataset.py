import os
import json
from tqdm import tqdm
import cv2
import imageio
from PIL import Image


class EgoExo4d_Prerain_Dataset():
    '''
        Each episode is a pairs of action description and 
        corresponding exocentric views of a take.
    '''
    
    def __init__(self, split='val', data_root_dir='datasets/EgoExo4d', cooking_only=True, resolution=1200, preprocess=False) -> None:
        self.episodes = []
        self._data_root_dir = data_root_dir
        self._split = split
        self._cooking_only = cooking_only
        self._load_neccesary_data()
        
        if preprocess:
            print('INFO: [ Preprocess episodes and save ]')
            self._preprocess_episodes_and_save()
            
        print('INFO: [ Load preprocess episodes ]')
        self._load_episodes()
        pass
        
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
                
            
    def __getitem__(self, index):
        return self.episodes[index]
    
    def __len__(self):
        return len(self.episodes)   
    
    
if __name__ == '__main__':
    pretrain = EgoExo4d_Prerain_Dataset(split='val', preprocess=True)
    
    