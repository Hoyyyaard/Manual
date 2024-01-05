import os
import json
from tqdm import tqdm
import cv2
from torch.utils.data import Dataset
from typing import Any, Optional, Dict, Sequence
from pathlib import Path
import imageio
from PIL import Image
import sys
import random
import numpy as np
import re
import pandas as pd
import clip
import argparse
from decord import VideoReader
from decord import cpu, gpu
import math
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.calibration import CameraCalibration, CameraModelType
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.image import InterpolationMethod
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import RecordableTypeId, StreamId
import numpy as np
from matplotlib import pyplot as plt
from projectaria_tools.core.sophus import SE3
import torch
import av


class FisheyeDistortor:
    def __init__(self):
        # Ugly implementation for fisheye distortion
        OUTPUT_FOV = 700
        provider = data_provider.create_vrs_data_provider('datasets/EgoExo4d/example.vrs')
        sensor_stream_id = provider.get_stream_id_from_label('camera-rgb')
        image_data = provider.get_image_data_by_index(sensor_stream_id, 0)
        device_calib = provider.get_device_calibration()
        self._src_calib = device_calib.get_camera_calib('camera-rgb')
        tmp_image_array = image_data[0].to_numpy_array()
        self._dst_calib = calibration.get_linear_camera_calibration(tmp_image_array.shape[1], tmp_image_array.shape[0], OUTPUT_FOV, 'camera-rgb')
    
    def __call__(self, frame):
        rectified_array = calibration.distort_by_calibration(frame, self._dst_calib, self._src_calib, InterpolationMethod.BILINEAR)
        return rectified_array
    

class KeyframeFilter:
    def __init__(self, device='cuda'):
        self._device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=self._device)
        self.cache_video_path = 'results/cache_video'
        os.makedirs(self.cache_video_path, exist_ok=True)
        self.cache_video_path = os.path.join(self.cache_video_path, 'tmp.mp4')
        
    def _calculate_sharp_score(self, images:np.ndarray):
        def np_softmax(x):
            e_x = np.exp(x - np.max(x))  
            return e_x / e_x.sum(axis=0)
        scores = []
        for img in images:
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            blurred_image = cv2.GaussianBlur(img, (3, 3), 0)
            laplacian_image = cv2.Laplacian(blurred_image, cv2.CV_64F)
            sharpness = np.var(laplacian_image)
            scores.append(sharpness)
        return np_softmax(np.array(scores))

    def _clip_filter(self, images:Image, text):
        c_images = [self.preprocess(img) for img in images]
        c_images = torch.stack(c_images).to(self._device)
        c_text = clip.tokenize(text).to(self._device)
        with torch.no_grad():
            logits_per_image, logits_per_text = self.model(c_images, c_text)
            probs = logits_per_text.softmax(dim=-1).cpu().numpy()
        return probs
    
    def _extract_keyframes(self, images_np, FPS):
        images_np = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in images_np]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(self.cache_video_path, fourcc, FPS, images_np[0].shape[:2][::-1])
        
        for image in images_np:
            video.write(image)
        video.release()
        
        frames = []
        frames_np = []
        positions = [] 
        with av.open(self.cache_video_path) as container:
            # 表示我们只想查看关键帧
            stream = container.streams.video[0]
            avfps = stream.duration / stream.frames
            stream.codec_context.skip_frame = 'NONKEY'
            for frame in container.decode(stream):
                # 使用frame.pts的原因是frame.index对skip_frame没有意义,因为关键帧是从所有的帧中抽取中独立的图像，而pts显示的就是这些独立图像的index；
                # DTS（Decoding Time Stamp）：即解码时间戳，这个时间戳的意义在于告诉播放器该在什么时候解码这一帧的数据。
                # PTS（Presentation Time Stamp）：即显示时间戳，这个时间戳用来告诉播放器该在什么时候显示这一帧的数据。
                kframe = frame.to_image().convert('RGB')
                kframe_np= frame.to_ndarray()
                frames.append(kframe)
                frames_np.append(kframe_np)
                positions.append(int(frame.pts / avfps))
        
        os.system(f'rm {self.cache_video_path}')
        
        return frames, frames_np, positions
    
    def __call__(self, images_np, images_pil, text, FPS) -> Any:
        
        key_frames, key_frames_np, positions = self._extract_keyframes(images_np, FPS)
        print(positions)
        # if len(key_frames) > 1:
        #     sharp_scores = self._calculate_sharp_score(key_frames_np)
        #     probs = self._clip_filter(key_frames, text)
        #     # Weight the two scores
        #     meaningful_image = key_frames[(probs+sharp_scores).argmax()]
        # else:
        #     meaningful_image = key_frames[0]
            
        return key_frames[0]