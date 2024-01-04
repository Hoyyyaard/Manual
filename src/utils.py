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
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.calibration import CameraCalibration, CameraModelType
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.image import InterpolationMethod
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import RecordableTypeId, StreamId
import numpy as np
from matplotlib import pyplot as plt
from projectaria_tools.core.sophus import SE3


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
        self._preprocess_episodes_and_save()
        
    def _calculate_sharp_score(self, images:np.ndarray):
        def np_softmax(x):
            e_x = np.exp(x - np.max(x))  
            return e_x / e_x.sum(axis=0)
        scores = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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
    
    def __call__(self, probs, sharp_scores, images_np, images_pil, text) -> Any:
        sharp_scores = self._calculate_sharp_score(images_np)
        probs = self._clip_filter(images_pil, text)
        # Weight the two scores
        meaningful_image = images_pil[(probs+sharp_scores).argmax()]
        
        return meaningful_image