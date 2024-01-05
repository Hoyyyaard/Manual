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
    # args = parse_args()
    # assert args.chunk_id < args.chunk
    # print(args)
    # filter = Filter(args)
    # filter.filter_dataset()
    import cv2
    import numpy as np
    from projectaria_tools.core import data_provider, calibration
    from projectaria_tools.core.calibration import CameraCalibration, CameraModelType
    from projectaria_tools.core import data_provider, calibration
    from projectaria_tools.core.image import InterpolationMethod
    from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
    from projectaria_tools.core.stream_id import RecordableTypeId, StreamId
    import numpy as np
    from matplotlib import pyplot as plt
    from projectaria_tools.core.sophus import SE3
    
    provider = data_provider.create_vrs_data_provider('/project/pi_chuangg_umass_edu/chenpeihao/Projects/hongyanzhi/MiniGPT-5/datasets/EgoExo4d/captures/videos/aria01.vrs')
    
    camera_name = "camera-rgb"
    sensor_name = "camera-rgb"
    sensor_stream_id = provider.get_stream_id_from_label(sensor_name)
    image_data = provider.get_image_data_by_index(sensor_stream_id, 0)
    image_array = image_data[0].to_numpy_array()
    # input: retrieve image distortion
    device_calib = provider.get_device_calibration()
    src_calib = device_calib.get_camera_calib(sensor_name)

    # create output calibration: a linear model of image size 512x512 and focal length 150
    # Invisible pixels are shown as black.
    dst_calib = calibration.get_linear_camera_calibration(512, 512, 150, camera_name)

    # distort image
    rectified_array = calibration.distort_by_calibration(image_array, dst_calib, src_calib, InterpolationMethod.BILINEAR)
    
    # class CameraModelType:
    # """
    # Enum that represents the type of camera projection model. See Linear.h, Spherical.h, KannalaBrandtK3.h and FisheyeRadTanThinPrism.h for details.
    
    # Members:
    
    #   KANNALA_BRANDT_K3 : Spherical + polynomial radial distortion up to 9-th order.
    
    #   FISHEYE624 : Spherical + polynomial radial distortion up to 11-th order + tangential distortion.
    
    #   SPHERICAL : Spherical projection, linear in angular space.
    
    #   LINEAR : Linear pinhole projection, unit plane points and camera pixels are linearly related.
    # """

    # class CameraCalibration:
    # """
    # A class that provides APIs for camera calibration, including extrinsics, intrinsics, and projection.
    # """
    # @typing.overload
    # def __init__(self) -> None:
    #     ...
    # @typing.overload
    # def __init__(self, arg0: str, arg1: CameraModelType, arg2: numpy.ndarray[numpy.float64[m, 1]], arg3: SE3, arg4: int, arg5: int, arg6: float | None, arg7: float, arg8: str) -> None:
    #     """
    #     Constructor with a list of parameters for CameraCalibration.
    #       Args:
    #         label: The label of the camera, e.g. "camera-slam-left".
    #         projection_model_type The type of camera projection model, e.g. ModelType::Linear
    #         T_Device_Camera: The extrinsics of camera in Device frame.
    #         image_width: Width of camera image.
    #         image_height: Height of camera image.
    #         maybe_valid_radius: [optional] radius of a circular mask that represents the valid area on
    #                 the camera's sensor plane. Pixels out of this circular region are considered invalid. Setting
    #                 this to None means the entire sensor plane is valid.
    #         max_solid_angle an angle theta representing the FOV cone of the camera. Rays out of
    #                 [-theta, +theta] will be rejected during projection.
    #     """
    
    
    
    # {"Calibrated":true,"Projection":{"Params":[1220.023996423738,1465.740842972049,1444.456400131055,0.3873017203659512,-0.3028693288398382,-0.4233073492385598,1.992695711098763,-2.275773876360815,0.7960163711144228,0.0002886837163548054,0.0002861485206749526,-0.0006876592045433987,0.0001215696733149477,-0.001279450308854052,0.0002554384029135424],"Description":"see FisheyeRadTanThinPrism.h","Name":"FisheyeRadTanThinPrism"},"T_Device_Camera":{"Translation":[-0.004473207879487903,-0.01184645971386183,-0.004889051154160274],"UnitQuaternion":[0.9413135313012251,[0.33344618193065784,0.037802929278451275,0.03624111040038306]]},"SerialNumber":"0450577b730611814401100000000000","Label":"camera-rgb"}
    # label: camera-rgb, model name: Fisheye624, principal point: [714.296, 707.294], focal length: [610.01, 610.01], projection params: [610.01, 714.296, 707.294, 0.411036, -0.456468, 0.0383031, 1.29834, -1.77047, 0.654994, -7.92031e-05, 4.4207e-05, -0.000587384, 0.000564394, 0.000515795, 0.000235027], image size (w,h): [1408, 1408], T_Device_Camera:(translation:[-0.00400587, -0.0118698, -0.00441815], quaternion(x,y,z,w):[0.332484, 0.0339686, 0.0416556, 0.941576]), serialNumber:0450577b730401974401100000000000)
    # load video by cv2.VideoCapture
    cap = cv2.VideoCapture('/project/pi_chuangg_umass_edu/chenpeihao/Projects/hongyanzhi/MiniGPT-5/datasets/EgoExo4d/takes/fair_cooking_05_2/frame_aligned_videos/aria02_214-1.mp4')
    ret, frame = cap.read()
    
    # 摄像机内部参数矩阵
    K = np.array([[242.52078247070312, 0, 318.243408203125],
                [0, 242.52078247070312, 240.81936645507812],
                [0, 0, 1]])

     # [[k_0: k_5]  {p_0 p_1} {s_0 s_1 s_2 s_3}]
    D = np.array([-0.024876972660422325, 0.09810236096382141, -0.06655783951282501, 0.008734318427741528, 0.0025442445185035467, -0.0005746951792389154, -0.0003363479918334633, 1.1586302207433619e-05, -0.0005165732582099736, -5.950118065811694e-05, 0.00037607509875670075, -1.358488134428626e-05])
    
    DD = np.array([1220.023996423738,1465.740842972049,1444.456400131055,0.3873017203659512,-0.3028693288398382,-0.4233073492385598,1.992695711098763,-2.275773876360815,0.7960163711144228,0.0002886837163548054,0.0002861485206749526,-0.0006876592045433987,0.0001215696733149477,-0.001279450308854052,0.0002554384029135424])
    
    se3_instance = SE3.from_quat_and_translation(0.9413135313012251, np.array([0.33344618193065784,0.037802929278451275,0.03624111040038306]), np.array([-0.004473207879487903,-0.01184645971386183,-0.004889051154160274]))
    se3_instance = SE3.from_quat_and_translation(1, np.array([0.,0.,0.]), np.array([0.,0.,0.]))
    # 2. _core_pybinds.calibration.CameraCalibration(arg0: str, arg1: _core_pybinds.calibration.CameraModelType, arg2: numpy.ndarray[numpy.float64[m, 1]], arg3: SE3, arg4: int, arg5: int, arg6: Optional[float], arg7: float, arg8: str)

    
   

    # 输入图像宽度和高度
    image_width = frame.shape[1]
    image_height = frame.shape[0]
    
    # src_calib = CameraCalibration('camera-rgb', CameraModelType.FISHEYE624, DD , se3_instance, image_width, image_height, None, 0, '0450577b730401974401100000000000')
    
    dst_calib = calibration.get_linear_camera_calibration(image_width, image_height, 700, camera_name)
    print(src_calib)
    # distort image
    rectified_array = calibration.distort_by_calibration(frame, dst_calib, src_calib, InterpolationMethod.BILINEAR)

    # 初始化映射矩阵
    # mapx, mapy = cv2.fisheye.initUndistortRectifyMap(K, D, None, K, (image_width, image_height), cv2.CV_32FC1)
    # corrected_image = cv2.remap(frame, mapx, mapy, interpolation=cv2.INTER_LINEAR)
    
    cv2.imwrite('results/corrected_image.png', rectified_array)
    cv2.imwrite('results/frame.png', frame)
    