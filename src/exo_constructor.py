import av
from PIL import Image
from decord import VideoReader
from decord import cpu, gpu
import os
from tqdm import tqdm
import math

def extract_frames_from_video(video_path, output_dir, sample_per_second=1):
    output_path = os.path.join(output_dir, video_path.split('/')[-1].split('.')[0])
    os.makedirs(output_path, exist_ok=True)
    with open(video_path, 'rb') as f:
        vr = VideoReader(f, ctx=cpu(0))
    fps = vr.get_avg_fps()
    sample_interval_to_frame = math.ceil(fps / sample_per_second)
    for i in tqdm(range(0, len(vr), sample_interval_to_frame)):
        frame = Image.fromarray(vr[i].asnumpy())
        frame.save(output_path + f'/frame_{i}.png')  

if __name__ == '__main__':
    video_path = 'datasets/epic-kitchen/EK100_256p/P01/P01_08.MP4'
    output_dir = 'results/Exo/epic-kitchen/frames'
    extract_frames_from_video(video_path, output_dir, sample_per_second=20)