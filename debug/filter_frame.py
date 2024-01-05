import cv2
from io import BytesIO
import numpy as np
import os
import av

def debug(images_np):
    # 使用VideoWriter创建MP4文件
    frame_rate = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('tmp.mp4', fourcc, frame_rate, (images_np[0].shape[1], images_np[0].shape[0]))
    
    for image in images_np:
        video.write(image)
    video.release()
    
    frames = []
    with av.open('tmp.mp4') as container:
        # 表示我们只想查看关键帧
        stream = container.streams.video[0]
        stream.codec_context.skip_frame = 'NONKEY'
        for frame in container.decode(stream):
            # print(frame)
            # 使用frame.pts的原因是frame.index对skip_frame没有意义,因为关键帧是从所有的帧中抽取中独立的图像，而pts显示的就是这些独立图像的index；
            # DTS（Decoding Time Stamp）：即解码时间戳，这个时间戳的意义在于告诉播放器该在什么时候解码这一帧的数据。
            # PTS（Presentation Time Stamp）：即显示时间戳，这个时间戳用来告诉播放器该在什么时候显示这一帧的数据。
            frame = frame.to_image().convert('RGB')
            frames.append(frame)

    # print(f"{len(frames)}/{len(images_np)}")
    return(frames[0])