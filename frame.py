import os
import cv2
import ffmpeg
import numpy as np
from PIL import Image # pillow-simd
from decord import VideoReader
from decord import gpu
import utils

def extract_gpu(index, video_path, frame_path, frame_size, quality, origin_size, aspect_ratio, batch_size):
    # get filename and make a save directory
    filename, frame_path = utils.get_filename_frame_path(video_path, frame_path)

    # load a video
    video_reader = VideoReader(video_path, ctx=gpu(0))

    # get length info from video
    length = len(video_reader)

    if length == 0:
        message = "[ERROR] falied to read a video from '{}'".format(video_path)
        raise Exception(message)

    # message
    print("{}/{} name: {} length: {}".format(index[0]+1, index[1], filename, length))

    # read and save
    index = 0
    for i in range(length//batch_size if (length%batch_size) == 0 else (length//batch_size) + 1):
        frames = video_reader.get_batch(list(range(i*batch_size, (i+1)*batch_size if (length%batch_size) == 0 else i*batch_size + (length%batch_size)))).asnumpy()
        for frame in frames:
            height, width, _ = frame.shape
            frame = Image.fromarray(frame)
            if not origin_size:
                frame.thumbnail(utils.frame_resizing(height, width, frame_size, aspect_ratio)) # thumbnail
            frame.save(os.path.join(frame_path, "{}.jpeg".format(index)), quality=int(quality*100))
            index += 1

def extract_cpu(index, video_path, frame_path, frame_size, quality, origin_size, aspect_ratio):
    # get filename and make a save directory
    filename, frame_path = utils.get_filename_frame_path(video_path, frame_path)

    # read a informations
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    height, width = utils.frame_resizing(height, width, frame_size, aspect_ratio)

    # message
    print("{}/{} name: {} length: {}".format(index[0]+1, index[1], filename, length))

    # read and save
    pipe = ffmpeg.input(video_path)
    if not origin_size:
        pipe = pipe.filter("scale", height, width) # thumbnail
    pipe = pipe.output(os.path.join(frame_path, "%d.jpeg"), qscale=(1-quality)*30+1, format="image2", vcodec="mjpeg")
    pipe = pipe.global_args(*["-loglevel", "error", "-threads", "1"])
    pipe.run()
    