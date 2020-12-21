import os
import ffmpeg
import numpy as np
from PIL import Image # pillow-simd
from decord import VideoReader
from decord import gpu

def extract_gpu(index, video_path, frame_path, frame_size, quality, origin_size):
    # get filename
    filename = video_path.split("/")[-1][:-4]

    # make a save directory
    frame_path = os.path.join(frame_path, filename)
    os.makedirs(frame_path)

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
    for i in range(length):
        frame = video_reader[i]
        image = Image.fromarray(frame.asnumpy())
        if not origin_size:
            image.thumbnail([frame_size, frame_size]) # thumbnail
        image.save(os.path.join(frame_path, "{}.jpeg".format(i)), quality=int(quality*100))

def extract_cpu(index, video_path, frame_path, frame_size, quality, origin_size):
    # get filename
    filename = video_path.split("/")[-1][:-4]

    # make a save directory
    frame_path = os.path.join(frame_path, filename)
    os.makedirs(frame_path)

    # get probe to get information
    probe = ffmpeg.probe(video_path)
    streams = probe["streams"][0]
    
    # get information from the probe
    width = streams["width"]
    height = streams["height"]
    codec_type = streams["codec_type"]
    frame_rate_part = streams["r_frame_rate"].split("/")
    fps = int(frame_rate_part[0]) / int(frame_rate_part[1])
    length = int(fps * float(probe["format"]["duration"])) # some videos has no 'nb_frames'

    # message
    print("{}/{} name: {} length: {}".format(index[0]+1, index[1], filename, length))
    
    # read and save
    if codec_type == "video":
        pipe = ffmpeg.input(video_path)
        if not origin_size:
            pipe = pipe.filter("scale", frame_size, -1) # thumbnail
        pipe = pipe.output(os.path.join(frame_path, "%d.jpeg"), qscale=(1-quality)*30+1, format="image2", vcodec="mjpeg")
        pipe = pipe.global_args(*["-loglevel", "error", "-threads", "1"])
        pipe.run(capture_stdout=True, capture_stderr=True)
    else:
        message = "[ERROR] type is not supported '{}'".format(video_path)
        raise Exception(message)
