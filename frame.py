import os
from PIL import Image
from decord import VideoReader
from decord import cpu, gpu

def extract(index, video_path, frame_path, frame_size, quality, use_gpu):
    # remove 'v_' and extension
    filename = video_path.split("/")[-1][2:].split(".")[0]

    # make a save directory
    frame_path = os.path.join(frame_path, filename)
    os.makedirs(frame_path)

    # load a video
    video_reader = VideoReader(video_path, ctx=gpu(0) if use_gpu else cpu(0))

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
        image.thumbnail([frame_size, frame_size])
        image.save(os.path.join(frame_path, "{}.jpg".format(i)), quality=quality)