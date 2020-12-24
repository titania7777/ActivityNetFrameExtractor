import os
import sys
import utils

# ffmpeg GPU: https://github.com/kkroening/ffmpeg-python/issues/284
def extract_gpu(index, video_path, start_point, frame_path, frame_size, quality, origin_size, batch_size):
    from decord import VideoReader
    from decord import gpu
    from PIL import Image # pillow-simd

    # get filename and make a save directory
    filename, frame_path = utils.get_filename_frame_path(start_point, video_path, frame_path)

    # load a video
    video_reader = VideoReader(video_path, ctx=gpu(0))

    # get length info from video
    length = len(video_reader)

    if length == 0:
        message = "[ERROR] falied to read a video from '{}'".format(video_path)
        raise Exception(message)
    
    # get information
    width_original, height_original, _ = utils.get_info(video_path)

    # resizing
    if not origin_size:
        width_resize, height_resize = utils.frame_resizing(width_original, height_original, frame_size)
    else:
        width_resize, height_resize = width_original, height_original

    # message
    print(f"{index[0]+1}/{index[1]} ({width_original}x{height_original}) -> ({width_resize}x{height_resize}) length: {length:<{5}} name: {filename}")

    # read and save
    index = 0
    for i in range(length//batch_size if (length%batch_size) == 0 else (length//batch_size) + 1):
        frames = video_reader.get_batch(list(range(i*batch_size, (i+1)*batch_size if (length%batch_size) == 0 else i*batch_size + (length%batch_size)))).asnumpy()
        for frame in frames:
            height, width, _ = frame.shape
            frame = Image.fromarray(frame)
            frame.thumbnail([width_resize, height_resize])
            frame.save(os.path.join(frame_path, "{}.jpeg".format(index)), quality=int(quality*100))
            index += 1

def extract_cpu(index, video_path, start_point, frame_path, frame_size, quality, origin_size):
    import cv2
    import ffmpeg

    # get filename and make a save directory
    filename, frame_path = utils.get_filename_frame_path(start_point, video_path, frame_path)

    # get information
    width_original, height_original, length = utils.get_info(video_path)

    # resizing
    if not origin_size:
        width_resize, height_resize = utils.frame_resizing(width_original, height_original, frame_size)
    else:
        width_resize, height_resize = width_original, height_original

    # message
    print(f"{index[0]+1}/{index[1]} ({width_original}x{height_original}) -> ({width_resize}x{height_resize}) length: {length:<{5}} name: {filename}")

    # read and save
    (
        ffmpeg.input(video_path)
        .filter("scale", width_resize, height_resize)
        .output(os.path.join(frame_path, "%d.jpeg"), qscale=(1-quality)*30+1)
        .global_args("-loglevel", "error", "-threads", "1", "-nostdin")
        .run()
    )
