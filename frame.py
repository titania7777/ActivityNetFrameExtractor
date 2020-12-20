import os
import cv2
from PIL import Image
from decord import VideoReader
from decord import gpu

def extract_gpu(index, video_path, frame_path, frame_size, quality):
    # remove 'v_' and extension
    filename = video_path.split("/")[-1][2:].split(".")[0]

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
        image.thumbnail([frame_size, frame_size])
        image.save(os.path.join(frame_path, "{}.jpg".format(i)), quality=quality)

def extract_cpu(index, video_path, frame_path, frame_size, quality):
    # remove 'v_' and extension
    filename = video_path.split("/")[-1][2:].split(".")[0]

    # make a save directory
    frame_path = os.path.join(frame_path, filename)
    os.makedirs(frame_path)

    # load a video
    cap = cv2.VideoCapture(video_path)

    # get length info from video
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # message
    print("{}/{} name: {} length: {}".format(index[0]+1, index[1], filename, length))

    # read and save
    maximum_failure = 20
    failed = 0
    for i in range(length):
        ret, frame = cap.read()

        # missing frame
        if not ret:
            print("[WARNING] falied to read a frame from '{}'".format(video_path))
            failed += 1
            continue
        
        # failure
        if failed >= maximum_failure:
            message = "[ERROR] falied to read a video from '{}'".format(video_path)
            raise Exception(message)
        
        # save
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image.thumbnail([frame_size, frame_size])
        image.save(os.path.join(frame_path, "{}.jpg".format(i)), quality=quality)
    cap.release()
