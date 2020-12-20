import os
import cv2
import numpy as np
from PIL import Image

def extract(index, video_path, flows_path, frame_size, quality):
    # remove 'v_' and extension
    filename = video_path.split("/")[-1][2:].split(".")[0]

    # make a save directory
    flows_path = os.path.join(flows_path, filename)
    os.makedirs(flows_path)

    # load a video
    cap = cv2.VideoCapture(video_path)

    # get length info from video
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # read a video
    ret, frame_first = cap.read()
    if not ret:
        message = "[ERROR] falied to read a video from '{}'".format(video_path)
        raise Exception(message)

    # convert to gray(first)
    frame_prev_gray = cv2.cvtColor(frame_first, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame_first)
    hsv[..., 1] = 255

    # message
    print("{}/{} name: {} length: {}".format(index[0]+1, index[1], filename, length))

    # read and save
    maximum_failure = 20
    failed = 0
    for i in range(1, length):
        ret, frame_next = cap.read()
        
        if not ret:
            print("[WARNING] falied to read a frame from '{}'".format(video_path))
            failed += 1
            continue

        if failed >= maximum_failure:
            message = "[ERROR] falied to read a video from '{}'".format(video_path)
            raise Exception(message)

        # convert to gray(next)
        frame_next_gray = cv2.cvtColor(frame_next, cv2.COLOR_BGR2GRAY)

        # Computes a dense optical flow using the Gunnar Farneback's algorithm
        frame_flow = cv2.calcOpticalFlowFarneback(frame_prev_gray, frame_next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Calculates the magnitude and angle of 2D vectors
        mag, ang = cv2.cartToPolar(frame_flow[..., 0], frame_flow[..., 1])
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        # save
        image = Image.fromarray(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
        image.thumbnail([frame_size, frame_size])
        image.save(os.path.join(flows_path, "{}.jpg".format(i - 1)), quality=quality)
        frame_prev_gray = frame_next_gray
    cap.release()
