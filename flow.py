import os
import cv2
import ffmpeg
import numpy as np
from PIL import Image # pillow-simd
import utils

def extract(index, video_path, flows_path, frame_size, quality, origin_size):
    # get filename and make a save directory
    filename, flows_path = utils.get_filename_frame_path(video_path, flows_path)

    # read a informations
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # load a video
    out, _ = (
        ffmpeg
        .input(video_path)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        .global_args(*["-loglevel", "error", "-threads", "1"])
        .run(capture_stdout=True, capture_stderr=True)
    )
    video = (
        np
        .frombuffer(out, np.uint8)
        .reshape([-1, height, width, 3])
    )

    # message
    print(f"{index[0]+1}/{index[1]} ({width}x{height}) length: {length:<{5}} name: {filename}")

    # read a first frame and conver to gray scale
    frame_first = video[0]
    frame_prev_gray = cv2.cvtColor(frame_first, cv2.COLOR_RGB2GRAY)
    
    # saturation
    hsv = np.zeros_like(frame_first)
    hsv[..., 1] = 255

    # read and save
    for i in range(1, len(video)):
        # read a next frame and conver to gray scale
        frame_next = video[i]
        frame_next_gray = cv2.cvtColor(frame_next, cv2.COLOR_RGB2GRAY)
        
        # dense optical flow
        frame_flow = cv2.calcOpticalFlowFarneback(frame_prev_gray, frame_next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(frame_flow[..., 0], frame_flow[..., 1])
        # hue
        hsv[..., 0] = ang*180/np.pi/2
        # value
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        # save
        image = Image.fromarray(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
        if not origin_size:
            image.thumbnail(utils.frame_resizing(height, width, frame_size)) # thumbnail
        image.save(os.path.join(flows_path, "{}.jpeg".format(i - 1)), quality=int(quality*100))
        frame_prev_gray = frame_next_gray