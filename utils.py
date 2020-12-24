import cv2
import os
from glob import glob

def get_info(video_path:str) -> (int, int, int):
    # read a informations
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height, length

def get_filename_frame_path(start_point:int, video_path:str, frame_path:str) -> (str, str):
    # split a video path
    video_path = video_path.split("/")

    # get filename
    filename = video_path[-1][:-4]

    # make a save directory
    frame_path = os.path.join(frame_path, *video_path[start_point:-1], filename)
    os.makedirs(frame_path)
    return filename, frame_path

def frame_resizing(width:int, height:int, frame_size) -> list:
    if width > height:
        aspect_ratio = width / height
        if height >= frame_size:
            height = frame_size
        width = int(aspect_ratio*height)
    else:
        aspect_ratio = height / width
        if width >= frame_size:
            width = frame_size
        height = int(aspect_ratio*width)
    return [width, height]