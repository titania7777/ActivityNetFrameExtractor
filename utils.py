import os

def get_filename_frame_path(video_path:str, frame_path:str) -> (str, str):
    # get filename
    filename = video_path.split("/")[-1][:-4]

    # make a save directory
    frame_path = os.path.join(frame_path, filename)
    os.makedirs(frame_path)
    return filename, frame_path

def frame_resizing(height:int, width:int, frame_size) -> list:
    aspect_ratio = width / height
    if width > height:
        if height >= frame_size:
            height = frame_size
        width = int(aspect_ratio*height)
    else:
        if width >= frame_size:
            width = frame_size
        height = int(aspect_ratio*width)
    return [height, width]