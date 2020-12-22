import os

def get_filename_frame_path(video_path:str, frame_path:str) -> (str, str):
    # get filename
    filename = video_path.split("/")[-1][:-4]

    # make a save directory
    frame_path = os.path.join(frame_path, filename)
    os.makedirs(frame_path)
    return filename, frame_path

def frame_resizing(height:int, width:int, frame_size, aspect_ratio) -> list:
    if width > height:
        height = frame_size
        width = int((int(aspect_ratio[0])/int(aspect_ratio[1]))*height)
    else:
        width = frame_size
        height = int((int(aspect_ratio[1])/int(aspect_ratio[0]))*width)
    return [height, width]