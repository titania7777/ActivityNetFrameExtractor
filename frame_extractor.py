import os
import json
import argparse
import frame
import flow
from glob import glob
from joblib import Parallel, delayed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-path", type=str, default="./videos/")
    parser.add_argument("--frames-path", type=str, default="./frames/")
    parser.add_argument("--flows-path", type=str, default="./flows/")
    parser.add_argument("--frame-size", type=int, default=240)
    parser.add_argument("--quality", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--flow-mode", action="store_true")
    parser.add_argument("--origin-size", action="store_true")
    parser.add_argument("--use-gpu", action="store_true")
    args = parser.parse_args()

    # directory check
    assert os.path.exists(args.videos_path) is True, "'{}' directory is not exist !!".format(args.videos_path)
    assert os.path.exists(args.flows_path if args.flow_mode else args.frames_path) is False, "'{}' directory is already exist !!".format(args.flows_path)
    
    # ues it for make a frame directories
    start_point = len(os.path.join(args.videos_path, "hello").split("/")) - 1

    # get videos path
    videos_path = glob(os.path.join(args.videos_path, "**/*.*"), recursive=True)

    arguments = [start_point, args.flows_path if args.flow_mode else args.frames_path, args.frame_size, args.quality, args.origin_size]
    if args.use_gpu:
        arguments.append(args.batch_size)

    if args.flow_mode:
        extractor = flow.extract 
    else:
        extractor = frame.extract_gpu if args.use_gpu else frame.extract_cpu
    
    # run
    Parallel(n_jobs=args.workers, backend="threading")(delayed(extractor)([i, len(videos_path)], video_path, *arguments) for i, video_path in enumerate(videos_path))