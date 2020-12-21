import os
import json
import argparse
import frame
import flow
from glob import glob
from joblib import Parallel, delayed

# this code working for Toyota Smarthome dataset
# https://project.inria.fr/toyotasmarthome/
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-path", type=str, default="./videos/")
    parser.add_argument("--frames-path", type=str, default="./frames/")
    parser.add_argument("--flows-path", type=str, default="./flows/")
    parser.add_argument("--frame-size", type=int, default=480)
    parser.add_argument("--quality", type=int, default=0.75)
    parser.add_argument("--flow-mode", action="store_true")
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--use-gpu", action="store_true")
    args = parser.parse_args()

    # directory check
    assert os.path.exists(args.videos_path) is True, "'{}' directory is not exist !!".format(args.videos_path)
    if args.flow_mode:
        assert os.path.exists(args.flows_path) is False, "'{}' directory is already exist !!".format(args.flows_path)
    else:
        assert os.path.exists(args.frames_path) is False, "'{}' directory is already exist !!".format(args.frames_path)

    # get videos path
    videos_path = glob(os.path.join(args.videos_path, "*"))

    if args.flow_mode:
        arguments = [args.flows_path, args.frame_size, args.quality, 0.05]
        extractor = flow.extract
    else:
        arguments = [args.frames_path, args.frame_size, args.quality]
        if args.use_gpu:
            extractor = frame.extract_gpu
        else:
            extractor = frame.extract_cpu
    
    # run
    stats = Parallel(n_jobs=args.workers, backend="threading")(delayed(extractor)([i, len(videos_path)], video_path, *arguments) for i, video_path in enumerate(videos_path))
