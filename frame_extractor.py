import os
import json
import argparse
import frame
import flow
from glob import glob
from joblib import Parallel, delayed

# this code working for ActivityNet dataset
# http://activity-net.org/index.html
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-path", type=str, default="./videos/")
    parser.add_argument("--frames-path", type=str, default="./frames/")
    parser.add_argument("--flows-path", type=str, default="./flows/")
    parser.add_argument("--frame-size", type=int, default=480) # 480x270, 320x180
    parser.add_argument("--quality", type=int, default=30)
    parser.add_argument("--flow-mode", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
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

    # run
    if args.flow_mode:
        stats = Parallel(
            n_jobs=args.workers, backend="threading"
        )(delayed(flow.extract)(
            [i, len(videos_path)], video_path, args.flows_path, args.frame_size, args.quality) for i, video_path in enumerate(videos_path)
        )
    else:
        if args.use_gpu:
            stats = Parallel(
                n_jobs=args.workers, backend="threading"
            )(delayed(frame.extract_gpu)(
                [i, len(videos_path)], video_path, args.frames_path, args.frame_size, args.quality) for i, video_path in enumerate(videos_path)
            )
        else:
            stats = Parallel(
                n_jobs=args.workers, backend="threading"
            )(delayed(frame.extract_cpu)(
                [i, len(videos_path)], video_path, args.frames_path, args.frame_size, args.quality) for i, video_path in enumerate(videos_path)
            )
