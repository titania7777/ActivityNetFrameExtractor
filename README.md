# FrameExtractor
the frame extractor for Video Datasets with GPU Acceleration, powered by [Decord](https://github.com/dmlc/decord)

* if you want to use GPU then you will need install from the source, detail is follow the [readme](https://github.com/dmlc/decord#installation) of Decord repository

## My Environment Setting

*   Decord == 0.4.2
*   opencv-python == 4.4.0.44
*   pillow-simd == 7.0.0.post3
*   ffmpeg-python == 0.2.0

## Usage
first download the Video Dataset, for example an ActivityNet use a [crawler](https://github.com/activitynet/ActivityNet/tree/master/Crawler) or [request](https://github.com/activitynet/ActivityNet/issues/57) and then you will get the follow structure

```
   v1-3 (ActivityNet version 1.3)
     L train
     L val
     L test
```
then you can run a script like below

extract only frames (cpu version, just remove '--use-gpu' flag)
```
python frame_extractor.py --videos-path ./v1-3/train/ --frames-path ./frames/ --frame-size 360 --quality 0.7 --use-gpu
```

extract only optical flows (only cpu)
```
python frame_extractor.py --videos-path ./v1-3/train/ --flows-path ./flows/ --frame-size 360 --quality 0.7 --flow-mode
```

i use the ffmpeg when using a cpu, do if you need consider of quality then check it below formula
ffmpeg_qscale = (1 - args_quality) * 30 + 1
