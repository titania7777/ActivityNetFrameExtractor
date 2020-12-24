# FrameExtractor

<table style="border:0px">
  <tr>
    <td><img src="samples/rgb.gif"></td>
    <td><img src="samples/flow.gif"></td>
  </rt>
</table>

the frame extractor for Video Datasets with GPU Acceleration, powered by [Decord](https://github.com/dmlc/decord)

* if you want to use GPU then you will need to install from the source, detail is follow the [readme](https://github.com/dmlc/decord#installation) of Decord repository

## My Environment Setting

*   Decord == 0.4.2
*   opencv-python == 4.4.0.44
*   pillow-simd == 7.0.0.post3
*   ffmpeg-python == 0.2.0

## Usage
* first download the Video Dataset, for example an ActivityNet use a [crawler](https://github.com/activitynet/ActivityNet/tree/master/Crawler) or [request](https://github.com/activitynet/ActivityNet/issues/57) and then you will get the follow structure

```
   v1-3 (ActivityNet version 1.3)
     L train
     L val
     L test
```
then you can run a script like below

* extract only frames (cpu version, just remove '--use-gpu' flag)
```
python frame_extractor.py --videos-path ./v1-3/train/ --frames-path ./frames/ --frame-size 240 --quality 0.7 --use-gpu
```

* extract only optical flows (only cpu)
```
python frame_extractor.py --videos-path ./v1-3/train/ --flows-path ./flows/ --frame-size 240 --quality 0.7 --flow-mode
```
* if you want get origin size of frames in video then adds '--origin-size' flag

i use the ffmpeg when using a cpu, do if you need consider of quality then check it below formula and table

ffmpeg_qscale = (1 - args_quality) * 30 + 1

args_quality = (31 - ffmpeg_qscale) / 30

args_quality | ffmpeg_qscale
-- | -- 
1.0 | 1
0.9 | 4
0.8 | 7
0.7 | 10
0.6 | 13
0.5 | 16
0.4 | 19
0.3 | 22
0.2 | 25
0.1 | 28
0.05 | 29.5
0.0 | 31