# ActivityNetFrameExtractor
the frame extractor for ActivityNet with GPU Accelaration, powered by [Decord](https://github.com/dmlc/decord)

* if you want to use GPU then you will need install from the source, detail is follow the [readme](https://github.com/dmlc/decord#installation) of Decord repository

## My Environment Setting

*   Decord == 0.4.2
*   opencv-python == 4.4.0.44
*   pillow == 8.0.1

## Usage
first download the ActivityNet use a [crawler](https://github.com/activitynet/ActivityNet/tree/master/Crawler) or [request](https://github.com/activitynet/ActivityNet/issues/57) and then you will get the follow structure.

```
   v1-3
     L train
     L val
     L test
```
then you can run a script like below.

extract only frames
```
python frame_extractor.py --videos-path ./v1-3/train/ --frames-path ./frames/ --frame-size 480 --quality 30 --workers 16 --use-gpu
```

extract only optical flows
```
python frame_extractor.py --videos-path ./v1-3/train/ --flows-path ./flows/ --frame-size 480 --quality 30 --workers 16 --flow-mode
```
