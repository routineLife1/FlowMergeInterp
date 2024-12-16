# FlowMergeInterp
Merge consecutive optical flows for frame interpolation

# 👀Demo

## input
![input](assert/input.gif)
## output
![output](assert/output.gif)

## 🔧Installation

```bash
git clone https://github.com/routineLife1/FlowMergeInterp.git
cd FlowMergeInterp
pip3 install -r requirements.txt
```
The cupy package is included in the requirements, but its installation is optional. It is used to accelerate computation. If you encounter difficulties while installing this package, you can skip it.

## ⚡Usage 

**Video Interpolation**
```bash
  python interpolate_video.py -i input.mp4 -o output.mp4 -fps 60 -scale 1.0 -s -st 0.3 -hw -w 3
```

**Full Usage**
```bash
Usage: python interpolate_video_rife_anyfps.py -i in_video -o out_video [options]...
       
  -h                   show this help
  -i input             input video path (absolute path of output video)
  -o output            output video path (absolute path of output video)
  -fps dst_fps         target frame rate (default=60)
  -s enable_scdet      enable scene change detection (default Enable)
  -st scdet_threshold  ssim scene detection threshold (default=0.3)
  -hw hwaccel          enable hardware acceleration encode (default Enable) (require nvidia graph card)
  -scale scale         flow scale factor (default=1.0), generally use 1.0 with 1080P and 0.5 with 4K resolution
  -w window_size       merge frame window size (default=3)
```

- input accept absolute video file path. Example: E:/input.mp4
- output accept absolute video file path. Example: E:/output.mp4
- dst_fps = target interpolated video frame rate. Example: 60
- enable_scdet = enable scene change detection.
- scdet_threshold = scene change detection threshold. The larger the value, the more sensitive the detection.
- hwaccel = enable hardware acceleration during encoding output video.
- scale = flow scale factor. Decrease this value to reduce the computational difficulty of the model at higher resolutions. Generally, use 1.0 for 1080P and 0.5 for 4K resolution.
- window_size = merge frame window size. 


# 🔗Reference
Optical Flow: [GMFlow](https://github.com/haofeixu/gmflow)

Video Interpolation: [GMFSS](https://github.com/98mxr/GMFSS_Fortuna)