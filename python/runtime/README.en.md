# Board-Side Python Inference

English | [简体中文](README.md)

## `infer.py`

Provides a quick validation path for an RKNN model.

```bash
python python/runtime/infer.py \
  --model_path /path/to/model.rknn \
  --source img/uav.jpg \
  --conf_thres -0.53 \
  --img_size 640
```

The current script parses 300-query output and is not yet compatible with
150-query models. Use the C++ application for 150-query inference.

Requirements: RKNN Lite2 runtime, OpenCV, and NumPy on the board.

The full RGA, GStreamer/MPP, and multithreaded implementation is provided by
`cpp/rtdetr_pipeline`.
