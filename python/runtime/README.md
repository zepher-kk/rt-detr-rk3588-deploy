# 板端 Python 推理

简体中文 | [English](README.en.md)

## `infer.py`

用于快速验证 RKNN 模型。

```bash
python python/runtime/infer.py \
  --model_path /path/to/model.rknn \
  --source img/uav.jpg \
  --conf_thres -0.53 \
  --img_size 640
```

当前脚本按 300 query 输出解析，暂不兼容 150 query 模型。150 query 模型请
使用 C++ 主程序。

依赖板端 RKNN Lite2 runtime、OpenCV 和 NumPy。

完整 RGA、GStreamer/MPP 和多线程能力由 `cpp/rtdetr_pipeline` 提供。
