# Model evaluation

[简体中文](README.md) | English

All detection exports use one line per box:

```text
<stem> <class_id> <score> <x1> <y1> <x2> <y2>
```

VisDrone annotation arguments point to the directory containing the original
`annotations/*.txt` files unless a command states otherwise.

## AP and operating points

```bash
python python/evaluation/visdrone_map.py \
  --det detections.txt \
  --ann /path/to/VisDrone2019-DET-val/annotations \
  --img-dir /path/to/VisDrone2019-DET-val/images \
  --conf -0.53 \
  --ignore-regions
```

The evaluator reports COCO-style AP/AP50/AP75, size buckets, and P/R/F1 at the
requested operating thresholds.

## Threshold calibration

```bash
python python/evaluation/calib_class_thresholds.py \
  --det detections.txt \
  --ann /path/to/annotations \
  --ignore-regions
```

`sweep_thresholds.py` compares global thresholds, while
`multimodel_thresholds.py` evaluates several model exports with a reference
class-wise vector. `verify_profile.py` checks a global or class-wise threshold
specification against an existing detection export.

## Process cost

```bash
python python/evaluation/bench_subprocess.py -- \
  ./cpp/rtdetr_pipeline -m model.rknn -v sample.mp4 -G 0
```

`bench_subprocess.py` records wall time, CPU utilization, and peak RSS. Keep the
same media, thread count, NPU core mask, and performance governor when comparing
models.

The evaluation scripts do not contain model files or ground-truth datasets.
