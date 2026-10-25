# 模型评估

简体中文 | [English](README.en.md)

检测导出文件每行表示一个框：

```text
<stem> <class_id> <score> <x1> <y1> <x2> <y2>
```

除非另有说明，VisDrone 标注参数指向包含 `annotations/*.txt` 的目录。

## AP 与工作点

```bash
python python/evaluation/visdrone_map.py \
  --det detections.txt \
  --ann /path/to/VisDrone2019-DET-val/annotations \
  --img-dir /path/to/VisDrone2019-DET-val/images \
  --conf -0.53 \
  --ignore-regions
```

评估器输出 COCO 风格 AP/AP50/AP75、尺寸分组结果，以及指定阈值下的
Precision、Recall 和 F1。

## 阈值标定

```bash
python python/evaluation/calib_class_thresholds.py \
  --det detections.txt \
  --ann /path/to/annotations \
  --ignore-regions
```

`sweep_thresholds.py` 用于全局阈值扫描；`multimodel_thresholds.py` 用于
多个模型导出和按类阈值向量比较；`verify_profile.py` 用于检查已有检测结果
在全局或按类阈值配置下的指标。

## 进程开销

```bash
python python/evaluation/bench_subprocess.py -- \
  ./cpp/rtdetr_pipeline -m model.rknn -v sample.mp4 -G 0
```

`bench_subprocess.py` 记录 wall time、CPU 占用和峰值 RSS。比较模型时应固定
视频、线程数、NPU core mask、输入尺寸和输出方式。

评估脚本不包含模型文件或标注数据集。
