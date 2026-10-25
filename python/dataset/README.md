# 数据集转换

简体中文 | [English](README.en.md)

## `visdrone_to_yolo.py`

将 VisDrone 检测标注转换为归一化 YOLO 标签。

```bash
python python/dataset/visdrone_to_yolo.py \
  --root /path/to/VisDrone2019 \
  --output /path/to/yolo_dataset \
  --splits train val
```

输入目录结构：

```text
VisDrone2019/
├── VisDrone2019-DET-train/
│   ├── images/
│   └── annotations/
└── VisDrone2019-DET-val/
    ├── images/
    └── annotations/
```

`score=0`、类别 `0` 和类别 `11` 的记录会被忽略。有效类别 `1..10` 转换为
类别 ID `0..9`。除非指定 `--overwrite`，非空输出目录会被拒绝。

依赖：Pillow。

运行单元测试：

```bash
python -m unittest discover -s python/dataset/tests -v
```
