# Dataset conversion

[简体中文](README.md) | English

## `visdrone_to_yolo.py`

Converts VisDrone detection annotations to normalized YOLO labels.

```bash
python python/dataset/visdrone_to_yolo.py \
  --root /path/to/VisDrone2019 \
  --output /path/to/yolo_dataset \
  --splits train val
```

The expected input layout is:

```text
VisDrone2019/
├── VisDrone2019-DET-train/
│   ├── images/
│   └── annotations/
└── VisDrone2019-DET-val/
    ├── images/
    └── annotations/
```

Rows with `score=0` and categories `0` or `11` are ignored. Valid categories
`1..10` are written as class IDs `0..9`. Existing non-empty output directories
are rejected unless `--overwrite` is passed.

Dependencies: Pillow.

Run the unit test with:

```bash
python -m unittest discover -s python/dataset/tests -v
```
