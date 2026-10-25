# Model export

[简体中文](README.md) | English

## `export_uavdetr_onnx.py`

Exports a UAV-DETR/RT-DETR checkpoint to a static ONNX graph. The script patches
`grid_sample` and high-dimensional `Linear` operations so the graph avoids
dynamic shapes that the RKNN converter does not handle reliably.

```bash
python python/export/export_uavdetr_onnx.py \
  --repo /path/to/UAV-DETR \
  --checkpoint /path/to/best.pt \
  --output /path/to/model.onnx \
  --imgsz 640 \
  --num-queries 150 \
  --decoder-layers 2
```

Outputs are named `pred_logits` and `pred_boxes`. `--num-queries` and
`--decoder-layers` create the reduced 150-query, 2-layer decoder export used by
the current deployment model.

Dependencies: PyTorch, Ultralytics, and the specified UAV-DETR repository.

## `prune_rtdetr_outputs.py`

Removes the redundant RT-DETR tail and exposes only final boxes and scores.

```bash
python python/export/prune_rtdetr_outputs.py \
  model.onnx model_pruned.onnx --verify
```

`--verify` compares ONNX Runtime outputs against the original graph. The script
expects a 300-query reference graph, so validate tensor shapes before using it
on another architecture.

Dependencies: `onnx`, `onnxruntime`, and NumPy.
