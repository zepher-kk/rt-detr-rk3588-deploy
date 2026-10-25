# 模型导出

简体中文 | [English](README.en.md)

## `export_uavdetr_onnx.py`

将 UAV-DETR/RT-DETR 权重导出为静态 ONNX 图。脚本会替换 `grid_sample`
和高维 `Linear` 相关算子，减少 RKNN 转换器难以处理的动态形状。

```bash
python python/export/export_uavdetr_onnx.py \
  --repo /path/to/UAV-DETR \
  --checkpoint /path/to/best.pt \
  --output /path/to/model.onnx \
  --imgsz 640 \
  --num-queries 150 \
  --decoder-layers 2
```

输出名称为 `pred_logits` 和 `pred_boxes`。`--num-queries` 与
`--decoder-layers` 用于生成当前部署使用的 150-query、2 层 Decoder 导出。

依赖：PyTorch、Ultralytics 和指定的 UAV-DETR 仓库。

## `prune_rtdetr_outputs.py`

删除 RT-DETR 冗余尾部，只保留最终 boxes 和 scores。

```bash
python python/export/prune_rtdetr_outputs.py \
  model.onnx model_pruned.onnx --verify
```

`--verify` 会使用 ONNX Runtime 对比原始图与裁剪图的输出。脚本以
300-query 图为参考，用于其他结构前应先核对张量形状。

依赖：`onnx`、`onnxruntime` 和 NumPy。
