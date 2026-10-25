# Python Tools

English | [简体中文](README.md)

This directory contains standalone tools for model export, quantization,
evaluation, dataset conversion, training, diagnostics, and runtime validation.
Each script is designed to be reused and combined through the command line.

## Directories

| Directory | Purpose |
|---|---|
| `export/` | ONNX export and output pruning |
| `quantization/` | RKNN model conversion |
| `evaluation/` | mAP, threshold, and process-cost evaluation |
| `dataset/` | VisDrone annotation conversion |
| `training/` | UAV-DETR fine-tuning entry point |
| `diagnostics/` | ONNX and quantization diagnostics |
| `runtime/` | Board-side Python inference |
| `legacy/` | Basic export and conversion scripts |

## Main Workflow

```text
checkpoint
  -> export
  -> quantization
  -> evaluation
  -> target validation
```

See each directory README for detailed commands.

## Compatibility

- The C++ application supports 150- and 300-query outputs.
- The Python inference script currently supports 300-query output only.
- Deploy the current 150-query model through the C++ application.
- Model files, datasets, and host backups are not included in Git.
