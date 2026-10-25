# Basic Scripts

English | [简体中文](README.md)

This directory retains the original basic export and conversion scripts:

| File | Purpose |
|---|---|
| `export.py` | Export PyTorch weights to ONNX |
| `convert.py` | Convert ONNX to RKNN |

These scripts use fixed input paths and require local path edits before use.
The primary workflow uses the configurable tools under `../export/` and
`../quantization/`.
