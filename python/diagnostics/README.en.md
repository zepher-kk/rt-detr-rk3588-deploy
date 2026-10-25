# Diagnostics

[简体中文](README.md) | English

These scripts inspect model graphs and quantization behavior. They are not part
of the normal export-to-deploy path.

| Script | Purpose |
|---|---|
| `probe_onnx_squeeze.py` | Locate Squeeze nodes and their input/output tensors |
| `alias_quant_cfg.py` | Rename tensor keys in a quantization configuration |
| `clip_analysis.py` | Inspect layer clipping rates from toolkit output |
| `onnx_dump.py` | Export ONNX Runtime outputs for comparison |

Examples:

```bash
python python/diagnostics/probe_onnx_squeeze.py model.onnx
python python/diagnostics/probe_onnx_squeeze.py model.onnx --needle Squeeze_1
python python/diagnostics/onnx_dump.py \
  --onnx model.onnx --images /path/to/images --out reference.txt
```

Check each script's `--help` for the full argument set. These tools may import
different dependency sets, including ONNX, ONNX Runtime, NumPy, and
RKNN-Toolkit2.
