# RKNN Conversion

English | [简体中文](README.md)

## `convert_rknn_variant.py`

Converts an ONNX model to RKNN and exposes the main quantization controls.

```bash
python python/quantization/convert_rknn_variant.py \
  --onnx /path/to/model.onnx \
  --dataset /path/to/calibration.txt \
  --output /path/to/model.rknn
```

Common options:

| Option | Purpose |
|---|---|
| `--method layer` / `channel` | Quantization granularity |
| `--auto-hybrid` | Preserve higher precision for sensitive layers |
| `--optimization-level` | Graph optimization level |
| `--disable-rule` | Disable a specified conversion rule |
| `--crop-inputs` | Bind a 640 input explicitly |

Requires `rknn-toolkit2` and a calibration image list. The conversion
environment must be compatible with the target board runtime.

After conversion, validate accuracy and run inference on the target platform.
Do not use the converter return code alone as evidence that the model is
deployable.
