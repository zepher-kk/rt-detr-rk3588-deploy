# RKNN 转换

简体中文 | [English](README.en.md)

## `convert_rknn_variant.py`

将 ONNX 模型转换为 RKNN，并提供量化参数入口。

```bash
python python/quantization/convert_rknn_variant.py \
  --onnx /path/to/model.onnx \
  --dataset /path/to/calibration.txt \
  --output /path/to/model.rknn
```

常用参数：

| 参数 | 用途 |
|---|---|
| `--method layer` / `channel` | 量化粒度 |
| `--auto-hybrid` | 对敏感层保留更高精度 |
| `--optimization-level` | 图优化等级 |
| `--disable-rule` | 禁用指定转换规则 |
| `--crop-inputs` | 显式绑定 640 输入 |

依赖 `rknn-toolkit2` 和校准图片列表。转换环境应与目标板端 runtime 兼容。

转换成功后仍需执行精度评估和目标平台测试。不要仅根据转换返回码判断模型
是否可部署。
