# 诊断工具

简体中文 | [English](README.en.md)

这些脚本用于检查模型图和量化行为，不属于常规导出到部署链路。

| 脚本 | 用途 |
|---|---|
| `probe_onnx_squeeze.py` | 定位 Squeeze 节点及输入输出张量 |
| `alias_quant_cfg.py` | 修改量化配置中的张量键名 |
| `clip_analysis.py` | 分析 Toolkit 输出中的逐层裁剪率 |
| `onnx_dump.py` | 导出 ONNX Runtime 输出用于对比 |

示例：

```bash
python python/diagnostics/probe_onnx_squeeze.py model.onnx
python python/diagnostics/probe_onnx_squeeze.py model.onnx --needle Squeeze_1
python python/diagnostics/onnx_dump.py \
  --onnx model.onnx --images /path/to/images --out reference.txt
```

各脚本可能使用不同依赖，包括 ONNX、ONNX Runtime 和 NumPy。使用前可通过
`--help` 查看完整参数。
