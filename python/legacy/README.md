# 基础脚本

简体中文 | [English](README.en.md)

本目录保留原始基础导出和转换脚本：

| 文件 | 用途 |
|---|---|
| `export.py` | 将 PyTorch 权重导出为 ONNX |
| `convert.py` | 将 ONNX 转换为 RKNN |

这些脚本使用固定输入路径，运行时需要按本机实际路径修改。当前主流程使用
`../export/` 和 `../quantization/` 下的参数化工具。
