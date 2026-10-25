# Python 工具

简体中文 | [English](README.en.md)

本目录包含模型导出、量化、评估、数据集、训练、诊断和运行时工具。脚本保持
独立命令行形式，便于复用和组合。

## 目录

| 目录 | 用途 |
|---|---|
| `export/` | ONNX 导出和输出裁剪 |
| `quantization/` | RKNN 模型转换 |
| `evaluation/` | mAP、阈值和进程耗时评估 |
| `dataset/` | VisDrone 标注转换 |
| `training/` | UAV-DETR 微调入口 |
| `diagnostics/` | ONNX 与量化诊断 |
| `runtime/` | 板端 Python 推理验证 |
| `legacy/` | 基础导出和转换脚本 |

## 主流程

```text
checkpoint
  -> export
  -> quantization
  -> evaluation
  -> target validation
```

详细命令见各目录的说明文档。

## 兼容性

- C++ 主程序支持 150 和 300 query。
- Python 推理脚本目前只支持 300 query。
- 当前 150 query 模型应通过 C++ 程序部署。
- 模型文件、数据集和宿主机备份不进入 Git。
