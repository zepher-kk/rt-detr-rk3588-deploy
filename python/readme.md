------

# 🐍 Python 工具链指南 (Python Toolchain)

本目录包含了 RT-DETR 模型从 PC 端导出到 RK3588 板端部署的**全链路 Python 脚本**。这三个脚本各自独立，分工明确，构成了完整的转换与测试流水线。

## 📁 文件说明与使用方法

### 1. `export.py` (模型导出脚本)

- **运行环境**: PC 端 (需安装 PyTorch 等深度学习环境)

- **核心作用**: 负责将原始的 PyTorch 模型 (`.pt`) 转化为静态的 ONNX 模型 (`.onnx`)。

- **✨ V2 亮点**: 内部集成了针对 RK3588 的算子优化黑魔法（重写了 `grid_sample`），在根源上彻底消灭了 NPU 处理边缘特征时由 `GatherElements` 引起的 FP16 精度越界报错。同时去除了耗时的 NMS 后处理，保证输出最纯净的张量矩阵。

- **使用方法**:

  Bash

  ```
  # 将 best.pt 导出为 onnx 模型
  python export.py --weights best.pt --img_size 640
  ```

### 2. `convert.py` (RKNN 转换脚本)

- **运行环境**: PC 端 (需安装 `rknn-toolkit2`)

- **核心作用**: 负责将上一步导出的 `.onnx` 模型编译成 RK3588 NPU 能够识别的 `.rknn` 专属格式。

- **✨ V2 亮点**: 修复了 INT8 量化掉精度的问题，支持快速进行 FP16 或 INT8 的一键转换，并自动配置 `std_values` 等预处理参数。

- **使用方法**:

  Bash

  ```
  # 将 onnx 模型转换为 rknn 模型 (支持 fp16 或 i8)
  python convert.py --onnx_path best.onnx --platform rk3588 --quant_dtype i8 --output_name best_rtdetr.rknn
  ```

### 3. `infer.py` (板端推理测试脚本)

- **运行环境**: RK3588 板端 (需安装 `rknn-toolkit-lite2` 和 OpenCV)

- **核心作用**: 部署在开发板上的 Python 版推理代码。用于快速验证转换后的 `.rknn` 模型精度是否正常，支持单张图片、本地视频流和 USB 摄像头的直接接入。

- **✨ V2 亮点**: 包含了完全对齐 C++ 版的双张量 (`pred_boxes` 和 `pred_logits`) 解析后处理逻辑，是验证模型精度最权威的 Baseline。

- **使用方法**:

  Bash

  ```
  # 单图测试 (验证画框精度)
  python infer.py --model_path best_rtdetr.rknn --source ../img/uav.jpg
  
  # 视频流动态测速 (验证连续帧)
  python infer.py --model_path best_rtdetr.rknn --source ../img/cars.mp4
  ```

------

💡 **部署工作流建议**: `PC端运行 export.py -> PC端运行 convert.py -> 板端运行 infer.py 验证精度 -> 最后上 C++ 榨干性能！`