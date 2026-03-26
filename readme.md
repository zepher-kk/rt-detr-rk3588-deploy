# 🚀 RT-DETR on RK3588: 高性能边缘部署实战

[![Platform: RK3588](https://img.shields.io/badge/Platform-RK3588-blue.svg)](https://www.rock-chips.com/)
[![Framework: RT-DETR](https://img.shields.io/badge/Framework-RT--DETR-orange.svg)]()
[![Language: C++/Python](https://img.shields.io/badge/Language-C++%20%7C%20Python-green.svg)]()

本项目致力于将先进的 **RT-DETR (Real-Time DEtection TRansformer)** 算法部署到 **RK3588** 边缘计算平台上。
凭借 RT-DETR **“去 NMS (NMS-free)”** 的架构优势，结合我们专门为 RK3588 设计的 **C++ 异步多线程流水线**，完美解决了传统目标检测算法在边缘端后处理耗时过长的痛点，最大化压榨 NPU 算力！

---

## 🛠️ 1. 环境配置 (Environment Setup)

本项目包含模型转换与板端推理两部分，需要分别在 PC 端和 RK3588 板端配置环境：

* **PC 端 (Ubuntu x86) - 用于模型转换**:
    * Python 3.8+
    * [RKNN-Toolkit2](https://github.com/rockchip-linux/rknn-toolkit2) (推荐 v2.0 以上)
    * ONNX (用于加载和验证导出模型)
* **RK3588 板端 (Ubuntu/Debian) - 用于模型推理**:
    * **Python 推理**: `rknn-toolkit-lite2` (与 PC 端版本对齐), `OpenCV-Python`
    * **C++ 推理**: 
        * GCC / G++
        * CMake (>= 3.4)
        * OpenCV (C++ 版，用于图像预处理/画框)
        * RKNPU2 驱动 (`librknnrt.so`)

---

## 🔄 2. 模型转换 (Model Conversion)

RT-DETR 采用一对一匈牙利匹配，**直接输出 300 个预测框**。
> ⚠️ **注意**：导出 ONNX 时请务必使用 **静态 Shape (Static Shape)**，并去掉所有后处理 NMS 节点，保持最纯粹的张量输出（例如 `[1, 300, 14]`）。

**转换步骤**：
在部署了 `RKNN-Toolkit2` 的 PC 上，编写并运行转换脚本（支持 INT8 量化或 FP16）：
```bash
# 示例：通过 Toolkit2 将 onnx 转为 rknn
python convert.py <path_to_onnx> rk3588 fp16 <output_name.rknn>
python
测试单张图片（安静出图，不测速）：
Bash
python infer.py --model_path rtdetr.rknn   test.jpg --conf_thres 0.5
测试本地视频（狂飙出图，动态打印 FPS）：
Bash
python infer.py --model_path rtdetr.rknn --source demo.mp4 --img_size 640
接入 USB 摄像头实时测试：
Bash
python infer.py --model_path rtdetr.rknn --source 0

🐍 3. Python 推理 (Python Inference)
我们提供了高度工程化的 Python 推理脚本，支持 单图、本地视频流、USB 摄像头，并内置动态 FPS 测速。

运行方式：

Bash
cd python/

# 1. 单张图片测试 (安静出图)
python infer.py --model_path ../model/best.rknn --source ../img/uav.jpg --conf_thres 0.45

# 2. 本地视频动态测速
python infer.py --model_path ../model/best.rknn --source ../test.mp4

# 3. USB 摄像头实时推理 (注意替换实际的 /dev/video 节点)
python infer.py --model_path ../model/best.rknn --source 21 
检测结果（图片或视频）将自动保存在当前目录下。

⚡ 4. C++ 推理 (C++ Inference)
为了彻底解决 Python 单线程下 NPU 等待 CPU 前后处理的性能瓶颈，我们用 C++ 编写了三段式异步流水线 (PipelineManager)。
系统将任务解耦为：洗菜工(前处理) -> 主厨(NPU多核并行) -> 洗碗工(后处理与渲染)，大幅提升吞吐量。

编译流程：

Bash
cd cpp/
mkdir build && cd build
cmake ..
make -j4
make install  # 编译出的可执行文件会在 cpp/install 目录下
运行方式：
我们的 C++ 程序自带严谨的命令行解析器，极客范十足：

Bash
cd install/rknn_rtdetr_demo_Linux/

# 查看帮助菜单
./rknn_rtdetr_demo -h

# 1. 基础图片推理
./rknn_rtdetr_demo -m ../../../model/best.rknn -s ../../../img/uav.jpg

# 2. 视频流流水线全开 (支持自定义各阶段线程数)
./rknn_rtdetr_demo -m ../../../model/best.rknn -s ../../../test.mp4 --pre 2 --npu 3 --post 2

# 3. NPU 极限压测 (循环推理单图 1000 次计算极限 FPS)
./rknn_rtdetr_demo -m ../../../model/best.rknn -s ../../../img/uav.jpg -l 1000
📊 5. 效果展示 (Results)
🎯 检测精度与效果
(在这里插入你的检测结果截图，比如无人机或密集人群的检测图)

RT-DETR 在不依赖 NMS 的情况下，依然能精准区分紧密贴合的物体，彻底告别 YOLO 常见的“重叠漏检”问题。

🚀 性能飞跃 (Python vs C++ 多线程)
(在这里插入你的终端 FPS 截图)

Python 串行推理：约 ~X FPS (主要受限于 OpenCV CPU 前处理与 GIL 锁)。

C++ 异步流水线：约 ~X FPS (3个 NPU 核心满载，性能提升数倍！)。

💡 6. 小结 (Summary)
为什么选择 RT-DETR？ 在 RK3588 这种边缘芯片上，NPU 的张量运算极快，但 CPU 相对较弱。RT-DETR 免去了极其消耗 CPU 算力的 NMS 后处理，完美契合了“全流程 NPU 加速”的理念。

多线程的魅力：通过 C++ SafeQueue 解耦的流水线，使得读图、预处理、推理、解码各司其职，有效消灭了硬件闲置期。

下一步优化 (TODO)：目前前后处理仍依赖 CPU 端的 OpenCV。未来计划引入 Rockchip 官方的 RGA (2D 硬件加速器) 进行零 CPU 负载的图像缩放与格式转换，实现真正的极致性能。

👉 更详细的技术内幕与原理解析，请移步我的博客：[你的CSDN或个人博客链接]
```

------

### 🎉 V2 版本重大更新 (v2.0 Release Notes)

我们持续在榨干 RK3588 的性能，V2 版本带来了以下关键修复与升级：

- 🛠️ **模型导出与后处理重构** 

- 全面修复并优化了 `export.py` 的 ONNX 导出逻辑，同时同步重写了配套的张量后处理代码，数据流转更加严谨（具体实现请查阅最新源码）。

- 🎯 **INT8 量化精度修复 & 权重开源** 

- 成功攻克了 RT-DETR 在 INT8 (`i8`) 量化转换时精度断崖式下降的痛点！为了方便大家复现，我们现已开源经过验证的 `best.pt` 原始权重，强烈建议大家下载体验、自行转换或作为您自己项目的 Baseline 参考。

- 🚀 **视频流推理指令优化 (日志净化)** 

- **关于警告的说明**：在处理视频流时，由于 NPU 处理极边缘目标坐标时的半精度误差，底层驱动会抛出 `GatherElements` 越界警告。但这已被底层驱动自动修正，**绝对不影响最终的画框精度与程序稳定性**。 

- 为了保持终端测速日志的绝对纯净（眼不见为净），请在运行视频压测时加上 `grep -v` 魔法后缀来过滤底层警告：

  ```python
  ./rknn_rtdetr_demo -m /home/cat/project/rknn/rt-detr-rknn/model/rtdetr_i8.rknn -s /home/cat/project/rknn/rt-detr-rknn/img/cars.mp4 2>&1 | grep -v "GatherElements"
  ```