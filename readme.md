# 🚀 RT-DETR on RK3588: 高性能边缘部署实战

[![Platform: RK3588](https://img.shields.io/badge/Platform-RK3588-blue.svg)](https://www.rock-chips.com/)
[![Framework: RT-DETR](https://img.shields.io/badge/Framework-RT--DETR-orange.svg)]()
[![Language: C++/Python](https://img.shields.io/badge/Language-C++%20%7C%20Python-green.svg)]()

本项目致力于将先进的 **RT-DETR (Real-Time DEtection TRansformer)** 算法部署到 **RK3588** 边缘计算平台上。

凭借 RT-DETR **“去 NMS (NMS-free)”** 的架构优势，结合我们专门为 RK3588 设计的 **C++ 异步多线程流水线**，完美解决了传统目标检测算法在边缘端后处理耗时过长的痛点，最大化压榨 NPU 算力！

**算子优化请参考博客：[Matmul#2](https://blog.csdn.net/weixin_65585850/article/details/158509230?spm=1001.2014.3001.5502), [grid_sample](https://blog.csdn.net/weixin_65585850/article/details/158575468?spm=1001.2014.3001.5502)**
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
        * librga / libdrm / libv4l2（V3 起 RGA_DMA 零拷贝通路依赖）

---

## 🔄 2. 模型转换 (Model Conversion)
**[模型下载](https://drive.google.com/drive/folders/1zS1hq9Hf5GdOKOKwz2djPmIrUm8cQLn_?usp=sharing),包含了pt，onnx，rknn模型**

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
```
🐍 3. Python 推理 (Python Inference)
我们提供了高度工程化的 Python 推理脚本，支持 单图、本地视频流、USB 摄像头，并内置动态 FPS 测速。

运行方式：

```bash
cd python/

# 1. 单张图片测试 (安静出图)
python infer.py --model_path ../model/best.rknn --source ../img/uav.jpg --conf_thres 0.45

# 2. 本地视频动态测速
python infer.py --model_path ../model/best.rknn --source ../test.mp4

# 3. USB 摄像头实时推理 (注意替换实际的 /dev/video 节点)
python infer.py --model_path ../model/best.rknn --source 21 
检测结果（图片或视频）将自动保存在当前目录下。
```
⚡ 4. C++ 推理 (C++ Inference)
为了彻底解决 Python 单线程下 NPU 等待 CPU 前后处理的性能瓶颈，我们用 C++ 编写了三段式异步流水线 (PipelineManager)。
系统将任务解耦为：洗菜工(前处理) -> 主厨(NPU多核并行) -> 洗碗工(后处理与渲染)，大幅提升吞吐量。

编译流程：
```bash
cd cpp/
mkdir build && cd build
cmake ..
make -j4
make install  # 编译出的可执行文件会在 cpp/install 目录下
运行方式：
我们的 C++ 程序自带严谨的命令行解析器，极客范十足！！！
```
📊 5. 效果展示 (Results)
🎯 检测精度与效果


RT-DETR 在不依赖 NMS 的情况下，依然能精准区分紧密贴合的物体，彻底告别 YOLO 常见的“重叠漏检”问题。

🚀 性能飞跃 (Python vs C++ 多线程)
(在这里插入你的终端 FPS 截图)

Python 串行推理：约 2.67X FPS (主要受限于 OpenCV CPU 前处理与 GIL 锁)。

C++ 异步流水线：约 7X FPS (3个 NPU 核心满载，性能提升数倍！)。

💡 6. 小结 (Summary)
为什么选择 RT-DETR？ 在 RK3588 这种边缘芯片上，NPU 的张量运算极快，但 CPU 相对较弱。RT-DETR 免去了极其消耗 CPU 算力的 NMS 后处理，完美契合了“全流程 NPU 加速”的理念。

多线程的魅力：通过 C++ SafeQueue 解耦的流水线，使得读图、预处理、推理、解码各司其职，有效消灭了硬件闲置期。

下一步优化 (TODO)：目前视频文件的解码与结果视频的编码仍依赖 CPU 端 OpenCV 软编解码。未来计划引入 **GStreamer + Rockchip MPP 硬件编解码 (gst_mpp)**，用硬件解码/编码替换 OpenCV 软解码/软编码，并与现有 RGA_DMA 零拷贝通路打通，实现采集—解码—预处理—推理—编码的全链路硬件加速。

👉 更详细的技术内幕与原理解析，请移步博客：[你的CSDN或个人博客链接](https://blog.csdn.net/weixin_65585850?type=blog)

------

### 🎉 V2 版本重大更新 (v2.0 Release Notes)

我们持续在榨干 RK3588 的性能，V2 版本带来了以下关键修复与升级：

- 🛠️ **模型导出与后处理重构** 

- 全面修复并优化了 `export.py` 的 ONNX 导出逻辑，同时同步重写了配套的张量后处理代码，数据流转更加严谨（具体实现请查阅最新源码）。

- 🎯 **INT8 量化精度修复 & 权重开源** 

- 成功攻克了 RT-DETR 在 INT8 (`i8`) 量化转换时精度断崖式下降的痛点(FPS=12)！为了方便大家复现，我们现已开源经过验证的 `best.pt` 原始权重，强烈建议大家下载体验、自行转换或作为您自己项目的 Baseline 参考。

- 🚀 **视频流推理指令优化 (日志净化)** 

- **关于警告的说明**：在处理视频流时，由于 NPU 处理极边缘目标坐标时的半精度误差，底层驱动会抛出 `GatherElements` 越界警告。但这已被底层驱动自动修正，**绝对不影响最终的画框精度与程序稳定性**。 

- 为了保持终端测速日志的绝对纯净（眼不见为净），请在运行视频压测时加上 `grep -v` 魔法后缀来过滤底层警告：

  ```python
  ./rknn_rtdetr_demo -m /home/cat/project/rknn/rt-detr-rknn/model/rtdetr_i8.rknn -s /home/cat/project/rknn/rt-detr-rknn/img/cars.mp4 2>&1 | grep -v "GatherElements"
  ```

------

### 🎉 V3 版本重大更新 (v3.0 Release Notes: RGA_DMA + NEON 集成加速优化)

最新提交 (430aa1a) 为 cpp/ 带来了 **RGA_DMA 端到端零拷贝 + NEON SIMD** 集成加速优化，核心内容如下：

- 🧩 **端到端零拷贝数据通路 (Camera → RGA → NPU)**：新增 `DmaBufferPool`（DRM dumb buffer + PRIME fd + mmap，预分配循环复用），打通 **V4L2 摄像头 → RGA 预处理 → NPU 推理** 全链路，全程无 memcpy：
  - `V4l2ZeroCopyCapture`：V4L2 MMAP + EXPBUF 零拷贝采集，替代 OpenCV VideoCapture 摄像头路径；
  - `RgaPreprocessor`：RGA `wrapbuffer_fd` DMA→DMA，单 pass `imresize` + `imcvtcolor` 完成缩放 + BGR→RGB；
  - `RKNNDetector::infer_zero_copy`：`rknn_create_mem_from_fd` + `rknn_set_io_mem`，NPU 直接读取 DMA 缓冲。
  - 保留回退路径：视频文件 / OpenCV 摄像头 → cv::Mat → DMA 桥接（仅一次拷贝）。
- ⚡ **预处理性能提升**（1080p→640p @30fps，代码注释基准）：OpenCV CPU `resize+cvtColor` 约 8ms/帧 / CPU 92% → RGA virt→DMA 约 1.5ms/帧 / CPU ~5% → **RGA DMA→DMA 约 0.8ms/帧 / CPU ~2%**。
- ⚡ **巅峰fps性能提升**：将近***<u>16fps</u>***（需要指令：-p 2 -n 14 -P 3，实现NPU 3核满载）
- 🎛️ **NPU 多核绑定**：`rknn_set_core_mask` + 新 CLI `--npu-cores auto|0|1|2|0,1|0,1,2`，配合多线程流水线按核心分配推理线程。
- 🔧 **后处理 NEON SIMD 加速**：`decode_rtdetr_output` 集成 NEON，进一步降低 300 框解码耗时。
- 🛠️ **构建与测试升级**：CMake 升级为 C++17 + `-O3`/OpenMP/NEON 编译选项，支持 `RK3588_TOOLCHAIN` 交叉编译，新增链接 `librga/libdrm/libv4l2/libv4lconvert`；可执行文件更名为 **`rtdetr_pipeline`**，新增单元测试 `test_unit` 与板端健壮性/压测脚本 `test_robustness.sh`。
- 📦 **新增/重构源码模块**：新增 `drm_alloc`、`rga_utils`、`v4l2_capture`；重构 `npu_pipeline`、`rknn_detector`、`postprocess`、`types`、`main` 等。

**V3 快速上手**：

```bash
cd cpp/
mkdir -p build && cd build
cmake .. && make -j4

# 1. V4L2 摄像头零拷贝实时推理（3 个 NPU 线程 + 双核绑定示例）
./rtdetr_pipeline -m rtdetr_r18.rknn -d /dev/video0 -W 1920 -H 1080 -o output.mp4

# 2. 视频文件推理[最佳性能]（当前仍为 OpenCV 软解码 + DMA 桥接，MPP 硬解见 TODO）
./rtdetr_pipeline -m rtdetr_r18.rknn -v test.mp4 -o result.mp4 -p 2 -n 14 -P 3 -c -0.13f  # 模型输出未归一化，阈值范围 > -1

# 3. 处理单张图片
./rtdetr_pipeline -m rtdetr_r18.rknn -i uav.jpg -o result_detect.jpg -c -0.13f

# 4. 仅显示性能（不保存视频）
./rtdetr_pipeline -m rtdetr_r18.rknn -v test.mp4

# 5. 查看完整帮助
./rtdetr_pipeline -h
```
