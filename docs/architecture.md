# 系统架构

项目由输入、预处理、推理、后处理和输出五个阶段组成。摄像头路径尽量保持
DMA 零拷贝，视频文件路径使用 GStreamer 与 RK MPP 完成硬解和硬编。

```text
V4L2 camera ──> V4l2ZeroCopyCapture ──┐
                                      ├──> RGA preprocess ──> RKNN/NPU
video/image ──> GStreamer + MPP ──────┘                         │
                                                                v
display/encoded output <── draw_results <── NEON postprocess
```

## 模块

| 模块 | 职责 |
|---|---|
| `v4l2_capture` | V4L2 采集、格式协商、超时和缓冲区生命周期 |
| `gst_io` | 视频/JPEG 硬解、视频硬编、CFR/VFR 帧率处理 |
| `rga_utils` | 图像缩放与颜色转换，DMA 或虚拟地址输入 |
| `drm_alloc` | DMA buffer 分配、PRIME fd 和映射 |
| `rknn_detector` | 模型加载、输出形状识别和零拷贝推理 |
| `postprocess` | RT-DETR 解码、置信度过滤、NMS 和画框 |
| `npu_pipeline` | 多 worker 调度、有界队列和批量图片复用 |
| `main` | 命令行解析、输入路由、显示与输出 |

## 数据通路

摄像头帧优先使用 DRM/V4L2 DMA buffer。RGA 输出 640×640 RGB DMA buffer，
RKNN 通过 `rknn_create_mem_from_fd` 和 `rknn_set_io_mem` 直接读取。视频文件
由 MPP 解码为 NV12，再转换为 BGR；输出默认使用 H.264 硬编。

当 RGA 无法安全处理源 stride 或格式时，模块回退到紧凑 DMA 或 CPU 路径。
回退路径优先保证检测正确性，性能测试需要注明是否触发回退。

## 模型输出

当前运行时同时支持 150 和 300 queries。实现根据 boxes/logits 的元素数
配对 query 数，并推导类别数；不再假定输出形状固定为 `300×4` 和
`300×10`。

模型输入默认是 640×640。`RTDETR_INPUT_SIZE` 可在 CMake 构建时覆盖，但
运行时模型输入、预处理尺寸和编译尺寸必须一致。
