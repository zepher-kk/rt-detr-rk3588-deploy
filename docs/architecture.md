# v4 架构

## 1. 总览

```text
图片 / 图片列表 / 视频
        │
        ▼
  [读取线程]  （视频：OpenCV/FFmpeg 软解 + clone）
        │
        ▼
   queue_raw  （有界，背压）
        │
        ▼
  [pre workers]  RGA：letterbox resize + BGR→RGB → 640×640 uint8 画布
        │
        ▼
   queue_npu
        │
        ▼
  [npu workers]  每 worker 独立 RKNN context + 独立 NPU 核
        │        ├ NEON: uint8 RGB → FP16 ×(1/255)
        │        └ rknn_set_io_mem → rknn_run
        ▼
   queue_post
        │
        ▼
  [post workers]  解码 / 阈值过滤 / 计数
```

## 2. 模块

| 模块 | 文件 | 职责 |
|---|---|---|
| 有界队列 | `cpp/include/pipeline.h` | 阻塞队列、毒丸终止、`shared_ptr` 传帧 |
| 流水线 | `cpp/src/pipeline.cc` | 三级 worker、绑核、分段计时 |
| 预处理 | `cpp/src/rga_utils.cc` | RGA letterbox + 转色，CPU 回退 |
| 推理（单模型） | `cpp/src/rknn_detector.cc` | context、io_mem、输出预分配 |
| 推理（v4.1 拆分） | `cpp/src/split_detector.cc` | 双 context、int8 反量化 + 布局重排 |
| DRM/内存池 | `cpp/src/drm_alloc.cc` | DRM dumb + PRIME 或 cached heap |
| 输入 | `cpp/src/main.cc` | 单图 / 列表 / 视频 / 时延模式 |

## 3. 关键设计

### 3.1 每 worker 独立 context + 独立核

单实例使用三核（`CORE_0_1_2`）；多实例时按 `worker_index % 3` 绑定
`CORE_0 / CORE_1 / CORE_2`。

实测：3 实例共享三核只有 10.2 FPS，各绑一核 18.0 FPS。

### 3.2 输入路径

FP16 单模型的原生输入是 **FP16 NHWC 640×640×3**（`rknn_query(NATIVE_INPUT_ATTR)`，
2,457,600 字节），不是 uint8。因此：

1. 用 `rknn_create_mem()` 分配 FP16 输入缓冲
2. `rknn_set_io_mem(..., pass_through=1)` 绑定一次
3. 每帧用 NEON 把 `uint8 RGB` 转成 `FP16 × (1/255)`

比 `rknn_inputs_set` 的通用转换快约 6.8 ms/帧。

### 3.2b v4.1 拆分的推理路径

`--decoder` 指定后，worker 使用 `SplitDetector`（两个 context 绑同一个核）：

```text
uint8 RGB 640×640  ──(^0x80)──>  backbone INT8 输入(1.2 MB)
                                      │
                              backbone 推理
                                      │
        3 × INT8 NCHW 特征(2.1 MB)  ──(反量化 + NCHW→NC1HWC2)──>
                                      │
                    decoder FP16 输入(4.3 MB, 3 个 io_mem)
                                      │
                              decoder 推理
                                      │
                                 y = (150, 14)
```

- backbone 输入是 1/4 大小的 uint8，只需一次符号位翻转；
- 交接是纯顺序读写（2.1 MB 读 + 4.3 MB 写），实测 <1 ms；
- 两个模型同核串行，不存在跨核同步；
- 启动时用 `custom_string` 校验两模型是否配套。

### 3.3 为什么不用 DMA 直接喂 NPU

DRM dumb buffer 的 `mmap` 是 uncached，`rknn_inputs_set` 读取它要 **58 ms/帧**。
而 RGA 只能输出 8bit，原生输入又是 FP16，物理零拷贝不成立。
因此预处理输出到 **cached heap**，再做一次显式 FP16 转换。

### 3.4 视频背压

读取线程 `push_frame()` 在队列满时阻塞，因此解码速度自动跟随推理速度，
不会无限堆积内存。`-F` 可额外按源帧率节流。

## 4. 计时口径

`[stage]` 行输出每帧各阶段平均耗时：

```
wait_raw | pre | wait_npu | npu | wait_post | post
```

以及 `e2e`（从入队到完成）。注意：**压测时 e2e 包含排队时间**
（队列 16×2+1 ≈ 33 帧在飞），不是单帧延迟。单帧延迟请用 `--latency`。
