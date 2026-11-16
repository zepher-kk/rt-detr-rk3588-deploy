# RT-DETR 在 RK3588 上的移植与优化记录（v4.0.0 → v4.1.0）

> 内部工程记录。目标是把 v3 的 Gather 展开方案换成 **native GridSample（GPU）**，
> 修复过程中的一个静默编译缺陷，并最终实现 **backbone+neck INT8 / decoder FP16**
> 的拆分部署。
>
> 记录时间：2026-11-09 ~ 2026-11-10
> 平台：RK3588（LubanCat），rknn-toolkit2 2.5.0 / librknnrt 2.5.0，Mali G610 OpenCL

---

## 目录

- [0. 结论速览](#0-结论速览)
- [1. 背景与目标](#1-背景与目标)
- [2. 耗时定位：怎么知道慢在哪](#2-耗时定位怎么知道慢在哪)
- [3. 模型图改造（导出期）](#3-模型图改造导出期)
- [4. 缺陷一：6 维广播 Mul 被编译成全 0](#4-缺陷一6-维广播-mul-被编译成全-0)
- [5. INT8 路线探索：为什么最终必须拆模型](#5-int8-路线探索为什么最终必须拆模型)
- [6. v4.1 拆分架构](#6-v41-拆分架构)
- [7. C++ 工程化](#7-c-工程化)
- [8. 调度与性能工程](#8-调度与性能工程)
- [9. 评估方法与结果](#9-评估方法与结果)
- [10. 复现清单](#10-复现清单)
- [11. 踩坑索引](#11-踩坑索引)
- [12. 经验与结论](#12-经验与结论)
- [13. 附录](#13-附录)

---

## 0. 结论速览

两套可部署配置，共用一套 C++ 代码（`--decoder` 切换）：

| 配置 | 形态 | AP (VisDrone val 548) | 单帧 e2e | 3 worker | 4 worker | 体积 |
|---|---|---|---|---|---|---|
| FP16 单模型（v4.0.1） | 1 个模型 | **0.1985** | 122 ms | 17.99 FPS | 18.53 FPS | 46.6 MB |
| INT8 拆分（v4.1.0） | 2 个模型 | **0.1882** | **70 ms** | **29.9 FPS** | **33.3 FPS** | 17.5 + 8.1 MB |

三个决定性的技术转折：

1. **GridSample 必须走 GPU**（`op_target`），否则运行时直接报 `Unsupport CPU op`；这是 v4 性能的基础（335 ms → 186 ms）。
2. **RKNN 2.5 会把 decoder 里的 6 维广播 `Mul` 编译成全 0**，模型能跑、速度还快，但输出全错。修复方式是把网格运算写成 4 维。
3. **单模型内混合精度在 RKNN 2.5 上不可行**，INT8 只能通过“拆两个模型”实现；拆分后反而比单模型快 1.7 倍。

---

## 1. 背景与目标

### 1.1 硬件与软件基线

| 项 | 值 |
|---|---|
| 板卡 | RK3588（LubanCat），Linux 6.1.84，aarch64 |
| NPU | 3 核，1 GHz，6 TOPS（INT8） |
| GPU | Mali-G610，OpenCL 3.0（Mali blob） |
| 推理运行时 | `/usr/lib/librknnrt.so` 2.5.0 |
| 模型工具链 | rknn-toolkit2 2.5.0（PC/WSL） |
| 预处理 | RGA（letterbox + BGR→RGB），DRM/PRIME |

### 1.2 v3 的技术债

v3.0.5 的路线是：

- 把 deformable attention 里的 `F.grid_sample` **展开成 Gather 图**，因为 RKNN 2.3.x 的
  runtime 不支持 GridSample；
- 用 INT8 全量化 + 按类阈值压精度损失；
- 最终 28.43 MB、AP 0.2049、1080p 25.04 FPS。

代价：

- 展开图产生大量 Gather/索引算子，decoder 单帧 255 ms；
- INT8 全量化精度受限，需要按类阈值来“找补”；
- 只能整模型一个精度，无法按区域选择。

### 1.3 v4 的目标

1. 用 **native GridSample**（RKNN 2.5 + GPU）替代 Gather 展开；
2. 精度与 PyTorch 对齐（逐位或 <1e-3）；
3. C++ 多实例吞吐最大化（利用 3 个 NPU 核）；
4. 在能跑的前提下，探索 INT8 的可行性。

---

## 2. 耗时定位：怎么知道慢在哪

> 这一章讲方法，下一章才讲具体改造。顺序很重要：
> 本项目里至少两次“以为的瓶颈”和“实际的瓶颈”不一致。

### 2.1 为什么要先 profile，而不是先改

直觉上会认为“RT-DETR 慢在 Transformer，所以要优化 attention”。
但实测下来，真正的大头是：

- **布局转换**（transpose / reshape / NCHW↔序列）——两次改写合计 29.7 ms；
- **CPU 边界**（decoder 中间有一个算子落 CPU）——固定 27 ms；
- **设备选择**（GridSample 被分到 CPU）——直接不可用。

如果一上来就改 attention 的数学形式，这些都不会消失。所以顺序应该是：

```
组件级（哪一段慢） → 算子级（哪个算子/哪个设备） → 设备处理级（搬运/转换）
        ↓                      ↓                          ↓
   决定改哪里            决定能不能改算法            决定改数据通路
```

### 2.2 组件级 profile：先切三段

目标：把整模型拆成可独立测量的几段，找出“大头”。

#### 方法 A：按模块前缀切子图（最准确）

ONNX 的节点名带模块前缀（`/model.0/...`、`/model.26/...`），可以按前缀
`extract_model` 出子图，分别编译、分别测：

```python
# 示意：把 model.0-8（backbone）单独切出来
onnx.utils.extract_model(src, 'backbone.onnx', ['images'],
                         ['/model.8/.../output'])
```

注意：切点必须选在张量依赖完整的位置（本项目在 neck 输出处切过一次，
见第 6 章），否则需要补输入/输出声明。

#### 方法 B：整模型 A/B 差值（最省事）

不改图，只改一个编译参数，用**整模型耗时差**推断某一段的代价：

| 对比 | 差值 | 结论 |
|---|---|---|
| `Mul=npu` vs `Mul=cpu` | 130.2 → 157.3 ms（+27 ms） | decoder 内的 CPU 边界很贵 |
| GridSample 有/无 GPU | 不可用 | GPU 是硬约束 |
| FP16 vs INT8（拆分） | 131 → 70 ms | backbone 占大头 |

这个方法便宜，但只能得到“差值”，不能定位到具体算子。

#### 方法 C：官方分段数据 + 实测校准

项目早期记录过分段（未公开的 prof 构建）：

| 阶段 | 耗时 |
|---|---|
| Backbone（model.0–8） | ~45 ms |
| AIFI（model.9） | ~5 ms |
| Neck（model.10–25） | ~27 ms |
| Decoder/Head（model.26） | ~108.5 ms → 优化后 ~46 ms |

结论很明确：**decoder 是大头**。后面的改造（4 维网格、value channel-first、
拆分 INT8）都是围绕它做的。

> 注意：这些分段数字来自不同构建，不能和最终 131 ms 直接相加；
> 它们只用于判断“量级”。

### 2.3 算子级 profile：看设备分配表 + 做 A/B

#### 2.3.1 读 build log 的算子分配表

RKNN 编译日志里有一张表：

```
D RKNN: [17:42:04.327] 0    InputOperator      FLOAT16  CPU    (1,3,640,640)
D RKNN: [17:42:04.327] 132  Transpose          FLOAT16  CPU    (1,10,1,8400)
D RKNN: [17:42:04.327] 133  ReduceMax          FLOAT16  CPU    (1,8400,1,10)
D RKNN: [17:42:04.327] 134  TopK               FLOAT16  CPU    (1,8400,1,1)
D RKNN: [17:42:04.327] 136  Gather             FLOAT16  CPU    (1,256,1,8400)
```

解析它就能得到“每个算子在哪里、什么精度”。本项目的统计：

| 版本 | NPU | GPU | CPU | INT8 |
|---|---|---|---|---|
| v4.0.1（FP16） | 333 | 9（GridSample） | 18 | 0 |
| v4.1 backbone（INT8） | 全部 | 0 | 0 | 是 |

CPU 上的 18 个是：`InputOperator`、`OutputOperator`、`TopK`、`ReduceMax`、
`Gather`×3、`Transpose`×2、`Reshape`×2、`Log`——即 **query 选择（topk）** 和 I/O
胶水算子，属于结构性限制（RKNN 不支持 NPU 上做 TopK）。

#### 2.3.2 用 A/B 定位“昂贵的那一类算子”

这是本项目最有效的手段。做法：固定其它条件，只改 `op_target` 里一个 op 的
设备，看整模型差多少。

| 实验 | 结果 | 解读 |
|---|---|---|
| `Mul=npu` | 130.2 ms | 基准 |
| `Mul=cpu`（全部） | 157.3 ms | +27.1 ms |
| **只把 6 个 `cross_attn/Mul_1` 放 CPU** | **157.6 ms** | 与全量几乎相同 |
| 把 `Mul` 按 tensor 名放 CPU，同时保留 `Mul=npu` | 130.2 ms | 被 op_type 级覆盖，没生效 |

关键结论：**代价与 CPU 上放几个算子无关**。放 1 个和放 40 个都是 +27 ms，
因为成本来自 decoder 内部的 CPU/NPU 边界（3 层 × 约 9 ms 的同步与布局转换）。

> 这条结论直接改变了方案：不能用“把少量算子放 CPU”来换正确性，
> 代价是固定的；要么不切边界，要么整段拆出去（后来的 v4.1）。

#### 2.3.3 逐层 profile 不可用（记录为已知限制）

RKNN 的 `RKNN_FLAG_COLLECT_PERF_MASK` 可以输出逐层耗时，但在本模型上
（v4 的 native GridSample 图）会 SIGSEGV，崩在 `rknn_run`。

```
[1]    12345 segmentation fault  ./infer
```

所以本项目的耗时数据全部来自：

- 分段打点（C++ 里 `t_set/t_run/t_get`）；
- 墙钟压测（`--latency`、`-N` 帧）；
- A/B 差值。

### 2.4 设备处理级 profile：搬运和转换

即使算子都选对了设备，数据在设备之间**搬运**也可能成为瓶颈。
这一类问题不会出现在算子表里，只能靠分段计时发现。

#### 2.4.1 输入路径

| 方案 | 代价 | 说明 |
|---|---|---|
| `rknn_inputs_set`（uint8 → 模型原生） | ~6.8 ms/帧 | 每次都有通用格式转换 |
| `rknn_set_io_mem` + 自行转换 | ~0.5 ms/帧 | v4 采用；绑定一次，之后只写内存 |
| 直接喂 DRM uncached 缓冲 | **58 ms/帧** | `mmap` 是 uncached，读一次极慢 |

结论：预处理输出到 cached heap，自己做显式转换，再 `memcpy` 进
`rknn_create_mem` 的输入缓冲。

#### 2.4.2 输出路径

| 方案 | 代价 | 说明 |
|---|---|---|
| `want_float=1` | 每个输出做一次 dequant + fp32 拷贝 | 输出小的话可接受（本项目 y 只有 150×14） |
| `want_float=0` | 无转换，但要自己处理布局和反量化 | 拆分交接时采用（见第 7 章） |

#### 2.4.3 预处理（RGA vs CPU）

| 方案 | 代价 |
|---|---|
| CPU letterbox + BGR→RGB（cv2） | 4~6 ms/帧，且占 CPU |
| RGA（硬件 2D） | <1 ms/帧，且与 NPU 并行 |

RGA 的收益不只是省时间，还把预处理从关键路径上拿掉了（三级流水线）。

#### 2.4.4 量化与设备的耦合

INT8 不只是“算得快”，还会改变**输入格式**：

- FP16 模型的输入是 `FP16 NHWC`（2.4 MB/帧），C++ 要做一次 `u8→fp16 /255`；
- INT8 模型的输入是 `INT8 NHWC`（1.2 MB/帧），只需 `u8 ^ 0x80`（符号位）。

后者的转换量只有前者的一半，而且更简单——这是拆分方案额外的一个小收益。

### 2.5 把 profile 结论变成改造动作

| 定位到的瓶颈 | 改造动作 | 实测收益 |
|---|---|---|
| GridSample 只能用 CPU / 展开成 Gather | 改用 native GridSample + `op_target=gpu` | 335 → 186 ms |
| decoder 里 12 宽 transpose 多 | 重写 attention，逐 level 累加 | −13.1 ms |
| value 的 NCHW→sequence 布局往返 | `--value-nchw`，保持 channel-first | −16.6 ms |
| grid 通路的 6 维 Mul 被编译坏 | 改 4 维（第 4 章） | 让 Mul 留在 NPU，比 CPU 省 26 ms |
| backbone 仍占 ~45 ms | 拆模型 + INT8 | 131 → 70 ms（整链） |
| 多实例共享三核扩展性差 | 每 worker 绑一核（数据并行） | 8 → 18/30 FPS |

### 2.6 profile 的陷阱

1. **模拟器 ≠ 板端**：`rknn.eval_perf` 在 PC 上是模拟值，只能看相对趋势；
2. **板端被占用**：本项目遇到过另一个项目跑满 NPU，17:03 之后的所有计时全部作废；
   测之前先看 `ps`、NPU load、温度；
3. **热漂移**：同一配置长时间压测会有 3%~5% 的波动；对比实验要交替跑（A/B/A/B）；
4. **短时峰值**：`--latency` 只跑 25 帧，第一次可能偏高（冷启动），取三次中位数；
5. **口径混用**：`--latency`（单帧）与 `[stage] e2e`（含排队）不能放在一起比；
6. **只看总数**：同一模型“3 worker 17.99 FPS”和“1 worker 8.07 FPS”不是线性关系，
   说明瓶颈在核间调度而不是单帧计算——这时候要改的是并行方式，不是模型。


## 3. 模型图改造（导出期）

导出脚本：`python/export/export_transpose_opt.py`。

### 3.1 deformable attention 的重写

原始 `multi_scale_deformable_attn_pytorch` 对每个 level 做：

```python
# 原始实现（简化）
sampling_grids = 2 * sampling_locations - 1
for level, (H_, W_) in enumerate(value_spatial_shapes):
    value_l = value_list[level].flatten(2).transpose(1, 2).reshape(
        bs * num_heads, embed_dims, H_, W_)
    grid_l = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
    sampled = F.grid_sample(value_l, grid_l, ...)
    w = attention_weights[:, :, :, level].transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_points)
    term = (sampled * w).sum(-1)
    out = term if out is None else out + term
```

问题在于 `stack`/`flatten` 会产生大量 12 宽（`num_heads * num_points * 2`）的
transpose，RKNN 会退化成多个小算子。

v4 的改法（`msda_v4`，配合 `--value-nchw`）：

```python
def msda_v4(value, value_spatial_shapes, sampling_locations, attention_weights):
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, _, _, num_points, _ = sampling_locations.shape
    # (bs, sumHW, heads, dim) -> (bs, heads, dim, sumHW)
    v = value.permute(0, 2, 3, 1)
    sampling_grids = _grid_from_locations(sampling_locations)

    out = None
    start = 0
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        n = int(H_ * W_)
        v_l = v[..., start:start + n].reshape(
            bs * num_heads, embed_dims, int(H_), int(W_))
        start += n
        grid_l = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampled = F.grid_sample(v_l, grid_l, mode='bilinear',
                                padding_mode='zeros', align_corners=False)
        w = attention_weights[:, :, :, level].transpose(1, 2).reshape(
            bs * num_heads, 1, num_queries, num_points)
        term = (sampled * w).sum(-1)
        out = term if out is None else out + term

    return out.reshape(bs, num_heads * embed_dims, num_queries).transpose(1, 2).contiguous()
```

两处等价重写：

1. `value` 保持 channel-first（`permute(0,2,3,1)` 后按序列切片），
   避免 `value_proj` 之后再插一次 NCHW→sequence 的 transpose；
2. 逐 level 累加替代 `stack`，消掉 12 宽 transpose。

收益：`-13.1 ms`（transpose）+ `-16.6 ms`（布局往返）。

### 3.2 4 维网格运算（后面会解释为什么必须）

```python
def _grid_from_locations(sampling_locations):
    """grid = 2 * locations - 1.

    With `_GRID_4D` the arithmetic is done on a 4-D view
    (bs, queries, heads*levels*points, 2): RKNN 2.5 miscompiles the 6-D
    broadcast form and returns zeros for the Mul that produces the locations.
    """
    if not _GRID_4D:
        return 2 * sampling_locations - 1
    bs, queries, heads, levels, points, _ = sampling_locations.shape
    flat = sampling_locations.reshape(bs, queries, heads * levels * points, 2)
    return (2 * flat - 1).reshape(bs, queries, heads, levels, points, 2)
```

以及 `MSDeformAttn.forward` 里的定位计算：

```python
elif num_points == 4:
    if _GRID_4D:
        bs_, q_ = sampling_offsets.shape[:2]
        h, l, p = self.n_heads, self.n_levels, self.n_points
        off4 = sampling_offsets.reshape(bs_, q_, h * l * p, 2)
        ref4 = refer_bbox.reshape(bs_, q_, 1, 4)
        loc4 = ref4[..., :2] + off4 * ref4[..., 2:]
        sampling_locations = loc4.reshape(bs_, q_, h, l, p, 2)
    else:
        add = sampling_offsets * refer_bbox[:, :, None, :, None, 2:]
        sampling_locations = refer_bbox[:, :, None, :, None, :2] + add
```

> 这两段就是第 4 章缺陷的修复。它们与原图**数学等价**，
> ONNX 输出逐位相同（`max|d| = 0`），但 RKNN 编译结果完全不同。

### 3.3 折叠偏移尺度

`0.5 / n_points = 0.125` 预先乘进 `sampling_offsets` 的权重和偏置：

```python
def fold_offset_scaling(model, n_points=4):
    factor = 0.5 / n_points
    decoder = model.model[-1]
    with torch.no_grad():
        for layer in decoder.decoder.layers:
            ca = layer.cross_attn
            ca.sampling_offsets.weight.mul_(factor)
            ca.sampling_offsets.bias.mul_(factor)
    return factor
```

效果：图上少掉 `Div(n_points)` 和 `Mul(0.5)` 两个节点，ONNX 与原图逐位相同。
（注意：它**没有**修复第 4 章的缺陷，后面会解释。）

### 3.4 GridSample 在 RKNN 2.5 上的前后处理：三种方案对比

#### 3.4.1 问题定义

ONNX 的 `GridSample` 语义很固定：

```
输入 value: (N, C, H, W)
输入 grid : (N, H_out, W_out, 2)   最后一维是 (x, y)，归一化到 [-1, 1]
属性      : mode=bilinear, padding_mode=zeros, align_corners=0
输出      : (N, C, H_out, W_out)
```

deformable attention 里的用法是逐 level 的：

```
value: (bs*heads, C, H, W)
grid : (bs*heads, num_queries, num_points, 2)
```

即“每个 query 的每个采样点，在对应 level 的特征图上做一次双线性插值”。

RKNN 2.3.x 没有这个算子，2.5.0 才有（且只在 GPU 上）。因此历史上出现了三种处理方式。

#### 3.4.2 方案 A：Gather 展开（v3 路线）

把双线性采样手工拆开：

```python
# 伪代码：把归一化坐标换成像素坐标，再取 4 邻域加权
gx = (grid_x + 1) * (W - 1) / 2
gy = (grid_y + 1) * (H - 1) / 2
x0, y0 = floor(gx), floor(gy)
wx, wy = gx - x0, gy - y0

v00 = Gather(value, y0 * W + x0)
v01 = Gather(value, y0 * W + x0 + 1)
v10 = Gather(value, (y0 + 1) * W + x0)
v11 = Gather(value, (y0 + 1) * W + x0 + 1)

out = (v00 * (1-wx) * (1-wy) + v01 * wx * (1-wy)
     + v10 * (1-wx) * wy     + v11 * wx * wy)
```

- **前处理**：坐标归一化 → 像素坐标 → `floor`/`Clip` → 4 个整型索引；
- **后处理**：4 次 `Gather` + 8 次乘加；
- 全部是 NPU 支持的算子，**不依赖 GPU、可以 INT8**。

代价：

| 项 | 数值 |
|---|---|
| 图规模 | 每个 level × 每个 head × 每个 point 都展开，节点数暴涨 |
| 索引算子 | `Floor`/`Clip`/`Cast`/`Mul` 大量出现，NPU 利用率低 |
| decoder 单帧 | ~255 ms（整模型） |
| INT8 精度 | 索引/权重被量化后误差明显，需要按类阈值找补 |

#### 3.4.3 方案 B：native GridSample（v4 路线）

直接保留 ONNX 的 `GridSample`，交给 RKNN 2.5 的 GPU 实现。

**前处理**（`msda_v4`）：

```python
def msda_v4(value, value_spatial_shapes, sampling_locations, attention_weights):
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, _, _, num_points, _ = sampling_locations.shape

    # 关键：value 保持 channel-first，按序列轴切片，避免 NCHW->sequence 的 transpose
    v = value.permute(0, 2, 3, 1)

    # grid = 2 * locations - 1（4 维，见第 4 章）
    sampling_grids = _grid_from_locations(sampling_locations)

    out = None
    start = 0
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        n = int(H_ * W_)
        v_l = v[..., start:start + n].reshape(
            bs * num_heads, embed_dims, int(H_), int(W_))
        start += n

        # 每个 level 的 grid: (bs*heads, Q, P, 2)
        grid_l = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)

        sampled = F.grid_sample(v_l, grid_l, mode='bilinear',
                                padding_mode='zeros', align_corners=False)

        # 后处理：按 attention weight 加权求和
        w = attention_weights[:, :, :, level].transpose(1, 2).reshape(
            bs * num_heads, 1, num_queries, num_points)
        term = (sampled * w).sum(-1)
        out = term if out is None else out + term

    return out.reshape(bs, num_heads * embed_dims, num_queries).transpose(1, 2).contiguous()
```

- **前处理**：`grid = 2*loc - 1` + 按 level 切片 + `transpose(1,2).flatten(0,1)` 得到
  `(bs*heads, Q, P, 2)`；`value` 只做 `permute + reshape`，不改变内存布局；
- **后处理**：`sampled * w`（在通道维广播）→ `sum(-1)` → 三个 level 相加；
- 采样本身交给 GPU，NPU 只负责前后处理。

与方案 A 相比，**图小了近一个数量级**，索引算子全部消失。

#### 3.4.4 方案 C：自定义算子（未采用）

运行时错误信息里其实给了这条路：

```
Unsupport CPU op: GridSample in this librknnrt.so, please try to register
custom op by calling rknn_register_custom_ops ...
```

可以自己注册 CPU/GPU kernel。但没有采用：

- native 实现已经可用，且在 GPU 上；
- 自定义算子要自己维护正确性和性能，跨版本兼容性差；
- 一旦注册失败，整条链路不可用。

#### 3.4.5 前后处理逐项对比

| 环节 | A. Gather 展开 | B. native GridSample |
|---|---|---|
| 坐标变换 | 归一化 → 像素 → floor/clip → 4 索引 | 归一化即可（`2*loc-1`） |
| value 布局 | 需要转成 sequence / 或按索引 gather | 保持 NCHW（我们的 `--value-nchw`） |
| grid 布局 | 拆成 4 组索引，逐点计算 | 整体张量 `(bs*heads, Q, P, 2)` |
| 采样 | 4× `Gather` + 8× 乘加 | 1× `GridSample`（GPU） |
| 加权 | `Mul` + `ReduceSum` | 同左（`sampled * w` + `sum`） |
| 设备 | NPU（纯） | NPU 前后处理 + **GPU 采样** |
| dtype | 可 INT8 | **只能 FP16** |
| 图规模 | 大（索引/权重算子多） | 小 |
| decoder 单帧 | ~255 ms | 46 ms（估算）→ 实测整模型 131 ms |

> “decoder 46 ms”是分段推算值（benchmarks 里的分段表），不是单独 profile；
> RKNN 2.5 的逐层 profile（`RKNN_FLAG_COLLECT_PERF_MASK`）在这张图上会段错误。

#### 3.4.6 RKNN 2.5 上 native GridSample 的边界条件（实测）

| 条件 | 表现 | 应对 |
|---|---|---|
| 没指定设备 | auto 分到 CPU，运行时报 `Unsupport CPU op: GridSample` | `--op-target-map 'GridSample=gpu'` |
| 模型是 INT8 | 运行时崩溃 `std::out_of_range map::at` | 整段 decoder 必须 FP16（→ 第 5/6 章） |
| grid 由 6 维广播 `Mul` 产生 | 该 Mul 被编译成全 0，采样点塌陷 | 改成 4 维（→ 第 4 章） |
| grid 张量直接作为模型输出 | 运行时段错误（GPU `cl_mem` 与“同时作为输出”冲突） | 只 dump 非 GridSample 直接输入 |
| `mode` 非 bilinear / padding 非 zeros | 未验证，不建议 | 只用 bilinear + zeros |
| GPU 算子与 NPU 算子交替 | 每次切换有同步/布局转换成本 | 尽量让 GPU 段连续（本模型只有 3 个 GridSample） |

#### 3.4.7 为什么最终选 native GridSample

1. **图规模**：Gather 展开的节点数是 native 的数倍，NPU 上全是零碎索引算子；
2. **布局成本**：native 只需要一次 `permute/reshape`，展开方案每次采样都要构造索引；
3. **可维护性**：native 是工具链内置算子，升级 runtime 即可受益；
4. **代价可控**：整段 decoder 保持 FP16，只让 backbone+neck 走 INT8（第 6 章），
   AP 实测无损。

它带来的新约束（必须 GPU、必须 FP16、6 维 Mul 缺陷）都在第 4–6 章解决。

### 3.5 导出参数

```bash
python python/export/export_transpose_opt.py --weights best.pt --value-nchw \
    --fold-offset-scale --msda-4d-grid \
    --output model/rtdetr_tpopt2_4d.onnx
```

| 参数 | 作用 |
|---|---|
| `--value-nchw` | value 保持 channel-first（3.1） |
| `--fold-offset-scale` | 折叠 0.5/n_points（3.3） |
| `--msda-4d-grid` | 4 维网格运算（3.2），**不能漏** |
| `--num-queries` | 裁剪 query 数（默认 300） |
| `--decoder-layers` | 裁剪 decoder 层数（默认全部） |

---

## 4. 缺陷一：6 维广播 Mul 被编译成全 0

这是整个项目最隐蔽的一个问题：**模型能跑、速度正常、encoder 输出完全正确，
只有 decoder 错了。**

### 4.1 现象

| 输出 | Mul=cpu（正确） | Mul=npu（错误） |
|---|---|---|
| `out[0]` pred_logits `y` | max +0.9907 | max +0.9922，高置信 query 平均差 0.069 |
| `out[2]` dec_scores（raw） | max **+2.7969** | max **−1.8008** |
| `out[3]` / `out[4]` encoder | — | **逐位相同** |

`Mul` 是 `op_target` 的一个键。把它放 CPU 就正确，放 NPU 就错——
而 encoder 完全不受影响，说明问题精确限定在 decoder 的 cross-attention。

### 4.2 定位方法：把中间张量加成图输出

最初的猜测都是错的（尺度错、w/h 互换、广播形状……），
最后用了一个笨办法：**把 layer 0 cross-attn 的中间张量直接加成 ONNX 的输出**，
上板 dump 出来和 ONNX Runtime 对比。

```python
# 只加这些"不是 GridSample 直接输入"的张量，否则运行时会段错误
NAMES = [
    (P + 'Div_output_0',      [1, 300, 8, 3, 4, 2]),
    (P + 'Mul_output_0',      [1, 300, 8, 3, 4, 2]),
    (P + 'Mul_1_output_0',    [1, 300, 8, 3, 4, 2]),
    (P + 'Add_output_0',      [1, 300, 8, 3, 4, 2]),
]
```

结果（uav.jpg）：

| 张量 | ONNX 正确值 | 板上（Mul=npu） | 判定 |
|---|---|---|---|
| `Div_output_0`（offsets / n_points） | −10.19 … +4.62 | −10.18 … +4.63 | ✅ 正常 |
| `Slice_output_0`（ref_wh） | q0 = [0.10501, 0.08963] | q0 = [0.10492, 0.08966] | ✅ 正常 |
| **`Mul_output_0`（offsets × ref_wh）** | −0.54 … +0.30 | **100% 全 0** | ❌ |
| `Add_output_0`（+ ref_xy） | 有偏移 | 每个 query **严格常数** = 参考框中心 | 后果 |

两个操作数都是对的，乘出来是全 0 —— 这不是数值精度问题，是编译器把这个
`Mul` 整个“优化”没了。

### 4.3 根因

被编译坏的表达式：

```
(1, 300, 8, 3, 4, 2) × (1, 300, 1, 1, 1, 2)
```

即 deformable attention 里“把偏移量按参考框宽高缩放”的那一步
（源码 `transformer.py:287`）：

```python
add = sampling_offsets / self.n_points * refer_bbox[..., 2:] * 0.5
```

RKNN 2.5 对这个 **6 维、且第二个操作数全是单例维** 的广播乘法返回全 0。

### 4.4 排除清单（都试过，都不行）

| 尝试 | 结果 | 说明 |
|---|---|---|
| 导出期折叠 `0.5/n_points` | ONNX 逐位等价，仍坏 | 坏的是剩下的 `offsets × ref_wh` |
| `Expand(ref_wh)` 成同形状再乘 | 仍坏 | 优化器把 Expand 又折回广播 |
| 常量尺度扫描 k = 0.25/0.5/1/2/4 | 无任何 k 能复现坏输出 | 排除“漏乘/多乘” |
| `ref_wh` 的 w/h 交换 | 不匹配（out2 与坏模型 mean|d| 0.876） | 排除轴序 |
| `Mul=gpu`（OpenCL） | 运行时段错误（RC=139） | GPU 不支持 Mul |
| 禁用 `fuse_mul_into_conv` | 不修，且 encoder 数值被带偏 | 融合规则不是元凶 |
| 按 tensor 名把 Mul 放 CPU | 有效 | 但 `Mul=npu`（op_type 级）会覆盖 tensor 级 key，测试时踩过 |

补充一个关键实验：只把每层 `cross_attn/Mul_1_output_0`（6 个 tensor，含 `_rs`）
放 CPU，输出与“全部 Mul 放 CPU”**逐位相同**。由此确认：

- 元凶就是 grid 通路上的这一族 Mul；
- **CPU 边界的代价与 CPU 上放几个算子无关**：放 1 个和放 40 个都是 +27 ms
  （157.3 vs 130.2 ms），因为代价来自 decoder 内部的同步与布局转换，3 层 × 约 9 ms。

### 4.5 修复：改 4 维

```python
# 原：6 维广播
add = sampling_offsets * refer_bbox[:, :, None, :, None, 2:]

# 新：先摊成 4 维，算完 reshape 回去
off4 = sampling_offsets.reshape(bs_, q_, h * l * p, 2)   # (1,300,96,2)
ref4 = refer_bbox.reshape(bs_, q_, 1, 4)                 # (1,300,1,4)
loc4 = ref4[..., :2] + off4 * ref4[..., 2:]              # (1,300,96,2)
sampling_locations = loc4.reshape(bs_, q_, h, l, p, 2)
```

`grid = 2*loc - 1` 同理改在 4 维上算。

验证：

- **ONNX 等价性**：与原图输出逐位相同（`max|d| = 0`）；
- **板上**：`out[2]` max 恢复 **+2.7969**，高置信 query 与正确版差 0；
- **速度**：131.05 ms（比坏的 130.2 ms 只慢 0.85 ms）。

### 4.6 为什么“把任意一步放 CPU”也能修（但不可取）

只要 grid 通路里有一个算子落在 CPU，这个 Mul 就不会被那套错误的融合/布局
处理捕获，结果就正确。但这**不是修复**，只是绕开：

- 代价固定 +27 ms（decoder 内部 CPU 边界）；
- 换一个模型/版本就可能复发；
- 正确做法是让表达式本身避开缺陷（4 维）。

### 4.7 后续影响

- v4.0.0 的“18.06 FPS”是在 decoder 损坏的情况下测的，**不能引用**；
- v4.0.1 修复后 17.99 FPS，`+0.9 ms` 换来正确性；
- 这个缺陷也解释了为什么“只比 encoder 头输出”会漏掉问题——
  早期的一致性检查用的是 `out[3]/out[4]`（encoder），它们逐位相同。

---

## 5. INT8 路线探索：为什么最终必须拆模型

目标：backbone+neck 走 INT8，decoder 保持 FP16。

### 5.1 纯 INT8：运行时崩溃

`do_quantization=True` 整模型量化，编译能过，运行时：

```
E RKNN: ... std::out_of_range map::at
```

GridSample 的 GPU 实现不接受 INT8 输入。

### 5.2 auto-hybrid：反而慢 3 倍

```bash
convert_native_gpu.py ... （内部 build(auto_hybrid=True)）
```

结果：179 个 FP16 张量，INT8/FP16 反复切换。

| 模型 | 推理 |
|---|---|
| 300q/3L auto-hybrid | **410.5 ms** |
| 150q/2L auto-hybrid | 267.0 ms |
| 同结构纯 FP16 | 131 / 105 ms |

### 5.3 区域混合：被工具链逐条堵死

思路：`hybrid_quantization_step1` 生成 cfg，把 `model.26/*` 标成 FP16，
其余 INT8。实际过程：

1. `hybrid_quantization_step1(op_target={'GridSample':'gpu'})`
   → `ValueError: Invalid key of 'GridSample' in 'op_target'!`
   （第一步的 op_target 白名单里没有 GridSample）
2. 在 step2 重新 `rknn.config(op_target=...)`
   → `ValueError: The config of 'step2' is inconsistent with 'step1'`
3. 直接改 `.quantization.cfg` 里某些 tensor 的 `dtype`
   → `The quantize_parameters['.../input_proj...']['dtype'] is not allowed to be modified!`
4. `custom_quantize_layers` 标算子
   → `Invalid operands name '.../Cast_2_output_0' in custom_quantize_layers!`
   （只接受可量化的 operand 名）
5. 最终编出来的区域混合模型：**GridSample 落到 CPU**，运行时
   `Unsupport CPU op: GridSample in this librknnrt.so`

### 5.4 其它限制

- `w8a16` 在 RK3588 不支持：`The quantized_dtype = 'w8a16' not support in 'rk3588'!`
- RKNN 2.5 的单模型构建只有一个全局 dtype，没有“逐层指定精度”的常规接口。

### 5.5 结论

**在 native GridSample 架构下，单模型内混合精度不可行。** 要实现 INT8，
只能把模型拆成两个：backbone+neck 单独 INT8，decoder 单独 FP16。
这也直接引出了 v4.1。

## 6. v4.1 拆分架构

### 6.1 切分点

decoder 模块（`/model.26/*`）的所有外部输入只有 3 张 neck 特征图：

```
/model.19/cv3/act/Mul_output_0   (1, 256, 80, 80)
/model.22/cv3/act/Mul_output_0   (1, 256, 40, 40)
/model.25/cv3/act/Mul_output_0   (1, 256, 20, 20)
```

合计 2,150,400 个元素，FP16 = **4.3 MB/帧**。这是一个干净的切点：
不跨越任何跨层依赖，也不涉及权重。

### 6.2 切图脚本

`python/export/split_decoder.py`，核心就是 `onnx.utils.extract_model` 两次：

```python
onnx.utils.extract_model(src, out_a, ['images'], feat_names)
onnx.utils.extract_model(src, out_b, feat_names, outputs)
```

结果：

| 模型 | 节点数 | 输入 | 输出 |
|---|---|---|---|
| backbone | 305 | `images` | 3 张特征图 |
| decoder | 751 | 3 张特征图 | `y` + 4 个辅助输出 |

**等价性验证**（ORT）：A→B 跑出来的 5 个输出与原图**逐位相同**（`max|d| = 0`）。

### 6.3 数据流

```
uint8 RGB 640×640 ──(^0x80)──> backbone INT8 输入 (1.2 MB)
                                     │
                             backbone 推理（INT8）
                                     │
       3 × INT8 连续 NCHW 特征 (2.1 MB)
                                     │  反量化 (q−zp)·scale
                                     │  + NCHW → NC1HWC2 重排
                                     ▼
                   decoder FP16 输入 (4.3 MB, 3 个 io_mem)
                                     │
                             decoder 推理（FP16，GridSample 在 GPU）
                                     ▼
                              y = (N, 14)
```

### 6.4 两个模型的编译参数

backbone（INT8，w8a8，per-channel）：

```bash
python python/quantize/convert_int8.py model/q150_backbone.onnx \
    model/rtdetr_v41_backbone_int8.rknn \
    --dataset calib.txt --custom-string v41-q150l2
```

decoder（FP16，多输入，不做图像归一化）：

```bash
python python/quantize/convert_native_gpu.py model/q150_decoder.onnx \
    model/rtdetr_v41_decoder_fp16.rknn --opt-level 0 \
    --disable-rules squeeze_nd_mul,fuse_mul_into_gemm \
    --op-target-map 'GridSample=gpu,Mul=npu' \
    --inputs /model.19/cv3/act/Mul_output_0,/model.22/cv3/act/Mul_output_0,/model.25/cv3/act/Mul_output_0 \
    --input-sizes 1x256x80x80\;1x256x40x40\;1x256x20x20 --no-normalize \
    --custom-string v41-q150l2
```

产物：17.5 MB + 8.1 MB（合计 25.6 MB）。

### 6.5 精度代价来自哪里

| 模型 | AP |
|---|---|
| FP16 300q/3L | 0.1985 |
| FP16 150q/2L（同结构） | 0.1882 |
| **INT8 拆分 150q/2L** | **0.1882** |

结论：**INT8 本身在 AP 上无损失**，0.0103 的差距全部来自结构裁剪
（query 300→150、decoder 3→2 层）。

---

## 7. C++ 工程化

### 7.1 代码结构

保持一份代码、两种配置：

```
cpp/src/rknn_detector.cc   单模型（v4.0.1）
cpp/src/split_detector.cc  拆分（v4.1）
cpp/src/pipeline.cc        worker 里二选一
```

`pipeline.cc`：

```cpp
RKNNDetector single;
SplitDetector split;
std::function<bool(const DmaBufferPtr&, FrameBundle&)> infer;
if (!cfg_.decoder_path.empty()) {
    if (!split.init(cfg_.model_path, cfg_.decoder_path, mask)) { ... }
    infer = [&split](const DmaBufferPtr& in, FrameBundle& out) {
        return split.infer(in, out);
    };
} else {
    if (!single.init(cfg_.model_path, mask)) { ... }
    infer = [&single](const DmaBufferPtr& in, FrameBundle& out) {
        return single.infer(in, out);
    };
}
```

命令行：`-m backbone --decoder decoder`；不给 `--decoder` 就走原单模型路径。

### 7.2 先探测属性，不要猜

`cpp/tests/attr_probe.cc` 打印标准与 native 的输入/输出属性。实测：

**backbone（INT8）**

```
input        images      NHWC      INT8     [1,640,640,3]  scale=0.0039215689 zp=-128
output[0]               NCHW      INT8     [1,256,80,80]  scale=0.041215625  zp=-121
native_out[0]           NC1HWC2   INT8     [1,16,80,80,16]
output[1]               NCHW      INT8     [1,256,40,40]  scale=0.065719679  zp=-124
output[2]               NCHW      INT8     [1,256,20,20]  scale=0.040004157  zp=-121
```

**decoder（FP16）**

```
input[0]     NHWC    FLOAT16  [1,80,80,256]
native_in[0] NC1HWC2 FLOAT16  [1,32,80,80,8]
（input[1]/[2] 同理，40×40、20×20）
```

三个关键信息：

1. backbone 的输入是 **INT8**（不是 uint8），量化参数 scale=1/255、zp=−128；
2. backbone 的输出是**量化后的 INT8**；
3. decoder 的原生输入是 **NC1HWC2，C2=8**，且没有 stride padding。

### 7.3 坑一：输入符号位（+0.5 亮度偏移）

RGA 输出的是 uint8。如果直接 `memcpy` 进 INT8 输入缓冲，
runtime 会按 `real = (q − zp) × scale = (q + 128) / 255` 解释，
相当于把整幅图亮度抬高 0.5。

诊断方法：把 backbone 的输出 dump 出来，和 Python 链路对比——相关系数只有 0.88
（正常应该 >0.99）。

修复：

```cpp
// backbone input is uint8 RGB NHWC but the tensor is INT8 with zp=-128,
// so flip the sign bit instead of copying the bytes as-is
{
    const uint8_t* s = (const uint8_t*)input->ptr;
    uint8_t* d = (uint8_t*)a_in_mem_->virt_addr;
    const size_t n = (size_t)INPUT_W * INPUT_H * 3;
    for (size_t i = 0; i < n; ++i) d[i] = (uint8_t)(s[i] ^ 0x80u);
}
```

`u8 ^ 0x80` 正好等于 `int8(u8 − 128)` 的二进制表示。

### 7.4 坑二：`want_float=0` 返回的是连续 NCHW

我们以为 `rknn_outputs_get(want_float=0)` 会返回 native NC1HWC2。
实际 dump 的文件大小暴露了真相：

| 输出 | native 大小 | 实际 dump |
|---|---|---|
| P3 80×80 | 1,638,400 | 1,638,400 |
| P4 40×40 | 409,600 | **491,520** |
| P5 20×20 | 102,400 | **163,840** |

多出来的是**缓冲区末尾的 padding**，而数据本身是**连续 NCHW**
（每通道 stride = H×W）。

验证：把 dump 按不同解释与 Python 参考特征做相关：

| 解释 | P3 | P4 | P5 |
|---|---|---|---|
| **NCHW 连续** | **0.878** | **0.900** | **0.909** |
| NCHW + 宽度 padding | 0.878 | 0.098 | 0.016 |
| NC1HWC2 | 0.069 | 0.094 | 0.043 |
| NHWC | 0.004 | 0.003 | 0.002 |

（不是 1.0 是因为 C++ 走 RGA letterbox、Python 走 cv2，插值不同。）

所以：**按连续 NCHW 读，缓冲按 `max(size_with_stride, native_size)` 分配。**

### 7.5 坑三：反量化 + 重排（交接）

decoder 要的是 NC1HWC2 FP16（C2=8），backbone 给的是 NCHW INT8。
两边的通道映射：`c = c1×8 + c2`，因此一次遍历即可：

```cpp
static void dequant_nchw(const int8_t* src, __fp16* dst, int H, int W, int C,
                         float scale, float bias) {
    const size_t plane = (size_t)H * (size_t)W;
    for (int c = 0; c < C; ++c) {
        const int8_t* s = src + (size_t)c * plane;
        __fp16* d = dst + (size_t)(c / 8) * plane * 8 + (size_t)(c % 8);
        for (size_t p = 0; p < plane; ++p)
            d[p * 8] = (__fp16)(s[p] * scale + bias);
    }
}
```

`scale/zp` 是 **per-tensor** 的（属性里有单独一个 float/int），
所以 `bias = -zp × scale` 只需算一次。

数据量：2.1 MB 读 + 4.3 MB 写；两边访问模式都是顺序的，实测 **<1 ms**。

### 7.6 `SplitDetector::infer` 主流程

```cpp
bool SplitDetector::infer(const DmaBufferPtr& input, FrameBundle& out) {
    if (!ready_ || !input || !io_mem_ready_) return false;

    // 1) uint8 -> int8 符号位
    {
        const uint8_t* s = (const uint8_t*)input->ptr;
        uint8_t* d = (uint8_t*)a_in_mem_->virt_addr;
        const size_t n = (size_t)INPUT_W * INPUT_H * 3;
        for (size_t i = 0; i < n; ++i) d[i] = (uint8_t)(s[i] ^ 0x80u);
    }
    if (rknn_run(a_ctx_, nullptr) != 0) return false;

    // 2) 取 backbone 的 3 个 int8 输出（写入预分配缓冲）
    rknn_output a_outs[3];
    for (int i = 0; i < a_out_; ++i) {
        a_outs[i].index = i;
        a_outs[i].want_float = 0;
        a_outs[i].is_prealloc = 1;
        a_outs[i].buf = a_out_raw_[i].data();
        a_outs[i].size = (uint32_t)a_out_raw_[i].size();
    }
    if (rknn_outputs_get(a_ctx_, (uint32_t)a_out_, a_outs, nullptr) != 0) return false;

    // 3) 反量化 + 重排，直接写进 decoder 的 io_mem
    for (int i = 0; i < a_out_; ++i) {
        const float scale = a_out_attrs_[i].scale;
        const float bias = -(float)a_out_attrs_[i].zp * scale;
        dequant_nchw((const int8_t*)a_out_raw_[i].data(),
                     (__fp16*)b_in_mem_[i]->virt_addr,
                     feat_h_[i], feat_w_[i], feat_c_[i], scale, bias);
    }

    // 4) decoder
    if (rknn_run(b_ctx_, nullptr) != 0) return false;
    // ... outputs_get 拿 y，拷贝到 FrameBundle
}
```

### 7.7 配对校验

两个模型必须一起编译。用 `custom_string` 写 build id，启动时比对：

```cpp
static std::string query_custom_string(rknn_context ctx) {
    rknn_custom_string cs{};
    if (rknn_query(ctx, RKNN_QUERY_CUSTOM_STRING, &cs, sizeof(cs)) != 0)
        return std::string();
    return std::string(cs.string);
}
...
if (!id_a.empty() && !id_b.empty() && id_a != id_b) {
    fprintf(stderr, "[split] model mismatch: backbone '%s' vs decoder '%s'\n",
            id_a.c_str(), id_b.c_str());
    return false;
}
```

实测：故意用 `v41-WRONG` 编一个 decoder，启动即被拒绝：

```
[split] model mismatch: backbone 'v41-q150l2-20261109' vs decoder 'v41-WRONG'
[npu] worker 0 split init failed
```

### 7.8 调试开关

`RTDETR_SPLIT_DEBUG=1` 会打印每张特征图的 min/max 和 y 的分数范围。
正常（uav.jpg）：

```
[dbg] B in0 fp16 min=-0.2886 max=8.9844
[dbg] B in1 fp16 min=-0.2629 max=10.6484
[dbg] B in2 fp16 min=-0.2800 max=8.6406
[dbg] y scores min=0.0030 max=0.9268
```

如果 `y max` 只有 0.2~0.4，就是交接出了问题（符号位 / 布局 / 反量化）。

---

## 8. 调度与性能工程

### 8.1 核绑定

```cpp
if (cfg_.core_mode == "split") {
    if (cfg_.num_npu <= 1) {
        mask = RKNN_NPU_CORE_0_1_2;          // 单实例用满三核
    } else {
        switch (idx % 3) {                    // 多实例各绑一核
            case 0: mask = RKNN_NPU_CORE_0; break;
            case 1: mask = RKNN_NPU_CORE_1; break;
            default: mask = RKNN_NPU_CORE_2; break;
        }
    }
}
```

拆分的两个 context 用**同一个** mask。

### 8.2 实测对比

uav.jpg，30 帧 warmup，300 帧：

| 配置 | FP16 单模型 | INT8 拆分 |
|---|---|---|
| 1 worker × 3 核 | 8.07 FPS | 9.35 FPS |
| **3 worker × 1 核** | **17.99 FPS** | **29.87 FPS** |
| 4 worker（超订 3 核） | 18.53 FPS | 33.32 FPS |

核间并行（1 实例用 3 核）只有数据并行的约 1/3 效率，因为 RKNN 的
多核推理在模型中间有串行段，扩展性远低于线性。

### 8.3 Python 与 C++ 的差距

同一套拆分链路：

| 实现 | 3 worker 聚合 |
|---|---|
| Python（rknnlite，多进程） | 25.44 FPS |
| **C++（RGA + io_mem + 线程流水线）** | **29.87 FPS** |

差 17%，主要来自 Python 的调用开销、numpy 转置，以及预处理没有流水线化。

### 8.4 时延口径

- `--latency`：单帧串行、无排队，测的是**真实单帧延迟**（拆分约 70 ms）；
- `[stage] e2e`（压测模式）：包含队列排队时间（约 30 帧在飞），数值偏大。

两个数字不能混用。README 里同时给出时，要标清楚。

### 8.5 分段耗时（拆分，单核）

```
pre 4.5 ms | npu 102 ms | post 0.01 ms   （3 worker）
```

`npu` 包含：A 推理 + 反量化交接 + B 推理。交接占比 <1%，不是瓶颈。

---

## 9. 评估方法与结果

### 9.1 为什么之前的指标不够

v4.0.x 一直用“与 PyTorch 逐框一致性（recall/precision/meanIoU，11 张图）”。
问题：

- 对置信度阈值敏感（conf 差 0.05，recall 可能差几个点）；
- 11 张图样本太小；
- 无法与其他模型/论文的 AP 对话。

### 9.2 VisDrone val AP 流程

数据集：`VisDrone2019-DET-val`，548 张、38,759 条标注（COCO 格式 `val.json`）。

```bash
# 1) 标注转成逐图 txt（x,y,w,h,score,category,trunc,occ）
#    score=0 表示忽略区（iscrowd）
# 2) 板端导出检测
python python/evaluation/dump_dets_board.py \
    --model model.rknn --images "visdrone_val/*.jpg" \
    --conf 0.05 --out dets.txt

# 3) 计算 COCO 风格 AP
python python/evaluation/visdrone_map.py \
    --det dets.txt --ann ann_dir --conf 0.05 --ignore-regions
```

两个容易踩的坑：

1. **类号偏移**：检测文件的类号必须是 0–9（评估器内部 `gt[cat-1]`）。
   一开始写成 1–10，AP 直接掉到 0.0026；
2. **坐标反变换**：letterbox 的 `scale/left/top` 必须正确还原到原图，
   否则 IoU 全灭。

### 9.3 同口径结果

| 模型 | AP | AP50 | AP75 |
|---|---|---|---|
| PyTorch `best.pt`（300q/3L） | 0.1996 | 0.3589 | 0.1884 |
| ONNX FP32（导出） | 0.1996 | 0.3589 | 0.1884 |
| RKNN FP16 300q/3L | 0.1985 | 0.3582 | 0.1872 |
| RKNN FP16 150q/3L | 0.1925 | 0.3411 | 0.1856 |
| RKNN FP16 300q/2L | 0.1939 | 0.3528 | 0.1832 |
| RKNN FP16 150q/2L | 0.1882 | 0.3368 | 0.1804 |
| **RKNN INT8 拆分 150q/2L** | **0.1882** | 0.3359 | 0.1800 |

分析：

- **导出无损**：PT → ONNX 完全相同；ONNX → RKNN FP16 只掉 0.0011；
- **INT8 无损**：同结构下 FP16 与 INT8 的 AP 一致；
- **裁剪有损**：query 300→150 掉 0.0060，decoder 3→2 层掉 0.0046，可加和。

### 9.4 v3.0.5 为什么不能直接比较

v3 README 记录：FP32 ONNX AP 0.2180、FP16 0.2096、INT8 0.2049。
我们用同一套脚本复评 **v3 自己的 FP16 模型**（`rtdetr_fp16.rknn`，55.68 MB）：

```
AP = 0.0968 / AP50 = 0.2117 / AP75 = 0.0816
```

同一张 val 图上，v3 模型在 conf 0.5 时只有 14 个检出，我们的模型 45 个。
结论：**v3 的 AP 来自另一套前后处理口径**，两套数字不能放在一张表里比较。
（也可能是本地那个 55.68 MB 文件不是最终发布版，无法进一步确认。）

## 10. 复现清单

### 10.1 环境

| 环节 | 版本 |
|---|---|
| 板端 runtime | librknnrt 2.5.0（`/usr/lib/librknnrt.so`） |
| 模型工具链 | rknn-toolkit2 2.5.0（WSL，Python 3.10） |
| OpenCL | Mali blob + `clImportMemoryARM` shim |
| 导出 | PyTorch + ultralytics（`UAVDETR_SRC` 指向源码目录） |

### 10.2 从 best.pt 到四个模型

```bash
# 1) 导出（300q/3L，4 维网格）
python python/export/export_transpose_opt.py --weights best.pt --value-nchw \
    --fold-offset-scale --msda-4d-grid --output model/rtdetr_tpopt2_4d.onnx

# 2) 导出（150q/2L）
python python/export/export_transpose_opt.py --weights best.pt --value-nchw \
    --fold-offset-scale --msda-4d-grid --num-queries 150 --decoder-layers 2 \
    --output model/rtdetr_q150_l2_4d.onnx

# 3) FP16 单模型
python python/quantize/convert_native_gpu.py model/rtdetr_tpopt2_4d.onnx \
    model/rtdetr_tpopt2_4d_mulnpu.rknn --opt-level 0 \
    --disable-rules squeeze_nd_mul,fuse_mul_into_gemm \
    --op-target-map 'GridSample=gpu,Mul=npu'

# 4) 拆分 + INT8
python python/export/split_decoder.py model/rtdetr_q150_l2_4d.onnx \
    --feat /model.19/cv3/act/Mul_output_0 /model.22/cv3/act/Mul_output_0 \
           /model.25/cv3/act/Mul_output_0 \
    --out-a model/q150_backbone.onnx --out-b model/q150_decoder.onnx

python python/quantize/convert_int8.py model/q150_backbone.onnx \
    model/rtdetr_v41_backbone_int8.rknn \
    --dataset calib.txt --custom-string v41-q150l2

python python/quantize/convert_native_gpu.py model/q150_decoder.onnx \
    model/rtdetr_v41_decoder_fp16.rknn --opt-level 0 \
    --disable-rules squeeze_nd_mul,fuse_mul_into_gemm \
    --op-target-map 'GridSample=gpu,Mul=npu' \
    --inputs /model.19/cv3/act/Mul_output_0,/model.22/cv3/act/Mul_output_0,/model.25/cv3/act/Mul_output_0 \
    --input-sizes 1x256x80x80\;1x256x40x40\;1x256x20x20 --no-normalize \
    --custom-string v41-q150l2
```

`calib.txt`：每行一个校准图片的绝对路径。本次用 VisDrone subset 的 45 张
（20 张时 AP 略低约 1.1 个点，边际收益很小）。

### 10.3 板端运行与压测

```bash
export LD_LIBRARY_PATH=/home/cat/opencl_shim:$LD_LIBRARY_PATH

# FP16
./build/rtdetr_pipeline -m rtdetr_tpopt2_4d_mulnpu.rknn \
    --image-list list.txt -n 3 -p 2 -P 2 -N 300 --cores split

# INT8 拆分
./build/rtdetr_pipeline -m rtdetr_v41_backbone_int8.rknn \
    --decoder rtdetr_v41_decoder_fp16.rknn \
    --image-list list.txt -n 4 -p 2 -P 2 -N 300 --cores split

# 单帧延迟
./build/rtdetr_pipeline ... -i uav.jpg -n 1 -p 1 -P 1 --latency
```

### 10.4 模型指纹

| 文件 | 大小 | md5 |
|---|---|---|
| `rtdetr_tpopt2_4d_mulnpu.rknn` | 46,561,837 | `8961371a7b32952edccb0f1af553aa0f` |
| `rtdetr_v41_backbone_int8.rknn` | 17,528,453 | `a66a4e12b9d560a78517c503d194c3f5` |
| `rtdetr_v41_decoder_fp16.rknn` | 8,127,043 | `5a182d1eb20736987c57f1cdcb58c3da` |

> md5 与 toolkit 版本、校准集、build id 绑定；换环境会变，仅作本机对照。

---

## 11. 踩坑索引

| # | 现象 | 原因 | 解决 |
|---|---|---|---|
| 1 | `Unsupport CPU op: GridSample` | 编译时没指定 GPU | `--op-target-map 'GridSample=gpu'` |
| 2 | `Unsupport GPU op: GridSample` | OpenCL 库不导出 `clImportMemoryARM` | 用 Mali blob 建 shim |
| 3 | 模型能跑但 decoder 全错 | 6 维广播 Mul 被编译成全 0 | `--msda-4d-grid`（4 维） |
| 4 | 只比 encoder 输出时一切正常 | encoder 不受该缺陷影响 | 必须比对 `y` / `out[2]` |
| 5 | 把 Mul 放 tensor 级 CPU 没反应 | op_type 级 `Mul=npu` 覆盖 tensor 级 key | 二者不要同时给 |
| 6 | 放 1 个算子到 CPU 也慢 27 ms | 代价是 CPU 边界本身，不是算子数量 | 尽量别在 decoder 中间切边界 |
| 7 | 纯 INT8 运行崩溃 `map::at` | GridSample GPU 不支持 INT8 | 拆模型 |
| 8 | auto-hybrid 慢 3 倍 | 179 个 FP16 边界 | 不用 auto-hybrid |
| 9 | hybrid step1 报 `Invalid key 'GridSample'` | step1 的 op_target 白名单不含它 | 无解，改拆模型 |
| 10 | step2 报 config inconsistent | step2 禁止再 `rknn.config` | 无解 |
| 11 | `w8a16` 编译失败 | RK3588 不支持 | 只能 w8a8 |
| 12 | 拆分后检出为 0 | 输入符号位 / 输出布局 / 反量化 | 见 7.3–7.5 |
| 13 | 特征相关性只有 0.88 | uint8 当 int8 用（亮度 +0.5） | `u8 ^ 0x80` |
| 14 | 按 NC1HWC2 读输出全是乱的 | `want_float=0` 返回连续 NCHW | 按 `[C][H][W]` 读 |
| 15 | 多实例不提速 | 没绑核 | `--cores split` |
| 16 | `--latency` 与 `[stage] e2e` 差很多 | 后者含排队时间 | 单帧只看 `--latency` |
| 17 | AP 只有 0.0026 | 检测类号写成 1–10 | 改成 0–9 |
| 18 | 启动即退出 `model mismatch` | 两个模型 build id 不同 | 相同 `--custom-string` |
| 19 | 把 GridSample 的中间张量加成输出 → 段错误 | GPU cl_mem 与“同时作为输出”冲突 | 只 dump 非 GridSample 直接输入 |

---

## 12. 经验与结论

### 12.1 可复用的经验

1. **“能跑”不等于“正确”。**
   这个项目里有两个模型都是“能跑、还更快”：v4.0.0 的 Mul=npu 和
   只用 encoder 头对比的早期检查。建立中间张量的 dump 流程，比看最终指标更可靠。

2. **布局和量化语义必须探测，不能猜。**
   `rknn_query` 的标准属性/native 属性和实际行为并不总是一致
   （`want_float=0` 的返回布局就是例子）。先写 30 行的探针，能省几小时。

3. **CPU 边界的代价是“次数 × 固定开销”。**
   在 decoder 内部放 1 个或 40 个 CPU 算子，代价都是 ~27 ms（3 层 × 9 ms）。
   所以“优化”时不要用 CPU 兜底换正确性，代价比想象的大。

4. **多核并行优先选数据并行。**
   1 实例用 3 核只有 9~13 FPS，3 实例各绑 1 核能到 30 FPS。
   模型中间有串行段，核间扩展性远低于线性。

5. **跨版本比较前先固定评估协议。**
   v3 的 0.2049 和我们的 0.1882 不可比（复评 v3 模型只有 0.0968）。
   同一套脚本、同一份标注、同一套后处理，是数字可信的前提。

### 12.2 仍然存在的限制

- **长稳未验证**：目前只有 200–500 帧级压测，没有 30 分钟以上的降频曲线；
- **v3 功能未迁移**：V4L2 摄像头、GStreamer+MPP 硬解、实时显示、
  按类阈值、`--dump-det` 都没有搬到 v4；
- **INT8 只能拆模型**：单模型混合精度在 RKNN 2.5 上不可行；
- **拆分多一次交接**：4.3 MB/帧的特征搬运，虽然 <1 ms，但存在；
- **精度与速度的取舍**：150q/2L 比 300q/3L 低 0.0103 AP。

### 12.3 后续

1. **蒸馏微调**：以 300q/3L 为教师，KL(logits) + L1(box)，
   把 150q/2L 的 0.0103 AP 找回来（不需要改推理代码）；
2. **NEON 优化交接**：当前是标量循环（已 <1 ms），大模型时再考虑；
3. **长稳与温控**：3/4 worker 满载下的降频曲线；
4. **多板扩展**：每板独立跑 3–4 worker，按路数水平扩展；
5. **能力补齐**：按优先级迁移摄像头、显示、按类阈值。

---

## 13. 附录

### A. 关键参数速查

| 参数 | 值 / 建议 |
|---|---|
| 编译 | `--opt-level 0 --disable-rules squeeze_nd_mul,fuse_mul_into_gemm` |
| device | `--op-target-map 'GridSample=gpu,Mul=npu'` |
| 导出 | `--value-nchw --fold-offset-scale --msda-4d-grid` |
| 拆分 | `--num-queries 150 --decoder-layers 2` |
| worker | FP16 用 `-n 3`；拆分用 `-n 4` |
| 绑核 | `--cores split` |
| 校准 | 45 张 VisDrone 子集 |

### B. 复测原始日志摘录

```
# FP16 300q/3L
[latency] e2e mean 122.04 ms, best 118.95 ms (n=25)
[result] workers=3 cores=split frames=300 wall=17.505s
[result] aggregate 17.14 FPS
[result] pre 3.51 ms | npu 174.47 ms | post 0.01 ms
[result] detections total=16500 (avg 55.0/frame)

# INT8 拆分 150q/2L
[latency] e2e mean 69.64 ms, best 68.41 ms (n=25)
[result] workers=3 cores=split frames=300 wall=10.234s
[result] aggregate 29.31 FPS
[result] pre 4.48 ms | npu 102.05 ms | post 0.01 ms
[result] detections total=18300 (avg 61.0/frame)

# 配对校验（故意错配）
[split] model mismatch: backbone 'v41-q150l2-20261109' vs decoder 'v41-WRONG'
[npu] worker 0 split init failed
```

### C. 术语表

| 术语 | 说明 |
|---|---|
| **native GridSample** | RKNN 2.5 内置的 GridSample 算子，只在 GPU（OpenCL）上执行 |
| **NC1HWC2** | RKNN 的内部张量布局：`[N][C1][H][W][C2]`，C2 通常是 8（FP16）或 16（INT8） |
| **io_mem** | `rknn_set_io_mem`，把用户缓冲直接绑定为模型输入/输出，避免每帧拷贝 |
| **op_target** | 指定某类算子或某个输出 tensor 的执行设备（cpu/npu/gpu/auto） |
| **hybrid quantization** | RKNN 的混合量化流程（step1 生成 cfg，step2 编译），用于逐层指定精度 |
| **query** | RT-DETR decoder 的候选框数量；本项目用 300 或 150 |
| **AP / AP50 / AP75** | COCO 风格的平均精度；AP50/75 是 IoU 阈值 0.5/0.75 下的值 |
| **letterbox** | 等比缩放 + 居中填充到 640×640，保证长宽比不变 |
| **RGA** | RK3588 的 2D 硬件加速器，用于缩放/转色 |
| **DRM/PRIME** | Linux 的显存/缓冲共享机制，RGA 与 NPU 之间零拷贝的基础 |

---

（完）


