# 6 维广播 Mul 缺陷与 v4.0.1 修复

基线：`rtdetr_native_tpopt2.onnx`（300 query / 3 层 decoder / native GridSample FP16）
板端：RK3588 + rknn-toolkit2 2.5.0 / librknnrt 2.5.0
阈值口径：模型输出 `y = (300, 4+10)`，分数为 sigmoid 概率，conf = 0.45

---

## 1. 已修复的缺陷：`Mul=npu` 让 decoder 输出全错

### 现象

`op_target` 把 `Mul` 放到 NPU 时，decoder 的框和分数都错；encoder 头输出**逐位相同**。

| 输出 | Mul=cpu（正确） | Mul=npu（错误） |
|---|---|---|
| `out[0]` pred_logits `y` | max +0.9907 | max +0.9922，高置信 query 平均差 0.069 |
| `out[2]` dec_scores（raw） | max **+2.7969** | max **−1.8008** |
| `out[3]` / `out[4]` encoder | — | **逐位相同** |

### 根因（逐 tensor 定位）

在 layer 0 的 cross-attn 上把这些张量加成 ONNX 输出后上板 dump：

| 张量 | ONNX 正确值 | 板上（Mul=npu） | 判定 |
|---|---|---|---|
| `Div_output_0`（offsets / n_points） | −10.19 … +4.62 | −10.18 … +4.63 | ✅ 正常 |
| `Slice_output_0`（ref_wh） | q0 = [0.10501, 0.08963] | q0 = [0.10492, 0.08966] | ✅ 正常 |
| **`Mul_output_0`（offsets × ref_wh）** | −0.54 … +0.30 | **100% 全 0** | ❌ 编译坏 |
| `Add_output_0`（+ ref_xy） | 有偏移 | 每个 query **严格常数**，等于参考框中心 | 后果 |

**结论：不是尺度错、不是精度损失，而是 RKNN 2.5 无法正确执行这个 6 维广播 Mul**
`(1,300,8,3,4,2) × (1,300,1,1,1,2)`，直接返回全 0。
采样点因此全部塌到参考框中心，decoder 只在锚点取特征。

### 已排除的方案

| 尝试 | 结果 |
|---|---|
| 导出期折叠 `0.5/n_points` 进权重 | ONNX 与原图**逐位相同**，仍坏（坏的是剩下的乘法） |
| `Expand(ref_wh)` 成同形状再乘 | 仍坏（优化器把 Expand 折回广播） |
| 常量尺度扫描 k = 0.25 / 0.5 / 1 / 2 / 4 | 无任何 k 能复现坏输出 |
| `ref_wh` 的 w/h 交换 | 不匹配 |
| `Mul=gpu`（OpenCL） | 运行时段错误（RC=139） |
| 禁用 `fuse_mul_into_conv` | 不修复，还把 encoder 数值带偏 |
| 把 grid 中间张量加成模型输出 | GridSample 相关张量会导致段错误 |

### 修复：把网格运算降到 4 维

RKNN 原生是 4 维布局。把采样点 / grid 的计算从 6 维改成 4 维：

```python
# 原来（6-D 广播，被编译成 0）
add = sampling_offsets * refer_bbox[:, :, None, :, None, 2:]

# 现在（4-D，等价）
off4 = sampling_offsets.reshape(bs, q, H * L * P, 2)      # (1,300,96,2)
ref4 = refer_bbox.reshape(bs, q, 1, 4)
loc4 = ref4[..., :2] + off4 * ref4[..., 2:]               # (1,300,96,2)
sampling_locations = loc4.reshape(bs, q, H, L, P, 2)
```

`grid = 2 * locations - 1` 同样在 4 维上算完再 reshape 回去。

- **ONNX 等价性**：与原图输出**逐位相同**（max|d| = 0）
- **板上数值**：`out[2]` max 恢复 +2.7969，高置信 query 与正确版**完全一致**
- **导出开关**：`--msda-4d-grid`（建议与 `--fold-offset-scale` 同用）

### 修复后的性能

| 配置 | 推理 | 单实例 | 3 实例（绑核） | decoder |
|---|---|---|---|---|
| Mul=npu（旧，坏） | 130.2 ms | 7.48 FPS | 18.06 FPS | ❌ |
| Mul=cpu（避开缺陷） | 157.3 ms | 6.20 FPS | 15.41 FPS | ✅ |
| **4-D grid + Mul=npu（修复）** | **131.1 ms** | **7.95 FPS** | **17.83 FPS** | ✅ |

精度（vs PyTorch 300q/3L，11 图）：**recall 98.22% / precision 98.62% / meanIoU 0.9805**。

> 代价从 +27 ms 降到 **+0.9 ms**。
> 注：把 grid 通路上任意一个算子挪到 CPU 都会带来约 +27 ms——成本来自 decoder 内部的
> CPU 边界（3 层 × ~9 ms 的同步与布局转换），与 CPU 上放几个算子无关。

---

## 2. 未改动模型（300q / 3L）其他变体

| 变体 | 推理 | 单实例 | 3 实例 | 4 实例 | decoder | 体积 |
|---|---|---|---|---|---|---|
| FP16 + 4-D grid（推荐） | 131.0 ms | 8.11 | 17.99 | 18.53 | ✅ | 46.6 MB |
| FP16 + Mul=cpu | 157.3 ms | 6.20 | 15.41 | — | ✅ | 46.6 MB |
| auto-hybrid | 410.5 ms | 2.41 | — | — | ✅ | 28.5 MB |
| 纯 INT8 | 崩溃 | — | — | — | — | 22.6 MB |
| 区域混合（decoder 全 FP16） | 不可运行 | — | — | — | GridSample 落 CPU | 30.3 MB |

## 3. 裁剪模型（150q / 2L）

| 变体 | 推理 | 单实例 | 3 实例 | 自参考 recall |
|---|---|---|---|---|
| FP16 + Mul=cpu | 113.3 ms | 9.08 | 20.97 | 98.28% / IoU 0.9825 |
| **FP16 + 4-D grid** | **104.9 ms** | **10.26** | **22.63** | **98.28% / IoU 0.9825** |

裁剪本身的损失（150q 板端 vs PyTorch 300q/3L，11 图）：检出 96.6%、**recall 88.36%**、precision 89.66%、IoU 0.9530。

## 4. 为什么 INT8 / 混合量化在这张图上不 work

1. **纯 INT8 崩溃**：GridSample 的 GPU 实现不接受 INT8，运行时报 `std::out_of_range map::at`。
2. **auto-hybrid 反而慢 3 倍**：179 个 FP16 tensor，INT8/FP16 频繁切换，300q 410 ms。
3. **区域混合被工具链限制**：
   - `hybrid_quantization_step1` 拒绝 `op_target={'GridSample': ...}`
   - step2 不允许再调 `rknn.config`，否则报 config inconsistent
   - `custom_quantize_layers` 只接受可量化 operand 名
   - 最终模型里 GridSample 仍落到 CPU → `Unsupport CPU op`

结论：在 native-GridSample 架构下，INT8 路线在 RKNN 2.5 上不可用。

## 5. 结论

- **v4.0.1 默认配置**：FP16 + `--fold-offset-scale --msda-4d-grid` + `GridSample=gpu, Mul=npu`

  | 模型 | 推理 | 单实例 | 3 实例 | 4 实例 | recall |
  |---|---|---|---|---|---|
  | 300q / 3L | 131.0 ms | 8.11 FPS | 17.99 FPS | 18.53 FPS | 98.22% |
  | 150q / 2L | 104.9 ms | 10.26 FPS | 22.63 FPS | — | 88.36%（对 300q） |

- v4.0.0 的“18 FPS”是在 decoder 损坏的情况下测得的，**不可引用**；现在 17.99 FPS 是同一速度且正确。
- 150q/2L 更快，代价是 recall 从 98.2% 降到 88.4%，需要业务侧确认。
- 300q/3L 已测完，作为 v4.0.1 的发布配置。
