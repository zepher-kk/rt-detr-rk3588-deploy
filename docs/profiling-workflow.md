# RKNN 测量与 Profile 流程（RK3588 / rknn-toolkit2 2.5.0）

> 配套 `porting-and-optimization.md` §2 与 `learning-questions.md` Q2。
> 目标：每一步都能落到具体命令/API，并且知道拿到数据后怎么判断。

---

## 0. 三条原则

1. **一次只改一个变量**；改完必须同时验证数值正确性（快而错没有意义）。
2. **区分“算子成本”和“干预代价”**：A/B 差值回答的是后者。
3. **同一张表只放同一次会话、同一协议的数据**；跨会话数字单独标注。

---

## 1. 定口径

先写清楚这四个问题，再开始测：

| 问题 | 选项 | 本项目建议 |
|---|---|---|
| 延迟还是吞吐 | `--latency` 串行 / `-N` 压测 | 两个都报，分开列表 |
| 统计量 | mean / p50 / p95 / p99 | 主指标 p50，附 p95 |
| 是否含排队 | e2e 含队列 / 纯推理 | 单帧只看 `--latency`；吞吐看聚合 FPS |
| 计时边界 | 从哪到哪 | 读帧完成 → 后处理结束；不含读文件 |

实现要点：

- C++ 用 `clock_gettime(CLOCK_MONOTONIC_RAW)`，避免 NTP 调整影响；
- Python 用 `time.perf_counter_ns()`；
- 每帧记录 `worker_id / t_start / t_end`，后续可画时间线；
- 每次实验输出一个 sidecar JSON：

```json
{
  "model_md5": "...", "build_id": "v41-q150l2",
  "cmdline": "...", "governor": "performance",
  "npu_freq": 1000000000, "temp_c": 52.3,
  "warmup": 30, "n": 300,
  "stages_ms": {"pre": 4.5, "npu": 102.0, "post": 0.01},
  "p50_ms": 69.8, "p95_ms": 72.1, "fps": 29.87
}
```

样本量：单配置 ≥100 次；对比实验 ≥3 轮，每轮 ≥50 次。先跑一次 A/A
（同配置测两遍）得到噪声底，后面的差异必须大于它。

---

## 2. 固定环境

```bash
# CPU governor
for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  echo performance | sudo tee "$c" >/dev/null
done
# 或 cpupower frequency-set -g performance

# NPU / GPU / DMC 频率策略（路径随机型/内核不同，先 ls 确认）
ls /sys/class/devfreq/
for d in /sys/class/devfreq/*; do
  echo performance | sudo tee "$d/governor" >/dev/null 2>&1
done
cat /sys/class/devfreq/*/cur_freq

# 温度与 NPU 负载
for z in /sys/class/thermal/thermal_zone*/temp; do echo "$z $(cat $z)"; done
cat /sys/kernel/debug/rknpu/load        # 需要 root + debugfs 已挂载

# 独占检查
ps -eo pid,comm,args | grep -Ei 'rknn|infer|python|ffmpeg|gst'
```

其他固定项：

- 同一张图、同一条预处理路径（RGA 开/关不能混）；
- warmup 20–30 帧后丢弃；
- 长测前先跑 5 分钟热机，记录温度曲线，确认没有降频；
- 用 `taskset -c 4-7` 把 pre/post 线程绑到 A76（先用 `lscpu -e` 确认编号）；
- 记录模型 md5 + `custom_string`，确保对比的是同一个构建。

> `RKNN_FLAG_COLLECT_PERF_MASK` 本身会降低帧率（头文件注释写明），
> **绝不能在吞吐/延迟基准里开**，只在单独的 profile 运行里开。

---

## 3. 系统级

目标是回答“时间花在哪一级”：排队、CPU、驱动、NPU、GPU。

### 3.1 自有打点

```
wait_raw | pre | wait_npu | npu | wait_post | post
e2e = 从入队到 post 完成
```

同时记录队列水位（push/pop 时采样 `size()`）和阻塞次数。判读：

| 现象 | 结论 |
|---|---|
| `wait_npu` 高、`npu` 低 | NPU worker 饿死：pre 太慢或队列太小 |
| `npu` 高、NPU load 低 | 驱动/同步/内存带宽瓶颈，不是算力 |
| `pre` 高 | RGA ioctl 或 CPU 转换 |
| 4 worker > 3 worker | 流水线有气泡，瓶颈不在单帧计算 |

### 3.2 外部工具

```bash
# CPU 侧总览
perf stat -p <pid> -e task-clock,cycles,instructions,branches,branch-misses,\
cache-references,cache-misses,context-switches,page-faults -- sleep 10

# CPU 热点（dequant / XOR / OpenCV / memcpy）
perf record -F 999 -g -p <pid> -- sleep 10
perf report --stdio | head -50

# rknn_run 的 ioctl 时间（短窗口，strace 有开销）
strace -f -T -e trace=ioctl -p <pid> 2>&1 | head -40
# 或更低开销：
perf trace -e ioctl -p <pid> -- sleep 2

# 调度延迟 / 唤醒延迟
perf sched record -- sleep 10 && perf sched latency

# ftrace 时间线（需 root）
trace-cmd record -e sched_switch -e sched_wakeup -e irq
trace-cmd report | head -100
```

看什么：

- `rknn_run` 的 ioctl 时间 ≈ 驱动 + 硬件等待；和用户态 `npu` 计时对比，
  差值大说明有用户态拷贝/格式转换；
- `perf record` 里 `dequant_nchw`、`^0x80` 循环、`cv2`、`memcpy` 的占比；
- `perf sched latency` 的 max 值，判断三级流水线是否有毫秒级唤醒延迟；
- 运行中每 100 ms 采一次 `cur_freq`，确认没有中途降频。

---

## 4. 模型级

### 4.1 官方逐层（首选）

```c
rknn_context ctx;
rknn_init(&ctx, model, size, RKNN_FLAG_COLLECT_PERF_MASK, NULL);
/* ... rknn_inputs_set / rknn_run ... */
rknn_outputs_get(ctx, n_out, outs, NULL);   // 必须先 get，query 才有效

rknn_perf_detail detail = {0};
rknn_query(ctx, RKNN_QUERY_PERF_DETAIL, &detail, sizeof(detail));
printf("%.*s\n", (int)detail.data_len, detail.perf_data);

rknn_perf_run run = {0};
rknn_query(ctx, RKNN_QUERY_PERF_RUN, &run, sizeof(run));
printf("run_duration = %lld us\n", (long long)run.run_duration);
```

- `PERF_DETAIL` 返回逐层耗时字符串，`perf_data` 不需要 free；
- `PERF_RUN` 单位是 **微秒**；
- 本项目的 native GridSample 图会 SIGSEGV，属工具链缺陷，走 4.2。

### 4.2 崩溃时的替代：设计实验，不做切图求和

用两个因子做线性外推：

| 因子 | 取值 | 得到 |
|---|---|---|
| decoder 层数 L | 1 / 2 / 3 | `T(L) = aL + b`，`a` 是每层成本，`b` 是固定开销 |
| query 数 Q | 75 / 150 / 300 | `T(Q) = cQ + d`，`c` 是每 query 成本 |

每个构建用同一协议测 p50。两层拟合就能把“每层约 9 ms”从推断变成实测。
backbone（没有 GridSample）可以直接开 `COLLECT_PERF_MASK` 拿逐层数据。

### 4.3 辅助手段

```python
# 模拟器：只看相对排序，不看绝对值
rknn.eval_perf(is_print=True)
rknn.eval_memory()
```

```python
# ONNX Runtime 算子级参考（PC）
opts = ort.SessionOptions(); opts.enable_profiling = True
sess = ort.InferenceSession("model.onnx", opts)
# ... run ...
prof = sess.end_profiling()   # JSON：每个节点的 dur
```

再配合：

- 解析 RKNN 编译日志的算子分配表（NPU/GPU/CPU、dtype、shape），
  统计 CPU 算子与设备切换次数；
- Netron 看节点数、算子类型分布。

---

## 5. 干预级

### 5.1 设计

- 一次只改一个变量（op_target、opt-level、导出开关、线程数）；
- A/B/A/B 交替，同一进程内轮转，避免热漂移偏向某一方；
- 每轮记录：p50、p95、均值、温度、频率、输出校验值；
- 至少 3 轮，报每轮的中位数与全轮范围，不只报一个数。

`python/benchmark/bench_native.py` 已经做了轮转、warmup、p50/p95、温度记录，
在此基础上补三样：manifest（md5/build id/参数）、频率采样、输出校验。

### 5.2 判读

- 先跑 A/A，得到噪声底 `noise`；
- 收益 > `max(3%, 3×noise)` 才认为有效；
- 对配对数据（同一轮内 A、B 相邻）算逐轮差值的中位数，比直接比均值稳；
- 如果 A 快但数值错了，直接判负（参考 6 维 Mul 的教训）。

---

## 6. 验证级

三件事必须同时过：

1. **数值**：ONNX Runtime / PyTorch 参考 vs 板端逐输出对比，
   `max|d|`、余弦相似度；中间张量按需 dump（注意 GridSample 直连张量不能当输出）。
2. **精度**：`dump_dets_board.py` + `visdrone_map.py`，同一份标注、同一 conf、
   类号 0–9。
3. **趋势**：ORT 的算子排序 vs RKNN 的分段/逐层数据，方向应一致；
   若 top-3 完全不同，先怀疑测量而不是模型。

建议做成回归脚本：固定图集 → 出 AP + 延迟 JSON → 与基线比，
超过阈值（如 AP -0.002 / 延迟 +5%）即失败。

---

## 7. 工具矩阵

| 步骤 | 工具 | 产出 | 判读 |
|---|---|---|---|
| 1 口径 | `--latency` / `-N`，`CLOCK_MONOTONIC_RAW` | p50/p95、FPS、sidecar JSON | 口径是否混用 |
| 2 环境 | sysfs、`cpupower`、`taskset`、thermal | 频率/温度/独占 | 有无降频、抢占 |
| 3 系统 | 分段打点、`perf`、`strace -T`、`ftrace` | 阶段耗时、热点、ioctl、调度 | 排队/CPU/驱动/NPU 哪一级 |
| 4 模型 | `RKNN_QUERY_PERF_DETAIL`、多层/多 query 构建、ORT profiling | 逐层或斜率、算子排序 | 计算/内存/边界 |
| 5 干预 | `bench_native.py` 轮转、A/A 噪声底 | 配对差值 | 是否超过噪声 |
| 6 验证 | `dump_ref.py` / `compare_eval.py`、`visdrone_map.py` | max\|d\|、AP | 快而错则判负 |

---

## 8. 本项目建议补的代码

| 位置 | 改动 | 作用 |
|---|---|---|
| `cpp/src/pipeline.cc` | `--profile-json`：阶段耗时、队列水位、worker 时间线 | 系统级 |
| `cpp/tests/` | `rknn_perf.cc`：`COLLECT_PERF_MASK` + `PERF_DETAIL` | 模型级 |
| `python/benchmark/` | `bench_matrix.py`：{300q/3L, 150q/2L} × {FP16, INT8} 轮转 | 因子设计 |
| `python/evaluation/` | `regression.py`：AP + 延迟门禁 | 验证级 |

---

## 9. 常见坑

1. 在基准里开了 `COLLECT_PERF_MASK`，帧率被拉低还以为是模型慢；
2. 用 `strace` 跑完整压测，开销污染结果——只开 1–2 秒窗口；
3. `PERF_RUN` 当毫秒用（实际是微秒）；
4. 不同会话的 FPS 放同一张表（1-worker 拆分 9.35 vs 13.06 就是这么来的）；
5. 只比均值，不看 p95 和离散度；
6. 换了 governor/温度没记录，结论无法复现；
7. 只验证速度不验证输出，重演 6 维 Mul 的“快而错”。
