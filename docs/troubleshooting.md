# 常见问题

## 1. 编译失败：`Unsupport CPU op: GridSample`

没有指定 GPU 目标。加 `--op-target-map 'GridSample=gpu'`。

## 2. 运行失败：`Unsupport GPU op: GridSample`

两种可能：

1. `libOpenCL.so` 不导出 `clImportMemoryARM`（最常见）
   → 用 Mali blob 建 shim，见 `docs/model.md` §5
2. 链接到了旧版 `librknnrt.so`（例如某处自带的 2.3.2）
   → `ldd ./rtdetr_pipeline | grep rknn` 确认是 `/usr/lib/librknnrt.so`（2.5.0）

## 3. 编译崩溃：`operands could not be broadcast together with shapes (256,256) (8400,1)`

`optimization_level=3` 触发 `fuse_mul_into_gemm` 的确定性 bug。
同时禁用 `squeeze_nd_mul` 和 `fuse_mul_into_gemm`，或直接用 `--opt-level 0`。

## 4. `MAP_DUMB failed: Permission denied`

部分板子（如本项目的板卡）`renderD129` 支持 dumb buffer 但**不允许 mmap**，
而 `card1` 可以。v4 会在启动时探测 `CREATE + MAP`，自动选择可用节点。

## 5. 检出数量比 Python 多很多（例如 38 vs 24）

预处理用了直接拉伸而不是 letterbox。v4 默认 letterbox（等比缩放 + 零填充），
与 Python 参考一致。

## 6. 时延测出来只有 22 ms / 好几秒

- 22 ms：`wait_idle` 在帧"飞行中"（队列都为空但尚未处理完）时提前返回。
  v4 用 in-flight 计数，不会误判。
- 好几秒：压测时 `[stage] e2e` 包含排队时间（约 33 帧在飞），
  不是单帧延迟。看 `--latency` 或 `[stage] npu`。

## 7. 多实例没有变快

多个实例共享三核时几乎不扩展（10.2 FPS）。必须 `--cores split`
（每个 worker 绑一个核），才能到 18 FPS。

同时注意：视频场景 3 实例最优，4 实例反而变慢（解码与推理抢 CPU）。

## 8. 开 `RKNN_FLAG_COLLECT_PERF_MASK` 直接段错误

RKNN 2.5 runtime 在 v3 之后的这个图上采集性能会 SIGSEGV（崩在 `rknn_run`）。
不影响正常运行，只影响逐层 profile。v4 的耗时数据来自分段打点和墙钟基准。

## 9. `create_mem_from_fd` 报错 / 推理卡死数秒

模型的**原生输入是 FP16**（`rknn_query(NATIVE_INPUT_ATTR)`），RGA 的 8bit 输出
不能直接给它。v4 用 `rknn_create_mem()` + 自行做 uint8→FP16 转换。
详见 `docs/architecture.md` §3.2。

## 10. v4.1 拆分：`model mismatch: backbone '...' vs decoder '...'`

两个模型不是同一批编译的。用相同的 `--custom-string` 重新编译 backbone 和 decoder
（见 `docs/model.md` §3b）。这是刻意设计：配错的 backbone/decoder 会**能跑但输出垃圾**，
所以启动阶段直接拒绝。

## 11. v4.1 拆分：能跑但检出为 0 / 分数很低

三个已知原因（都已在 `split_detector.cc` 处理，改代码时不要退化）：

1. **backbone 输入符号位**：输入张量是 INT8（zp=−128），RGA 给的是 uint8。
   直接 `memcpy` 等于把亮度整体加 0.5 → 分数塌到 0.1~0.3。必须逐字节 `^ 0x80`。
2. **backbone 输出布局**：`rknn_outputs_get(want_float=0)` 返回的是**连续 NCHW**
   （不是 native NC1HWC2），且缓冲区按 `size_with_stride` 分配（尾部有 padding）。
3. **decoder 输入布局**：native NC1HWC2 FP16（`C2=8`），需要把 backbone 的
   INT8 NCHW 反量化（per-tensor `scale`/`zp`）并重排后写入。

自检方法：`RTDETR_SPLIT_DEBUG=1` 会打印每张特征图的 min/max 和 y 的分数范围，
正常时 uav.jpg 的 y max ≈ 0.93、conf≥0.45 约 61 个检出。
