# 交接：RKNN 2.5.0 6D 广播 Mul 全 0 问题

更新时间：2026-11-16
状态：**根因已定位到唯一变量 `disable_rules=['squeeze_nd_mul']`，
最小复现（2 节点）已上板验证，issue 正文已重写，待提交。**

---

## 1. 结论先说

同一份 2 节点 ONNX、同一台板子、同一天：

| `optimization_level` | `disable_rules` | 结果 |
|---|---|---|
| 0 | 无 | ✅ 正确 |
| 3 | 无 | ✅ 正确 |
| 0 | `['squeeze_nd_mul']` | ❌ **全 0** |
| 3 | `['squeeze_nd_mul']` | ❌ **全 0** |

也就是说：

- 触发条件是**禁用 `squeeze_nd_mul`**，与 `optimization_level` 无关；
- `fuse_mul_into_gemm` 与本次缺陷无关（只禁用它仍然正确）；
- 启用 `squeeze_nd_mul` 时，6D 广播 Mul 会被正确降级；
- 禁用之后，编译器仍然接受 6D Mul，并生成一个**输出全 0 的 kernel**，
  构建、加载、推理全部返回 0，没有任何警告。

这个结论直接解释了完整模型为什么坏：仓库里的
`python/quantize/convert_native_gpu.py` 的 `--disable-rules` 默认值就是
`squeeze_nd_mul`，而 v4.0 的失败模型正是用它编译的。

---

## 2. 最小复现

`issues/rknn-6d-mul/minimal/`

```text
off_in (1,300,8,3,4,2) ---------------------------.
                                                   Mul --> out (1,300,8,3,4,2)
ref_in (1,300,2) --Reshape--> (1,300,1,1,1,2) ----'
```

- `m2.onnx`：260 字节，2 个节点（`Reshape` + `Mul`）；
- `make.py`：重新生成 `m2.onnx` 与 ORT 参考 `ref/m2_ref.npz`；
- `build.py`：一次生成上表 4 个 `.rknn`（WSL / toolkit 2.5.0）；
- `run_board.py`：在板端跑 4 个模型并和 ORT 对比；
- `build/*.rknn`：已预编译，板端不需要 toolkit。

板端实测（2026-11-16）：

```text
m2_opt0_default    nz=57600/57600 max=11.367188 max|d|=0.005617 ok
m2_opt3_default    nz=57600/57600 max=11.367188 max|d|=0.005617 ok
m2_opt0_sqoff      nz=0/57600     max=0.000000  max|d|=12.426314 <-- ALL ZERO
m2_opt3_sqoff      nz=0/57600     max=0.000000  max|d|=12.426314 <-- ALL ZERO
```

`max|d|=0.0056` 是 FP16 正常误差；全 0 那两档 `max|d|` 完全相同（12.426314），
两个 `.rknn` 也都是 7451 字节，正确版都是 18267 字节。

---

## 3. 完整模型对应关系

失败模型：`rtdetr_native_tpopt2.onnx`（300q / 3L / native GridSample），
构建参数 opt0 + `op_target={'GridSample':'gpu','Mul':'npu'}` +
`disable_rules=['squeeze_nd_mul','fuse_mul_into_gemm']`。

| 张量 | `Mul=cpu` | `Mul=npu` |
|---|---|---|
| `Div_output_0` | -10.18 … +4.63 | 逐位相同 |
| `Slice_output_0`（ref_wh） | 0.0055 … 0.225 | 逐位相同 |
| `Mul_output_0` | 57600/57600 非零 | **0/57600** |
| `Mul_1_output_0`（0.5×） | 57600/57600 非零 | 0/57600（被上游带成 0） |
| `out[2]` max | **+2.7969** | **−1.8008** |

encoder 输出逐位相同，只有 decoder 坏。这与最小复现是同一个 `Mul`。

---

## 4. 已被否定的假设（避免重复实验）

| 假设 | 实验 | 结果 |
|---|---|---|
| 裸 6D Mul 不支持 | `mul_probe.onnx` 的原始 opt0 构建（未禁任何规则）| ✅ 正确，与“规则开启即正确”一致 |
| `optimization_level` 是主因 | m2 在 opt0 / opt3 各编一次 | 与 level 无关，两档都坏 |
| `Reshape` vs `Unsqueeze` 扩维是主因 | m3（多轴 Unsqueeze）、m4（三层单轴） | 两档 opt 都全 0，不是主因 |
| `Expand` 成同形状可绕过 | probe 的 `mul_expand` 分支 | opt3 下仍全 0 |
| 需要特定上游算子（Div / Sigmoid / Transpose / MatMul） | v0–v3 变体 | v0（只有 Reshape）就复现 |
| 三条 swap/elim 规则是触发点 | 分别禁用 `swap_reshape_mul` / `swap_transpose_mul` / `eliminate_mul_redundancy_const_dim` | 完整模型仍坏，不是 |
| `Mul` 落 CPU 是唯一出路 | `v3_both_cpu` / `Mul=cpu` | ✅ 正确，但慢 ~27 ms |

---

## 5. 同一规则的第二条失败路径（已复现，已写入 issue）

完整模型（`rtdetr_native_tpopt2.onnx`，opt0，`Mul=npu`，
`disable_rules=['fuse_mul_into_gemm']`，即**启用** `squeeze_nd_mul`）会直接在
规则内部崩：

```text
File "rknn/api/graph_optimizer.py", line 2221, in GraphOptimizer.fuse_ops
File "rknn/api/rules/dims.py", line 1699, in _p_squeeze_nd_elementwise_op
AttributeError: 'IRGraph' object has no attribute 'node_use_target_const'
```

所以完整模型两条路都走不通：

| `squeeze_nd_mul` | 结果 |
|---|---|
| 启用 | build 崩溃（`AttributeError`，`dims.py:1699`）|
| 禁用 | build 成功，`Mul` 输出全 0 |

2 节点最小图 **不会**触发这个崩溃（它能正常编译、正确推理），说明崩溃需要
真实图里更复杂的上下文。完整日志已复制到
`issues/rknn-6d-mul/logs/full_model_squeeze_enabled_crash.log`。

另外，`optimization_level=3` 下同一个完整模型会更早崩在 `fuse_mul_into_gemm`
（`operands could not be broadcast together with shapes (256,256) (8400,1)`），
这是第三个独立缺陷；它不是全 0 的必要条件（opt0 只禁用 `squeeze_nd_mul`
就全 0）。

---

## 6. 现状与待办

已完成：

- `issues/rknn-6d-mul/issue.md` 已按新结论重写（英文，含最小图、4 档 A/B、
  对照表、完整模型数字、两条失败路径与 traceback、诉求）；
- `minimal/` 自包含复现包 + 4 个预编译模型；
- `ctx/` 保留 v0–v3 变体与 ORT 参考，作为“算子本身没问题”的佐证。

待办：

1. 决定 issue 的发布位置（GitHub issue 正文 + 附件压缩包）；
2. 清理 `ctx/mini/` 下的中间产物（`m1*`、`m3*`、`m4*`、`m2_l0*`），
   只保留 `m2` 相关；`ctx/build/` 里被 7 个变体占用的空间也确认一下；
3. 若要提交，`issues/` 与 `docs/` 尚未 commit。

---

## 7. 板子与工具链

- 板子：`192.168.10.111`，用户 `cat`。**密码由用户提供，不要写进任何文件。**
- 板端 Python：`/home/cat/software/miniconda3/envs/rknn38/bin/python`
  （Python 3.8 + rknn-toolkit-lite2 2.5.0）。
- 板端复现目录：`/home/cat/rknn_minimal/`（对应本仓库 `minimal/`）。
- 板端变体目录：`/home/cat/rknn_ctx/`。
- WSL 构建环境：`/home/zepher/miniconda3/envs/rk25`（toolkit 2.5.0）。
- 完整模型 ONNX：`D:\learn\rknn2.5\rt-detr-rknn\model\rtdetr_native_tpopt2.onnx`。
- 失败/正确模型：`/home/cat/rknn2.5/model/rtdetr_tpopt2_dbg2_mulnpu.rknn`、
  `rtdetr_tpopt2_dbg_mulcpu.rknn`。
