# [RK3588] rknn-toolkit2 2.5.0: 6-D broadcast `Mul` compiles to all zeros when `squeeze_nd_mul` is disabled

## Summary

With rknn-toolkit2 / librknnrt 2.5.0 on RK3588, a 6-D broadcast `Mul`

```text
(1,300,8,3,4,2) * (1,300,1,1,1,2) -> (1,300,8,3,4,2)
```

is compiled into a kernel that returns **all zeros** when the optimizer rule
`squeeze_nd_mul` is disabled:

```python
rknn.config(
    target_platform='rk3588',
    optimization_level=0,            # 3 behaves the same
    op_target={'Mul': 'npu'},
    disable_rules=['squeeze_nd_mul'],
)
```

With the rule enabled, the same ONNX produces correct values. The failure is
silent: `build()` and `export_rknn()` return 0, `init_runtime()` returns 0,
and inference returns a correctly shaped tensor. There is no warning.

For us this is not an academic edge case: **disabling the rule is the only
way our real model builds at all.** On the full RT-DETR decoder ONNX,
enabling `squeeze_nd_mul` crashes the toolkit inside the same rule
(`AttributeError: 'IRGraph' object has no attribute 'node_use_target_const'`,
`rknn/api/rules/dims.py:1699`, `_p_squeeze_nd_elementwise_op`). Disabling it
builds successfully, runs, is *faster*, and silently returns wrong
detections. We found no configuration with `Mul=npu` that compiles this graph
correctly without rewriting the 6-D multiply (see "Two failures of the same
rule").

## Environment

| Item | Value |
|---|---|
| Board | RK3588 (LubanCat), Linux 6.1.84, aarch64 |
| Runtime | librknnrt 2.5.0 (`/usr/lib/librknnrt.so`) |
| Toolkit | rknn-toolkit2 2.5.0 (PC / WSL2, Python 3.10) |
| Model | FP16, `do_quantization=False` |
| Backend | NPU core 0 / all cores (same result) |
| Control | `op_target={'Mul': 'cpu'}` → correct |

## Minimal reproducer

Two nodes, two inputs, one output. `m2.onnx` is 260 bytes:

```text
off_in (1,300,8,3,4,2) -------------------------------------.
                                                             Mul --> out (1,300,8,3,4,2)
ref_in (1,300,2) --Reshape--> ref6 (1,300,1,1,1,2) ---------'
```

Attached in `minimal/`:

| File | Purpose |
|---|---|
| `m2.onnx` | the failing graph (2 nodes: `Reshape`, `Mul`) |
| `make.py` | regenerates `m2.onnx` + ORT reference `ref/m2_ref.npz` |
| `build.py` | builds the four configurations below (toolkit 2.5.0) |
| `run_board.py` | runs them on the board and compares with ORT |
| `build/*.rknn` | prebuilt models, so the board step needs no toolkit |

```bash
# PC / WSL, rknn-toolkit2 2.5.0
python make.py
python build.py

# copy minimal/ to the board, then (rknn-toolkit-lite2 2.5.0)
python run_board.py
```

`build.py` changes exactly one thing between the two groups:
`disable_rules`.

```python
CONFIGS = (
    ('m2_opt0_default', 0, None),
    ('m2_opt3_default', 3, None),
    ('m2_opt0_sqoff',   0, ['squeeze_nd_mul']),
    ('m2_opt3_sqoff',   3, ['squeeze_nd_mul']),
)
```

## Observed result (board, 2026-11-16)

ORT reference: `min=-3.99 max=4.04`, `57600/57600` non-zero.

| Build | `optimization_level` | `disable_rules` | non-zero | max\|NPU−ORT\| | verdict |
|---|---|---|---|---|---|
| `m2_opt0_default` | 0 | — | 57600/57600 | 0.0056 (FP16) | ✅ correct |
| `m2_opt3_default` | 3 | — | 57600/57600 | 0.0056 (FP16) | ✅ correct |
| `m2_opt0_sqoff` | 0 | `squeeze_nd_mul` | **0/57600** | 12.43 | ❌ **all zeros** |
| `m2_opt3_sqoff` | 3 | `squeeze_nd_mul` | **0/57600** | 12.43 | ❌ **all zeros** |

So:

- the trigger is `squeeze_nd_mul` being disabled, not `optimization_level`;
- the zero output is bit-identical at opt0 and opt3 (both failing files are
  7451 bytes);
- with the rule enabled the same ONNX is correct (both default files are
  18267 bytes).

## Controls

All measured on the same board, same day, same toolkit/runtime:

| Control | Config | Result |
|---|---|---|
| 4-D version: `(1,300,96,2) * (1,300,1,2)` | opt3, both rules off, `Mul=npu` | ✅ correct (57600/57600) |
| `Mul` on CPU: `op_target={'Mul':'cpu'}` | opt0, both rules off | ✅ correct |
| `Expand` the small operand to full shape, then `Mul` | opt3, both rules off | ❌ still all zeros |
| `fuse_mul_into_gemm` disabled only | opt0/opt3, `squeeze_nd_mul` on | ✅ correct |
| Larger variants (see below) | opt0 and opt3 | ❌ all zeros |

The 4-D control is important: it shows the NPU can execute the same
elementwise multiply when the broadcast rank is 4, and that disabling the
rules does not itself corrupt results.

## The operands are fine

We built variants that expose the inputs of the failing `Mul` as graph
outputs (one output per branch: `off`, `mul`, `mul1 = 0.5*mul`, `add`).
With `disable_rules=['squeeze_nd_mul','fuse_mul_into_gemm']`, opt3:

| Variant | `off` (operand) | `mul` | `add` |
|---|---|---|---|
| `v0_direct` (Reshape → Mul) | max\|d\| = 9.8e-4 ✅ | **0/57600** ❌ | wrong |
| `v1_div` (Reshape → Div(8) → Mul) | 2.3e-4 ✅ | **0/57600** ❌ | wrong |
| `v2_sigmoid` (Sigmoid → Reshape → Mul) | 1.9e-3 ✅ | **0/57600** ❌ | wrong |
| `v3_both` (Div + Sigmoid) | 2.1e-4 ✅ | **0/57600** ❌ | wrong |
| `v3_both_cpu` (`Mul=cpu`) | 2.1e-4 ✅ | 2.7e-4 ✅ | ✅ |

The operand tensors match ONNX Runtime to FP16 precision; only the `Mul`
output is zero. The second `Mul` (`mul1 = 0.5 * mul`) is also zero because its
input is zero, so there is exactly one broken kernel.

`v1_div_l0` and `v3_both_l0` (same graphs at `optimization_level=0`) fail
identically, confirming the level is irrelevant.

## Impact on a real model

RT-DETR multi-scale deformable attention computes sampling locations as

```python
sampling_locations = refer_bbox[..., :2] + sampling_offsets * refer_bbox[..., 2:]
# (1,300,8,3,4,2) * (1,300,1,1,1,2)
```

In the full 300-query / 3-layer decoder ONNX (`rtdetr_native_tpopt2.onnx`,
FP16, `GridSample=gpu`, `Mul=npu`, opt0, both rules disabled):

| Tensor | correct (`Mul=cpu`) | broken (`Mul=npu`) |
|---|---|---|
| `Div_output_0` (offsets / n_points) | -10.18 … +4.63 | identical (`max|d| = 0`) |
| `Slice_output_0` (ref_wh) | 0.0055 … 0.225 | identical |
| **`Mul_output_0`** | -0.54 … +0.30, 57600/57600 non-zero | **0/57600, all zeros** |
| `Mul_1_output_0` (0.5 × Mul) | 57600/57600 non-zero | 0/57600 |
| decoder score output `out[2]` max | **+2.7969** | **-1.8008** |

Encoder outputs are bitwise identical; only the decoder is wrong. Every
sampling point collapses to the reference-box center, so detection quality
drops silently instead of failing.

## Two failures of the same rule

On the full decoder ONNX (`rtdetr_native_tpopt2.onnx`, 300q / 3L), the rule
cannot be used either way:

**Enabled** — `build()` crashes (toolkit 2.5.0, opt0, `Mul=npu`,
`disable_rules=['fuse_mul_into_gemm']`):

```text
Traceback (most recent call last):
  File "rknn/api/rknn_log.py", line 344, in ...
  File "rknn/api/rknn_base.py", line 2083, in rknn.api.rknn_base.RKNNBase.build
  File "rknn/api/graph_optimizer.py", line 2221, in rknn.api.graph_optimizer.GraphOptimizer.fuse_ops
  File "rknn/api/rules/dims.py", line 1699, in rknn.api.rules.dims._p_squeeze_nd_elementwise_op
AttributeError: 'IRGraph' object has no attribute 'node_use_target_const'
```

Full build log: `logs/full_model_squeeze_enabled_crash.log`.

**Disabled** — build succeeds, and the 6-D `Mul` returns all zeros (the
minimal reproducer above).

The crash does **not** reproduce on the 2-node minimal graph (it builds and
runs correctly there), so it needs the surrounding real graph. Two separate
defects in the same pass:

1. `_p_squeeze_nd_elementwise_op` uses `IRGraph.node_use_target_const`, which
   does not exist in 2.5.0 → `AttributeError`;
2. when the pass is skipped, the 6-D broadcast `Mul` is still accepted by the
   backend and compiled to a kernel that writes zeros.

Note also that at `optimization_level=3` this model crashes earlier in
`fuse_mul_into_gemm`:

```text
operands could not be broadcast together with shapes (256,256) (8400,1)
```

That is a third defect, but it is not required for the zero-output bug —
opt0 shows it with only `squeeze_nd_mul` disabled.

## Suspected cause

`squeeze_nd_mul` looks like the only optimizer pass that lowers an
ND (>4) broadcast `Mul` into a form the NPU backend can emit correctly. When
it is disabled, the backend still accepts the 6-D `Mul` and emits a kernel
that writes zeros. If 6-D broadcast `Mul` is genuinely unsupported on NPU,
the build should fail, or the op should fall back to CPU — not produce a
valid but wrong model.

## Workaround

Rewrite the multiply in 4-D, which is numerically identical and works:

```python
off4 = sampling_offsets.reshape(bs, q, h * l * p, 2)   # (1,300,96,2)
ref4 = refer_bbox.reshape(bs, q, 1, 4)                 # (1,300,1,4)
loc4 = ref4[..., :2] + off4 * ref4[..., 2:]            # (1,300,96,2)
sampling_locations = loc4.reshape(bs, q, h, l, p, 2)
```

This matches the 6-D graph bitwise (`max|d| = 0` vs the original ONNX) and
restores `out[2] max = +2.7969` on the board. Enabling `squeeze_nd_mul` is
not a viable alternative for this model: the build then crashes inside the
rule itself (section above).

## Request

1. If a 6-D broadcast `Mul` cannot run on the NPU, reject it at build time
   (or emit an explicit CPU fallback) instead of silently returning zeros.
2. If it can run, fix the codegen/kernel so `squeeze_nd_mul` is not required
   for correctness.
3. Consider warning when `squeeze_nd_mul` is disabled and the graph still
   contains an ND (>4) broadcast `Mul`.
4. Fix the `AttributeError: 'IRGraph' object has no attribute
   'node_use_target_const'` in `_p_squeeze_nd_elementwise_op` so the rule can
   run on real graphs. Happy to test a patched wheel on our model.

## Attachments

- `minimal/m2.onnx`, `minimal/make.py`, `minimal/build.py`,
  `minimal/run_board.py`, `minimal/ref/m2_ref.npz`
- `minimal/build/m2_opt0_default.rknn`, `m2_opt3_default.rknn`,
  `m2_opt0_sqoff.rknn`, `m2_opt3_sqoff.rknn`
- `logs/full_model_squeeze_enabled_crash.log` (full-model build traceback)
- Variant generators and results in `ctx/` (`v0`–`v3`, references in `ctx/ref/`)
