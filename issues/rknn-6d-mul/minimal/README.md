# Minimal reproducer: 6-D broadcast `Mul` returns all zeros

RK3588 + rknn-toolkit2 / librknnrt **2.5.0**.

## Graph

```text
off_in (1,300,8,3,4,2) --------------------------------.
                                                        Mul --> out (1,300,8,3,4,2)
ref_in (1,300,2) --Reshape--> (1,300,1,1,1,2) ---------'
```

2 nodes (`Reshape`, `Mul`), 2 inputs, 1 output. `m2.onnx` is 260 bytes.

## Reproduce

```bash
# PC / WSL, rknn-toolkit2 2.5.0
python make.py      # writes m2.onnx and ref/m2_ref.npz (ORT reference)
python build.py     # writes build/*.rknn (four configs)

# copy this whole directory to the board, then
# rknn-toolkit-lite2 2.5.0
python run_board.py
```

`build.py` changes only `disable_rules`; everything else is identical
(`target_platform='rk3588'`, `op_target={'Mul': 'npu'}`,
`do_quantization=False`).

## Result (RK3588, 2026-11-16)

ORT reference: `min=-3.99 max=4.04`, `57600/57600` non-zero.

| Build | `optimization_level` | `disable_rules` | `.rknn` size | non-zero | max\|NPU−ORT\| |
|---|---|---|---|---|---|
| `m2_opt0_default` | 0 | — | 18267 | 57600/57600 | 0.0056 |
| `m2_opt3_default` | 3 | — | 18267 | 57600/57600 | 0.0056 |
| `m2_opt0_sqoff` | 0 | `['squeeze_nd_mul']` | 7451 | **0/57600** | 12.43 |
| `m2_opt3_sqoff` | 3 | `['squeeze_nd_mul']` | 7451 | **0/57600** | 12.43 |

- `0.0056` is normal FP16 error.
- Disabling `squeeze_nd_mul` is the **only** difference that matters;
  `optimization_level` and `fuse_mul_into_gemm` have no effect.
- Build, load and inference all succeed on the failing builds; the output is
  just zeros.

## Controls

| Control | Result |
|---|---|
| 4-D version `(1,300,96,2) × (1,300,1,2)` | correct (57600/57600) |
| `op_target={'Mul': 'cpu'}` | correct |
| `Expand` the 2-element operand to full shape first, then `Mul` | still all zeros |
| `disable_rules=['fuse_mul_into_gemm']` only | correct |

## Related: the same rule crashes on the real model

On the full RT-DETR decoder ONNX (300q / 3L), enabling `squeeze_nd_mul`
crashes `build()`:

```text
File "rknn/api/graph_optimizer.py", line 2221, in GraphOptimizer.fuse_ops
File "rknn/api/rules/dims.py", line 1699, in _p_squeeze_nd_elementwise_op
AttributeError: 'IRGraph' object has no attribute 'node_use_target_const'
```

So the rule cannot be used (crash) and cannot be skipped (zeros) without
rewriting the 6-D multiply as 4-D. See `../issue.md`.
