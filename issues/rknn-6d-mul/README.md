# RKNN 2.5 6D 广播 Mul 全 0：最小复现包

这个目录用来给 rknn-toolkit2 提 issue。

**当前结论（2026-11-16）**：触发条件是
`disable_rules=['squeeze_nd_mul']`，与 `optimization_level` 无关。
同一份 2 节点 ONNX，开规则正确、关规则全 0；完整模型开规则会崩在
`rules/dims.py:1699`。详见 `issue.md` 与 `HANDOFF.md`。

## 目录

| 路径 | 说明 |
|---|---|
| `issue.md` | GitHub issue 正文（英文，已完成，可直接贴） |
| `HANDOFF.md` | 中文交接：结论、已否定假设、板子信息、待办 |
| `minimal/` | **当前的最小复现包**，自包含，含 4 个预编译模型 |
| `logs/` | 完整模型启用 `squeeze_nd_mul` 时的构建崩溃日志 |
| `ctx/` | 变体实验（v0–v3 + ORT 参考），用于证明“操作数没问题、只有 Mul 坏” |
| `make_models.py`、`mul6d.onnx`、`mul4d.onnx`、`convert_rknn.py`、`run_ort.py`、`run_board.py` | **第一版复现（已被 `minimal/` 取代，保留作记录）** |

## 最小复现（用这个）

```bash
cd minimal

# PC / WSL，rknn-toolkit2 2.5.0
python make.py            # 生成 m2.onnx + ref/m2_ref.npz
python build.py           # 生成 build/ 下 4 个 .rknn

# 整个 minimal/ 拷到板子，rknn-toolkit-lite2 2.5.0
python run_board.py
```

`build.py` 只改一个变量：

```python
CONFIGS = (
    ('m2_opt0_default', 0, None),
    ('m2_opt3_default', 3, None),
    ('m2_opt0_sqoff',   0, ['squeeze_nd_mul']),
    ('m2_opt3_sqoff',   3, ['squeeze_nd_mul']),
)
```

板端预期：

```text
m2_opt0_default    nz=57600/57600  ok
m2_opt3_default    nz=57600/57600  ok
m2_opt0_sqoff      nz=0/57600      ALL ZERO
m2_opt3_sqoff      nz=0/57600      ALL ZERO
```

## 第一版复现（已被取代）

`make_models.py` + `convert_rknn.py` 生成的是：

```text
offsets_in (1,300,96,2) --Reshape--> (1,300,8,3,4,2)
ref_in     (1,300,1,2)  --Reshape--> (1,300,1,1,1,2)
Mul -> out6d (1,300,8,3,4,2)
```

它**不会复现**：`convert_rknn.py` 没有设置 `disable_rules`，优化器会启用
`squeeze_nd_mul` 把 6D Mul 降级掉。这些文件保留下来是为了记录排查过程；
提 issue 时用 `minimal/`，不要用这一版。

如果仍想跑：`python make_models.py && python run_ort.py`，再
`python convert_rknn.py mul6d.onnx mul6d_npu.rknn --op-target npu`，
结果会是正确的——这本身就是“规则修好了问题”的旁证。

## 提 issue 时附上

- `issue.md` 正文；
- `minimal/` 整个目录（或打包成 zip）：
  `m2.onnx`、`make.py`、`build.py`、`run_board.py`、`ref/m2_ref.npz`、
  `build/*.rknn`；
- 板端 `run_board.py` 的完整输出；
- `librknnrt.so` 和 rknn-toolkit2 的版本号（都是 2.5.0）；
- 完整模型的构建 traceback（`issue.md` 的
  “Two failures of the same rule”一节，日志在
  `logs/full_model_squeeze_enabled_crash.log`）。

## 备注

- `issue.md` 是英文的，Rockchip 的 issue 区通常用英文更快得到回复；
- 不要附带闭源库或反编译内容，黑盒复现就够了；
- 诉求是：不支持就 build 期报错或显式 fallback，不要静默返回 0；
  同时修 `_p_squeeze_nd_elementwise_op` 的 `AttributeError`。
