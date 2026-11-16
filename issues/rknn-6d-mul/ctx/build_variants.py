"""Build every context variant with rknn-toolkit2 2.5.0 (run in WSL).

Requires the same environment that produced the failing model:
    rknn-toolkit2 2.5.0, Python 3.10, target rk3588

Usage (from this directory):
    python build_variants.py

Outputs land in ./build/, which is visible from Windows, so the board
runner can pick them up without copying.
"""
import json
import os
import time

from rknn.api import RKNN

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")


def build(onnx_path, out_path, op_target="npu", opt_level=3):
    rk = RKNN()
    ret = rk.config(
        target_platform="rk3588",
        optimization_level=opt_level,
        op_target={"Mul": op_target},
        disable_rules=["squeeze_nd_mul", "fuse_mul_into_gemm"],
    )
    if ret != 0:
        raise SystemExit("config failed: %s" % out_path)
    ret = rk.load_onnx(model=onnx_path)
    if ret != 0:
        raise SystemExit("load_onnx failed: %s" % onnx_path)
    t0 = time.time()
    ret = rk.build(do_quantization=False)
    if ret != 0:
        raise SystemExit("build failed: %s" % onnx_path)
    ret = rk.export_rknn(out_path)
    if ret != 0:
        raise SystemExit("export failed: %s" % out_path)
    rk.release()
    print("OK %-28s %6.1fs %8d bytes" % (
        os.path.basename(out_path), time.time() - t0,
        os.path.getsize(out_path)))


def main():
    os.makedirs(BUILD, exist_ok=True)
    with open(os.path.join(HERE, "variants.json")) as f:
        variants = json.load(f)

    for v in variants:
        name = v["model"][:-5]
        onnx_path = os.path.join(HERE, v["model"])

        # 1) the failing configuration: Mul on NPU, default opt level
        build(onnx_path, os.path.join(BUILD, name + "_npu.rknn"))

        # 2) control: Mul on CPU
        if name == "v3_both":
            build(onnx_path, os.path.join(BUILD, name + "_cpu.rknn"),
                  op_target="cpu")

        # 3) does the optimizer stage matter?
        if name in ("v1_div", "v3_both"):
            build(onnx_path, os.path.join(BUILD, name + "_l0_npu.rknn"),
                  opt_level=0)

    print("\nall models in", BUILD)


if __name__ == "__main__":
    main()
