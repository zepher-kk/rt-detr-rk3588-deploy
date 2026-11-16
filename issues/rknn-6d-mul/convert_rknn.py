"""Convert the minimal ONNX to RKNN with Mul assigned to NPU or CPU.

Usage:
    python convert_rknn.py mul6d.onnx mul6d_npu.rknn --op-target npu
    python convert_rknn.py mul6d.onnx mul6d_cpu.rknn --op-target cpu
    python convert_rknn.py mul4d.onnx mul4d_npu.rknn --op-target npu
"""
import argparse

from rknn.api import RKNN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("out")
    ap.add_argument("--op-target", choices=["npu", "cpu", "gpu", "auto"],
                    default="npu")
    ap.add_argument("--opt-level", type=int, default=0)
    args = ap.parse_args()

    rknn = RKNN(verbose=True)
    rknn.config(
        target_platform="rk3588",
        optimization_level=args.opt_level,
        op_target={"Mul": args.op_target},
    )
    if rknn.load_onnx(model=args.onnx) != 0:
        raise SystemExit("load_onnx failed")
    if rknn.build(do_quantization=False) != 0:
        raise SystemExit("build failed")
    if rknn.export_rknn(args.out) != 0:
        raise SystemExit("export failed")
    rknn.release()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
