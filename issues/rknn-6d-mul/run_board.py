"""Run an RKNN model on the board and print output stats.

Usage:
    export LD_LIBRARY_PATH=/home/cat/opencl_shim:$LD_LIBRARY_PATH
    python run_board.py mul6d_npu.rknn --names offsets6d,ref6d,out6d
    python run_board.py mul6d_cpu.rknn --names offsets6d,ref6d,out6d
    python run_board.py mul4d_npu.rknn --names out4d
"""
import argparse

import numpy as np
from rknnlite.api import RKNNLite


def stats(name, a):
    a = np.asarray(a)
    print(f"{name:10s} shape={a.shape} min={a.min():.6g} max={a.max():.6g} "
          f"nonzero={np.count_nonzero(a)}/{a.size}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--names", default="offsets6d,ref6d,out6d")
    args = ap.parse_args()

    rk = RKNNLite()
    if rk.load_rknn(args.model) != 0:
        raise SystemExit("load failed")
    if rk.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0:
        raise SystemExit("init failed")

    rng = np.random.default_rng(0)
    a = rng.standard_normal((1, 300, 96, 2), dtype=np.float32)
    b = rng.standard_normal((1, 300, 1, 2), dtype=np.float32)
    outs = rk.inference(inputs=[a, b], data_format=["nchw", "nchw"])
    for name, out in zip(args.names.split(","), outs):
        stats(name, out)


if __name__ == "__main__":
    main()
