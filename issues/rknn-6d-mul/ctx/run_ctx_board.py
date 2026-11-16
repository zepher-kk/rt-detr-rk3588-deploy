"""Run the context variants on the board and compare with ONNX Runtime.

Usage (board, rknn38 env):
    python run_ctx_board.py

Reads ./build/*.rknn and ./ref/*.npz.  A variant is "reproduced" when
mul / mul1 come back exactly all-zero while the operands are correct.
"""
import glob
import json
import os

import numpy as np
from rknnlite.api import RKNNLite

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "ref")
BUILD = os.path.join(HERE, "build")


def base_of(rknn_path):
    name = os.path.basename(rknn_path)[:-5]
    for suffix in ("_l0_npu", "_npu", "_cpu"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def main():
    with open(os.path.join(HERE, "variants.json")) as f:
        variants = {v["model"][:-5]: v for v in json.load(f)}

    for path in sorted(glob.glob(os.path.join(BUILD, "*.rknn"))):
        name = base_of(path)
        v = variants[name]
        ref = np.load(os.path.join(REF, name + ".npz"))
        inputs = [ref["off_in"], ref["bbox_in"]]

        rk = RKNNLite()
        if rk.load_rknn(path) != 0:
            print("load failed:", path)
            continue
        if rk.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0:
            print("init failed:", path)
            continue
        outs = rk.inference(inputs=inputs,
                            data_format=["nchw"] * len(inputs))
        rk.release()

        tag = os.path.basename(path)
        print("\n== %s" % tag)
        bad = False
        for out_name, arr in zip(v["outputs"], outs):
            a = np.asarray(arr)
            r = np.asarray(ref[out_name])
            if a.shape != r.shape:
                print("   %-8s shape %s != ref %s" % (out_name, a.shape, r.shape))
                continue
            nz = np.count_nonzero(a)
            diff = np.abs(a.astype(np.float32) - r).max()
            flag = ""
            if out_name in ("mul", "mul1") and nz == 0:
                flag = "  <-- ALL ZERO"
                bad = True
            print("   %-8s min=% .5g max=% .5g nz=%d/%d max|d|=%.5g%s" % (
                out_name, a.min(), a.max(), nz, a.size, diff, flag))
        print("   %s" % ("REPRODUCED" if bad else "ok"))

    print("\ndone")


if __name__ == "__main__":
    main()
