"""Run every context variant in ONNX Runtime and save reference outputs.

Writes ref/<variant>.npz with the fixed inputs and the expected outputs,
so the board can compare bit-for-bit (modulo FP16).

Usage:
    python run_ort_ctx.py
"""
import json
import os

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(HERE, "ref")
Q = 300


def main():
    os.makedirs(REF_DIR, exist_ok=True)
    with open(os.path.join(HERE, "variants.json")) as f:
        variants = json.load(f)

    rng = np.random.default_rng(0)
    # Same seed per variant => identical inputs across builds.
    for v in variants:
        name = v["model"][:-5]
        off_shape = tuple(v["inputs"]["off_in"])
        bbox = rng.standard_normal((1, Q, 4), dtype=np.float32)
        off = rng.standard_normal(off_shape, dtype=np.float32)

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        sess = ort.InferenceSession(os.path.join(HERE, v["model"]), opts,
                                    providers=["CPUExecutionProvider"])
        outs = sess.run(None, {"off_in": off, "bbox_in": bbox})

        payload = {"off_in": off, "bbox_in": bbox}
        for out_name, arr in zip(v["outputs"], outs):
            payload[out_name] = np.asarray(arr)
            print("%-12s %-8s %-18s min=% .5g max=% .5g nz=%d/%d" % (
                name, out_name, str(arr.shape), arr.min(), arr.max(),
                np.count_nonzero(arr), arr.size))
        np.savez(os.path.join(REF_DIR, name + ".npz"), **payload)
    print("refs written to", REF_DIR)


if __name__ == "__main__":
    main()
