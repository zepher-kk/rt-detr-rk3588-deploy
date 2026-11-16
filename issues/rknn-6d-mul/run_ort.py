"""Run the two ONNX models in ONNX Runtime (CPU) and print output stats.

Usage:
    python run_ort.py
"""
import numpy as np
import onnxruntime as ort


def stats(name, a):
    a = np.asarray(a)
    print(f"{name:10s} shape={a.shape} min={a.min():.6g} max={a.max():.6g} "
          f"nonzero={np.count_nonzero(a)}/{a.size}")


def inputs():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((1, 300, 96, 2), dtype=np.float32)
    b = rng.standard_normal((1, 300, 1, 2), dtype=np.float32)
    return {"offsets_in": a, "ref_in": b}


def main():
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1

    for model, names in [
        ("mul6d.onnx", ["offsets6d", "ref6d", "out6d"]),
        ("mul4d.onnx", ["out4d"]),
    ]:
        sess = ort.InferenceSession(model, opts, providers=["CPUExecutionProvider"])
        outs = sess.run(None, inputs())
        print("==", model)
        for name, out in zip(names, outs):
            stats(name, out)


if __name__ == "__main__":
    main()
