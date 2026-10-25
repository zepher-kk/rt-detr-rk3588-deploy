#!/usr/bin/env python3
"""Convert an ONNX detector with an explicit RKNN quantization experiment."""

import argparse
import time
from pathlib import Path

from rknn.api import RKNN


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--algorithm", default="normal")
    parser.add_argument("--method", default="layer")
    parser.add_argument("--dtype", default="w8a8")
    parser.add_argument("--optimization-level", type=int, default=0)
    parser.add_argument("--auto-hybrid", action="store_true")
    parser.add_argument("--cos-threshold", type=float, default=0.98)
    parser.add_argument("--euc-threshold", type=float)
    parser.add_argument("--flash-attention", action="store_true")
    parser.add_argument("--crop-inputs", action="store_true")
    parser.add_argument("--disable-rule", action="append")
    parser.add_argument("--compress-weight", action="store_true")
    parser.add_argument("--remove-reshape", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    disabled_rules = ["fuse_mul_into_gemm"]
    for rule in args.disable_rule or []:
        if rule not in disabled_rules:
            disabled_rules.append(rule)
    config = {
        "mean_values": [[0, 0, 0]],
        "std_values": [[255, 255, 255]],
        "target_platform": "rk3588",
        "optimization_level": args.optimization_level,
        "quantized_dtype": args.dtype,
        "quantized_algorithm": args.algorithm,
        "quantized_method": args.method,
        "disable_rules": disabled_rules,
    }
    if args.compress_weight:
        config["compress_weight"] = True
    if args.remove_reshape:
        config["remove_reshape"] = True
    if args.auto_hybrid:
        config["auto_hybrid_cos_thresh"] = args.cos_threshold
        if args.euc_threshold is not None:
            config["auto_hybrid_euc_thresh"] = args.euc_threshold
    if args.flash_attention:
        config["enable_flash_attention"] = True

    print(f"config={config}", flush=True)
    started = time.monotonic()
    rknn = RKNN(verbose=args.verbose)
    ret = rknn.config(**config)
    print(f"config_ret={ret}", flush=True)
    if ret != 0:
        rknn.release()
        return ret

    load_args = {"model": str(args.onnx)}
    if args.crop_inputs:
        load_args.update(
            {"inputs": ["images"], "input_size_list": [[1, 3, 640, 640]]}
        )
    ret = rknn.load_onnx(**load_args)
    print(f"load_ret={ret}", flush=True)
    if ret != 0:
        rknn.release()
        return ret

    ret = rknn.build(
        do_quantization=True,
        dataset=str(args.dataset),
        auto_hybrid=args.auto_hybrid,
    )
    elapsed = time.monotonic() - started
    print(f"build_ret={ret} elapsed_s={elapsed:.1f}", flush=True)
    if ret != 0:
        rknn.release()
        return ret

    ret = rknn.export_rknn(str(args.output))
    elapsed = time.monotonic() - started
    print(f"export_ret={ret} elapsed_s={elapsed:.1f} output={args.output}", flush=True)
    rknn.release()
    return ret


if __name__ == "__main__":
    raise SystemExit(main())
