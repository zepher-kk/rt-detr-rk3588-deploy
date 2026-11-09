"""v4.1 hybrid quantization: auto-hybrid + per-channel.

Keeps sensitive layers at higher precision (auto_hybrid_cos_thresh) while
quantizing the rest to INT8. GridSample must stay FP16 (GPU op).
"""
import argparse
import os
import sys

from rknn.api import RKNN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('onnx')
    p.add_argument('out')
    p.add_argument('--dataset', required=True)
    p.add_argument('--platform', default='rk3588')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--opt-level', type=int, default=0)
    p.add_argument('--method', default='channel', choices=['layer', 'channel'])
    p.add_argument('--algorithm', default='normal')
    p.add_argument('--cos-thresh', type=float, default=0.98)
    p.add_argument('--euc-thresh', type=float, default=None)
    p.add_argument('--no-auto-hybrid', action='store_true')
    p.add_argument('--op-target-map', default='GridSample=gpu,Mul=cpu')
    p.add_argument('--disable-rules', default='squeeze_nd_mul,fuse_mul_into_gemm')
    return p.parse_args()


def main():
    args = parse_args()

    op_target = None
    if args.op_target_map:
        op_target = {}
        for pair in args.op_target_map.split(','):
            if not pair.strip():
                continue
            k, v = pair.split('=')
            op_target[k.strip()] = v.strip()

    rules = [r for r in args.disable_rules.split(',') if r] or None

    cfg = dict(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=args.platform,
        optimization_level=args.opt_level,
        quantized_dtype='w8a8',
        quantized_method=args.method,
        quantized_algorithm=args.algorithm,
        auto_hybrid_cos_thresh=args.cos_thresh,
        op_target=op_target,
    )
    if args.euc_thresh is not None:
        cfg['auto_hybrid_euc_thresh'] = args.euc_thresh
    if rules:
        cfg['disable_rules'] = rules

    print('config:', cfg, flush=True)
    rknn = RKNN(verbose=True)
    if rknn.config(**cfg) != 0:
        print('config failed')
        return 1

    if rknn.load_onnx(model=args.onnx, inputs=['images'],
                      input_size_list=[[1, 3, args.imgsz, args.imgsz]]) != 0:
        print('load_onnx failed')
        return 1

    if rknn.build(do_quantization=True, dataset=args.dataset,
                  auto_hybrid=not args.no_auto_hybrid) != 0:
        print('build failed')
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if rknn.export_rknn(args.out) != 0:
        print('export failed')
        return 1

    print('BUILD OK ->', args.out, os.path.getsize(args.out), 'bytes')
    rknn.release()
    return 0


if __name__ == '__main__':
    sys.exit(main())
