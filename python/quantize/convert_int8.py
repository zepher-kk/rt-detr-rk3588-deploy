"""Build an INT8 RKNN model (no GridSample: backbone / neck only)."""
import argparse
import os
import sys

from rknn.api import RKNN


def parse_args():
    p = argparse.ArgumentParser(description="RKNN 2.5 INT8 build")
    p.add_argument('onnx')
    p.add_argument('out')
    p.add_argument('--dataset', required=True,
                   help='calibration image list (one path per line)')
    p.add_argument('--platform', default='rk3588')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--opt-level', type=int, default=0)
    p.add_argument('--disable-rules', default='squeeze_nd_mul,fuse_mul_into_gemm')
    p.add_argument('--quantized-dtype', default='w8a8')
    p.add_argument('--quantized-method', default='channel')
    p.add_argument('--custom-string', default='',
                   help='build id stored in the model (checked by the runtime '
                        'when chaining backbone + decoder)')
    return p.parse_args()


def main():
    args = parse_args()
    rules = [r for r in args.disable_rules.split(',') if r]
    rknn = RKNN(verbose=True)
    kwargs = dict(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=args.platform,
        optimization_level=args.opt_level,
        quantized_dtype=args.quantized_dtype,
        quantized_method=args.quantized_method,
    )
    if args.custom_string:
        kwargs['custom_string'] = args.custom_string
    if rules:
        kwargs['disable_rules'] = rules
    if rknn.config(**kwargs) != 0:
        print('config failed')
        return 1

    if rknn.load_onnx(model=args.onnx, inputs=['images'],
                      input_size_list=[[1, 3, args.imgsz, args.imgsz]]) != 0:
        print('load_onnx failed')
        return 1

    if rknn.build(do_quantization=True, dataset=args.dataset) != 0:
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
