import argparse
import os
import sys

from rknn.api import RKNN


def parse_args():
    p = argparse.ArgumentParser(description="RKNN 2.5 native GridSample GPU build")
    p.add_argument('onnx', help='input ONNX path')
    p.add_argument('out', help='output RKNN path')
    p.add_argument('--platform', default='rk3588')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--opt-level', type=int, default=3)
    p.add_argument('--op-target', default='gpu',
                   choices=['gpu', 'cpu', 'npu', 'auto', 'none'],
                   help="single device applied to GridSample; use --op-target-map for multi-op")
    p.add_argument('--op-target-map', default='',
                   help="comma separated Op=device pairs, e.g. 'GridSample=gpu,Mul=gpu'")
    p.add_argument('--disable-rules', default='squeeze_nd_mul',
                   help="comma separated; empty string means no disabled rules")
    p.add_argument('--mul-cpu-scope', default='none',
                   choices=['none', 'all'],
                   help="place all Mul ops on CPU (fallback for models not "
                        "exported with --msda-4d-grid)")
    return p.parse_args()


def main():
    args = parse_args()

    rules = [r for r in args.disable_rules.split(',') if r]
    op_target = None if args.op_target == 'none' else {'GridSample': args.op_target}
    if args.op_target_map:
        op_target = {}
        for pair in args.op_target_map.split(','):
            if not pair.strip():
                continue
            key, val = pair.split('=')
            op_target[key.strip()] = val.strip()

    if args.mul_cpu_scope == 'all':
        if op_target is None:
            op_target = {}
        op_target['Mul'] = 'cpu'

    print('==== build config ====')
    print('onnx        :', args.onnx)
    print('output      :', args.out)
    print('platform    :', args.platform)
    print('imgsz       :', args.imgsz)
    print('opt_level   :', args.opt_level)
    print('op_target   :', op_target)
    print('disable_rules:', rules)
    print('======================')

    rknn = RKNN(verbose=True)
    config_kwargs = dict(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=args.platform,
        optimization_level=args.opt_level,
        op_target=op_target,
    )
    if rules:
        config_kwargs['disable_rules'] = rules
    rknn.config(**config_kwargs)

    ret = rknn.load_onnx(
        model=args.onnx,
        inputs=['images'],
        input_size_list=[[1, 3, args.imgsz, args.imgsz]],
    )
    if ret != 0:
        print('load_onnx failed:', ret)
        sys.exit(ret)

    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('build failed:', ret)
        sys.exit(ret)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    ret = rknn.export_rknn(args.out)
    if ret != 0:
        print('export failed:', ret)
        sys.exit(ret)

    print('BUILD OK ->', args.out)
    rknn.release()


if __name__ == '__main__':
    main()
