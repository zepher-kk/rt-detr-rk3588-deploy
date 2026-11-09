"""Step 1 of region-based hybrid quantization: generate .model/.data/.quantization.cfg."""
import argparse
import os
import sys

from rknn.api import RKNN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('onnx')
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--workdir', required=True)
    ap.add_argument('--op-target-map', default='GridSample=gpu,Mul=cpu')
    ap.add_argument('--imgsz', type=int, default=640)
    args = ap.parse_args()

    op_target = None
    if args.op_target_map:
        op_target = {}
        for pair in args.op_target_map.split(','):
            if pair.strip():
                k, v = pair.split('=')
                op_target[k.strip()] = v.strip()
        if not op_target:
            op_target = None

    os.makedirs(args.workdir, exist_ok=True)
    os.chdir(args.workdir)

    rknn = RKNN(verbose=True)
    cfg = dict(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform='rk3588',
        optimization_level=0,
        quantized_dtype='w8a8',
        quantized_method='channel',
        op_target=op_target,
        disable_rules=['squeeze_nd_mul', 'fuse_mul_into_gemm'],
    )
    if rknn.config(**cfg) != 0:
        print('config failed')
        return 1
    if rknn.load_onnx(model=args.onnx, inputs=['images'],
                      input_size_list=[[1, 3, args.imgsz, args.imgsz]]) != 0:
        print('load failed')
        return 1

    # proposal=True requires batch expansion, which this static model rejects.
    if rknn.hybrid_quantization_step1(dataset=args.dataset, proposal=False) != 0:
        print('step1 failed')
        return 1

    print('STEP1 OK; files:')
    for f in sorted(os.listdir('.')):
        print('  ', f, os.path.getsize(f))
    rknn.release()
    return 0


if __name__ == '__main__':
    sys.exit(main())
