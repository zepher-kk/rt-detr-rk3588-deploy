"""Mark decoder tensors as float16 via custom_quantize_layers, then step2."""
import argparse
import os
import sys

from rknn.api import RKNN


def build_cfg(src, dst, prefix, skip_substrings, keep_substrings):
    lines = open(src, encoding='utf-8').read().splitlines()
    keys = []
    for line in lines:
        s = line.strip()
        if line.startswith('    ') and not line.startswith('        ') and s.endswith(':'):
            name = s[:-1]
            if not name.startswith(prefix):
                continue
            if any(x in name for x in skip_substrings):
                continue
            if keep_substrings and not any(x in name for x in keep_substrings):
                continue
            keys.append(name)

    out = []
    for line in lines:
        if line.strip() == 'custom_quantize_layers: {}':
            out.append('custom_quantize_layers:')
            for k in keys:
                out.append(f'    {k}: float16')
        else:
            out.append(line)
    open(dst, 'w', encoding='utf-8').write('\n'.join(out) + '\n')
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workdir', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--prefix', default='/model.26/')
    ap.add_argument('--skip', default='input_proj')
    ap.add_argument('--keep', default='Conv,Gemm,MatMul')
    ap.add_argument('--out', required=True)
    ap.add_argument('--op-target-map', default='GridSample=gpu,Mul=cpu')
    args = ap.parse_args()

    os.chdir(args.workdir)
    src = f'{args.base}.quantization.cfg'
    dst = f'{args.base}.custom.cfg'
    skip = [x for x in args.skip.split(',') if x]
    keys = build_cfg(src, dst, args.prefix, skip,
                     [x for x in args.keep.split(',') if x])
    print(f'custom_quantize_layers entries: {len(keys)}')
    if not keys:
        print('no keys matched')
        return 1

    rknn = RKNN(verbose=True)
    op_target = None
    if args.op_target_map:
        op_target = {}
        for pair in args.op_target_map.split(','):
            if pair.strip():
                k, v = pair.split('=')
                op_target[k.strip()] = v.strip()
        if not op_target:
            op_target = None
    cfg = dict(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform='rk3588',
        optimization_level=0,
        op_target=op_target,
        disable_rules=['squeeze_nd_mul', 'fuse_mul_into_gemm'],
    )
    ret = rknn.config(**cfg)
    print('config ret:', ret, 'op_target:', op_target, flush=True)
    ret = rknn.hybrid_quantization_step2(
        model_input=f'{args.base}.model',
        data_input=f'{args.base}.data',
        model_quantization_cfg=dst,
    )
    if ret != 0:
        print('step2 failed')
        return 1
    if rknn.export_rknn(args.out) != 0:
        print('export failed')
        return 1
    print('BUILD OK ->', args.out, os.path.getsize(args.out), 'bytes')
    rknn.release()
    return 0


if __name__ == '__main__':
    sys.exit(main())
