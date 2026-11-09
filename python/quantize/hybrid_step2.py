"""Edit the hybrid quantization cfg: mark a prefix region as float16, then step2."""
import argparse
import os
import re
import sys

from rknn.api import RKNN


def edit_cfg(src, dst, prefix):
    lines = open(src, encoding='utf-8').read().splitlines()
    out = []
    in_block = False
    changed = 0
    for line in lines:
        stripped = line.strip()
        # tensor blocks are indented 4 spaces and end with ':'
        if line.startswith('    ') and not line.startswith('        ') and stripped.endswith(':'):
            in_block = stripped[:-1].startswith(prefix)
        if in_block and stripped.startswith('dtype:') and 'int8' in stripped:
            indent = line[:len(line) - len(line.lstrip())]
            line = f'{indent}dtype: float16'
            changed += 1
        out.append(line)
    open(dst, 'w', encoding='utf-8').write('\n'.join(out) + '\n')
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workdir', required=True)
    ap.add_argument('--base', required=True, help='basename of .model/.data/.cfg')
    ap.add_argument('--prefix', default='/model.26/')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.chdir(args.workdir)
    cfg = f'{args.base}.quantization.cfg'
    edited = f'{args.base}.fp16region.cfg'
    n = edit_cfg(cfg, edited, args.prefix)
    print(f'cfg tensors switched to float16: {n} -> {edited}', flush=True)
    if n == 0:
        print('WARNING: no tensors matched the prefix')

    rknn = RKNN(verbose=True)
    ret = rknn.hybrid_quantization_step2(
        model_input=f'{args.base}.model',
        data_input=f'{args.base}.data',
        model_quantization_cfg=edited,
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
