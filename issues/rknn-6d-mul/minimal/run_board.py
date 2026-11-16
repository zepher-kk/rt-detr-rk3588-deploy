"""Run the four reproducer builds on the board and compare with ORT.

Usage (board, rknn-toolkit-lite2 2.5.0):
    python run_board.py

Expected on RK3588 + librknnrt 2.5.0:

    m2_opt0_default   nz=57600/57600   ok
    m2_opt3_default   nz=57600/57600   ok
    m2_opt0_sqoff     nz=0/57600       <-- ALL ZERO
    m2_opt3_sqoff     nz=0/57600       <-- ALL ZERO
"""
import os

import numpy as np
from rknnlite.api import RKNNLite

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, 'ref', 'm2_ref.npz')
BUILD = os.path.join(HERE, 'build')

TAGS = ('m2_opt0_default', 'm2_opt3_default',
        'm2_opt0_sqoff', 'm2_opt3_sqoff')


def main():
    d = np.load(REF)
    feed = [d['off_in'], d['ref_in']]
    ref = d['out']
    for name in TAGS:
        path = os.path.join(BUILD, name + '.rknn')
        rk = RKNNLite()
        if rk.load_rknn(path) != 0:
            raise SystemExit('load_rknn failed: ' + path)
        if rk.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0:
            raise SystemExit('init_runtime failed')
        out = np.asarray(rk.inference(inputs=feed)[0])
        rk.release()
        nz = int(np.count_nonzero(out))
        verdict = '<-- ALL ZERO' if nz == 0 else 'ok'
        print(f'{name:18s} nz={nz}/{out.size} max={out.max():.6f} '
              f'max|d|={np.abs(out - ref).max():.6f} {verdict}')


if __name__ == '__main__':
    main()
