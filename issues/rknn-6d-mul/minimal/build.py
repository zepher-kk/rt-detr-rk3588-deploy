"""Build the reproducer under four configs (run in WSL, toolkit 2.5.0).

The only variable is `disable_rules`:

    m2_opt0_default   level=0  no disable_rules      -> correct
    m2_opt3_default   level=3  no disable_rules      -> correct
    m2_opt0_sqoff     level=0  disable squeeze_nd_mul -> ALL ZERO
    m2_opt3_sqoff     level=3  disable squeeze_nd_mul -> ALL ZERO
"""
import os
import sys

from rknn.api import RKNN

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, 'build')
ONNX = os.path.join(HERE, 'm2.onnx')

CONFIGS = (
    ('m2_opt0_default', 0, None),
    ('m2_opt3_default', 3, None),
    ('m2_opt0_sqoff', 0, ['squeeze_nd_mul']),
    ('m2_opt3_sqoff', 3, ['squeeze_nd_mul']),
)


def build(tag, level, rules):
    out = os.path.join(BUILD, tag + '.rknn')
    cfg = dict(
        target_platform='rk3588',
        optimization_level=level,
        op_target={'Mul': 'npu'},
    )
    if rules:
        cfg['disable_rules'] = rules
    rk = RKNN()
    if rk.config(**cfg) != 0:
        raise SystemExit('config failed: ' + tag)
    if rk.load_onnx(model=ONNX) != 0:
        raise SystemExit('load_onnx failed: ' + tag)
    if rk.build(do_quantization=False) != 0:
        raise SystemExit('build failed: ' + tag)
    if rk.export_rknn(out) != 0:
        raise SystemExit('export failed: ' + tag)
    rk.release()
    print(f'OK {tag} {os.path.getsize(out)} bytes')


if __name__ == '__main__':
    if not os.path.exists(ONNX):
        sys.exit('run make.py first')
    os.makedirs(BUILD, exist_ok=True)
    for tag, level, rules in CONFIGS:
        build(tag, level, rules)
