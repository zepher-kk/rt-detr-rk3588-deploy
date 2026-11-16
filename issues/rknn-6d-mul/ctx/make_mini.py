"""Two extra minimal variants to isolate the trigger.

m1: both operands arrive as 3-D/2-D inputs and are Reshaped to 6-D,
    then broadcast-Mul (no Slice, no Div).
m2: the 6-D operand is a graph input; only the small operand is Reshaped.
"""
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'mini')
os.makedirs(OUT, exist_ok=True)


def const(name, arr):
    return numpy_helper.from_array(np.asarray(arr, dtype=np.int64), name)


def make(m1=True):
    nodes, inits = [], []
    inputs = [helper.make_tensor_value_info(
        'off_in', TensorProto.FLOAT, [1, 300, 192] if m1 else [1, 300, 8, 3, 4, 2])]
    inputs.append(helper.make_tensor_value_info('ref_in', TensorProto.FLOAT, [1, 300, 2]))

    if m1:
        inits.append(const('shape_off', [1, 300, 8, 3, 4, 2]))
        nodes.append(helper.make_node('Reshape', ['off_in', 'shape_off'], ['off6']))
        main = 'off6'
    else:
        main = 'off_in'

    inits.append(const('shape_ref', [1, 300, 1, 1, 1, 2]))
    nodes.append(helper.make_node('Reshape', ['ref_in', 'shape_ref'], ['ref6']))
    nodes.append(helper.make_node('Mul', [main, 'ref6'], ['out']))

    g = helper.make_graph(nodes, 'mini', inputs,
                          [helper.make_tensor_value_info('out', TensorProto.FLOAT,
                                                         [1, 300, 8, 3, 4, 2])],
                          initializer=inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 15)])
    m.ir_version = 8
    return m


rng = np.random.default_rng(0)
off3 = rng.standard_normal((1, 300, 192), dtype=np.float32)
off6 = rng.standard_normal((1, 300, 8, 3, 4, 2), dtype=np.float32)
ref = rng.standard_normal((1, 300, 2), dtype=np.float32)

for tag, m1 in (('m1', True), ('m2', False)):
    path = os.path.join(OUT, f'{tag}.onnx')
    onnx.save(make(m1), path)
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    feed = {'off_in': off3 if m1 else off6, 'ref_in': ref}
    out = sess.run(None, feed)[0]
    np.savez(os.path.join(OUT, f'{tag}_ref.npz'), **feed, out=out)
    print(f'{tag}: {os.path.getsize(path)} bytes, out range '
          f'{out.min():.4f}..{out.max():.4f}')
