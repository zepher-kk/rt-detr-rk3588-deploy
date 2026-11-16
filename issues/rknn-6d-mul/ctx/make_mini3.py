"""m3 = m2 but the small operand is expanded with Unsqueeze instead of Reshape."""
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'mini')
os.makedirs(OUT, exist_ok=True)


def const(name, arr, dtype=np.int64):
    return numpy_helper.from_array(np.asarray(arr, dtype=dtype), name)


inputs = [
    helper.make_tensor_value_info('off_in', TensorProto.FLOAT, [1, 300, 8, 3, 4, 2]),
    helper.make_tensor_value_info('ref_in', TensorProto.FLOAT, [1, 300, 2]),
]
inits = [const('axes', [2, 3, 4])]
nodes = [
    helper.make_node('Unsqueeze', ['ref_in', 'axes'], ['ref6']),
    helper.make_node('Mul', ['off_in', 'ref6'], ['out']),
]
g = helper.make_graph(nodes, 'm3', inputs,
                      [helper.make_tensor_value_info('out', TensorProto.FLOAT,
                                                     [1, 300, 8, 3, 4, 2])],
                      initializer=inits)
m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 15)])
m.ir_version = 8
path = os.path.join(OUT, 'm3.onnx')
onnx.save(m, path)

rng = np.random.default_rng(0)
off6 = rng.standard_normal((1, 300, 8, 3, 4, 2), dtype=np.float32)
ref = rng.standard_normal((1, 300, 2), dtype=np.float32)
sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
out = sess.run(None, {'off_in': off6, 'ref_in': ref})[0]
np.savez(os.path.join(OUT, 'm3_ref.npz'), off_in=off6, ref_in=ref, out=out)
print('m3 onnx', os.path.getsize(path), 'out', out.min(), out.max())
