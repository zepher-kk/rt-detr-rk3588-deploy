"""Regenerate the minimal reproducer (m2.onnx) and its ORT reference.

Graph (5 nodes, 2 inputs, 1 output):

    off_in (1,300,8,3,4,2) -----------------------------.
                                                        Mul --> out (1,300,8,3,4,2)
    ref_in (1,300,2) --Reshape--> (1,300,1,1,1,2) ------'

On RK3588 + toolkit/runtime 2.5.0 the NPU output is all zeros.
"""
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, 'ref')
os.makedirs(REF, exist_ok=True)

shape = numpy_helper.from_array(np.asarray([1, 300, 1, 1, 1, 2], dtype=np.int64), 'shape_ref')

nodes = [
    helper.make_node('Reshape', ['ref_in', 'shape_ref'], ['ref6']),
    helper.make_node('Mul', ['off_in', 'ref6'], ['out']),
]
inputs = [
    helper.make_tensor_value_info('off_in', TensorProto.FLOAT, [1, 300, 8, 3, 4, 2]),
    helper.make_tensor_value_info('ref_in', TensorProto.FLOAT, [1, 300, 2]),
]
outputs = [
    helper.make_tensor_value_info('out', TensorProto.FLOAT, [1, 300, 8, 3, 4, 2]),
]
graph = helper.make_graph(nodes, 'mul6d_min', inputs, outputs, initializer=[shape])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 15)])
model.ir_version = 8
onnx_path = os.path.join(HERE, 'm2.onnx')
onnx.save(model, onnx_path)
print('wrote', onnx_path, os.path.getsize(onnx_path), 'bytes')

rng = np.random.default_rng(0)
off = rng.standard_normal((1, 300, 8, 3, 4, 2)).astype(np.float32)
ref = rng.standard_normal((1, 300, 2)).astype(np.float32)
sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
out = sess.run(None, {'off_in': off, 'ref_in': ref})[0]
np.savez(os.path.join(REF, 'm2_ref.npz'), off_in=off, ref_in=ref, out=out)
print(f'ORT: out min={out.min():.4f} max={out.max():.4f} '
      f'nonzero={int(np.count_nonzero(out))}/{out.size}')
