"""Build two minimal ONNX models for the 6D broadcast Mul issue.

mul6d.onnx
    offsets_in (1,300,96,2) --Reshape--> offsets6d (1,300,8,3,4,2)
    ref_in     (1,300,1,2)  --Reshape--> ref6d     (1,300,1,1,1,2)
    Mul(offsets6d, ref6d)   -> out6d     (1,300,8,3,4,2)

    offsets6d and ref6d are also graph outputs, so the board can dump the
    operands and verify they are correct while out6d is wrong.

mul4d.onnx
    Control: the same values, but the Mul is done in 4-D
    offsets_in (1,300,96,2) * ref_in (1,300,1,2) -> out4d (1,300,96,2)

Usage:
    python make_models.py
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


OPSET = 16
IR_VERSION = 8
SHAPE_OFFSETS_6D = [1, 300, 8, 3, 4, 2]
SHAPE_REF_6D = [1, 300, 1, 1, 1, 2]
SHAPE_OFFSETS_4D = [1, 300, 96, 2]
SHAPE_REF_4D = [1, 300, 1, 2]


def value_info(name, shape, dtype=TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, dtype, shape)


def save(model, path):
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print("wrote", path)


def make_6d(path="mul6d.onnx"):
    offsets_in = value_info("offsets_in", SHAPE_OFFSETS_4D)
    ref_in = value_info("ref_in", SHAPE_REF_4D)
    offsets6d = value_info("offsets6d", SHAPE_OFFSETS_6D)
    ref6d = value_info("ref6d", SHAPE_REF_6D)
    out6d = value_info("out6d", SHAPE_OFFSETS_6D)

    shape_offsets = numpy_helper.from_array(
        np.asarray(SHAPE_OFFSETS_6D, dtype=np.int64), "shape_offsets")
    shape_ref = numpy_helper.from_array(
        np.asarray(SHAPE_REF_6D, dtype=np.int64), "shape_ref")

    nodes = [
        helper.make_node("Reshape", ["offsets_in", "shape_offsets"], ["offsets6d"]),
        helper.make_node("Reshape", ["ref_in", "shape_ref"], ["ref6d"]),
        helper.make_node("Mul", ["offsets6d", "ref6d"], ["out6d"]),
    ]
    graph = helper.make_graph(
        nodes, "mul6d", [offsets_in, ref_in],
        [offsets6d, ref6d, out6d],
        initializer=[shape_offsets, shape_ref])
    save(helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", OPSET)]), path)


def make_4d(path="mul4d.onnx"):
    offsets_in = value_info("offsets_in", SHAPE_OFFSETS_4D)
    ref_in = value_info("ref_in", SHAPE_REF_4D)
    out4d = value_info("out4d", SHAPE_OFFSETS_4D)
    nodes = [helper.make_node("Mul", ["offsets_in", "ref_in"], ["out4d"])]
    graph = helper.make_graph(nodes, "mul4d", [offsets_in, ref_in], [out4d])
    save(helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", OPSET)]), path)


if __name__ == "__main__":
    make_6d()
    make_4d()
