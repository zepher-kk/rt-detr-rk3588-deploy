"""Generate 6D broadcast Mul variants that mirror the real RT-DETR decoder.

The bare 6D Mul (graph input -> Reshape -> Mul) does NOT reproduce on
RKNN 2.5.0; the real model does.  These variants add the surrounding
context one piece at a time, so we can find the minimal trigger.

Common tail (all variants):

    off6  (1,300,8,3,4,2)
    ref6  (1,300,1,1,1,4) --Slice--> xy (1,300,1,1,1,2), wh (1,300,1,1,1,2)
    mul   = off6 * wh
    mul1  = mul * 0.5
    add   = xy + mul

Outputs are [off/div, mul, mul1, add] -- the same set the failing
rtdetr_tpopt2_dbg2_mulnpu build exposes.

Variants:
    v0_direct    off: Reshape                      ref: Reshape -> Slice
    v1_div       off: Div(8)                       ref: Reshape -> Slice
    v2_sigmoid   off: Reshape                      ref: Sigmoid -> Reshape -> Slice
    v3_both      off: Div(8)                       ref: Sigmoid -> Reshape -> Slice
    v4_both_tp   off: Transpose -> Div(8)          ref: Sigmoid -> Reshape -> Slice
    v5_both_mm   off: MatMul -> Reshape -> Div(8)  ref: Sigmoid -> Reshape -> Slice

Usage:
    python make_ctx_variants.py
"""
import json
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET = 16
IR_VERSION = 8
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

SHAPE_OFF6 = [1, 300, 8, 3, 4, 2]
SHAPE_REF6 = [1, 300, 1, 1, 1, 4]
SHAPE_XY = [1, 300, 1, 1, 1, 2]
Q = 300
OFF_FLAT = 8 * 3 * 4 * 2  # 192


def vi(name, shape, dtype=TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, dtype, shape)


def const(name, value):
    return numpy_helper.from_array(np.asarray(value), name)


def build(name, off_mode, ref_mode):
    nodes = []
    inits = []
    inputs = []

    # ---- offsets branch ----
    if off_mode == "direct":
        inputs.append(vi("off_in", [1, Q, OFF_FLAT]))
        nodes.append(helper.make_node("Reshape", ["off_in", "shape_off6"], ["off6"]))
        inits.append(const("shape_off6", np.asarray(SHAPE_OFF6, dtype=np.int64)))
        off_out = "off6"
        cur = "off6"
    elif off_mode in ("div", "sigmoid_div"):
        inputs.append(vi("off_in", [1, Q, OFF_FLAT]))
        nodes.append(helper.make_node("Reshape", ["off_in", "shape_off6"], ["off6"]))
        inits.append(const("shape_off6", np.asarray(SHAPE_OFF6, dtype=np.int64)))
        inits.append(const("div8", np.asarray(8.0, dtype=np.float32)))
        nodes.append(helper.make_node("Div", ["off6", "div8"], ["off_div"]))
        cur = "off_div"
        off_out = "off_div"
    elif off_mode == "tp":
        inputs.append(vi("off_in", [1, Q, 2, 8, 3, 4]))
        nodes.append(helper.make_node(
            "Transpose", ["off_in"], ["off_tp"], perm=[0, 1, 3, 4, 5, 2]))
        inits.append(const("div8", np.asarray(8.0, dtype=np.float32)))
        nodes.append(helper.make_node("Div", ["off_tp", "div8"], ["off_div"]))
        cur = "off_div"
        off_out = "off_div"
    elif off_mode == "mm":
        inputs.append(vi("off_in", [1, Q, OFF_FLAT]))
        w = (np.random.default_rng(7).standard_normal(
            (OFF_FLAT, OFF_FLAT)) / np.sqrt(OFF_FLAT)).astype(np.float32)
        inits.append(numpy_helper.from_array(w, "mm_w"))
        nodes.append(helper.make_node("MatMul", ["off_in", "mm_w"], ["off_mm"]))
        nodes.append(helper.make_node("Reshape", ["off_mm", "shape_off6"], ["off6"]))
        inits.append(const("shape_off6", np.asarray(SHAPE_OFF6, dtype=np.int64)))
        inits.append(const("div8", np.asarray(8.0, dtype=np.float32)))
        nodes.append(helper.make_node("Div", ["off6", "div8"], ["off_div"]))
        cur = "off_div"
        off_out = "off_div"
    else:
        raise ValueError(off_mode)

    # ---- reference branch ----
    inputs.append(vi("bbox_in", [1, Q, 4]))
    ref_src = "bbox_in"
    if ref_mode == "sigmoid":
        nodes.append(helper.make_node("Sigmoid", ["bbox_in"], ["bbox_sig"]))
        ref_src = "bbox_sig"
    nodes.append(helper.make_node("Reshape", [ref_src, "shape_ref6"], ["ref6"]))
    inits.append(const("shape_ref6", np.asarray(SHAPE_REF6, dtype=np.int64)))
    inits += [
        const("st_xy", np.asarray([0], dtype=np.int64)),
        const("en_xy", np.asarray([2], dtype=np.int64)),
        const("st_wh", np.asarray([2], dtype=np.int64)),
        const("en_wh", np.asarray([4], dtype=np.int64)),
        const("ax", np.asarray([5], dtype=np.int64)),
    ]
    nodes += [
        helper.make_node("Slice", ["ref6", "st_xy", "en_xy", "ax"], ["xy"]),
        helper.make_node("Slice", ["ref6", "st_wh", "en_wh", "ax"], ["wh"]),
    ]

    # ---- the Multiply under test ----
    nodes += [
        helper.make_node("Mul", [cur, "wh"], ["mul"]),
        helper.make_node("Add", ["xy", "mul"], ["add"]),
        helper.make_node("Mul", ["mul", "half"], ["mul1"]),
    ]
    inits.append(const("half", np.asarray(0.5, dtype=np.float32)))

    outputs = [vi(off_out, SHAPE_OFF6), vi("mul", SHAPE_OFF6),
               vi("mul1", SHAPE_OFF6), vi("add", SHAPE_OFF6)]
    graph = helper.make_graph(nodes, name, inputs, outputs, initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model)
    path = os.path.join(OUT_DIR, name + ".onnx")
    onnx.save(model, path)
    print("wrote", path, "nodes", len(nodes))
    return {
        "model": name + ".onnx",
        "inputs": {i.name: [d.dim_value for d in i.type.tensor_type.shape.dim]
                   for i in inputs},
        "outputs": [o.name for o in outputs],
    }


VARIANTS = [
    ("v0_direct", "direct", "plain"),
    ("v1_div", "div", "plain"),
    ("v2_sigmoid", "direct", "sigmoid"),
    ("v3_both", "sigmoid_div", "sigmoid"),
    ("v4_both_tp", "tp", "sigmoid"),
    ("v5_both_mm", "mm", "sigmoid"),
]


def main():
    meta = []
    for name, off_mode, ref_mode in VARIANTS:
        meta.append(build(name, off_mode, ref_mode))
    with open(os.path.join(OUT_DIR, "variants.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("wrote variants.json")


if __name__ == "__main__":
    main()
