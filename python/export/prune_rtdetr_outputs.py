#!/usr/bin/env python3
"""Prune RT-DETR's redundant tail and expose only final boxes and scores."""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import onnx


def _producer(graph: onnx.GraphProto, tensor_name: str) -> onnx.NodeProto:
    for node in graph.node:
        if tensor_name in node.output:
            return node
    raise ValueError(f"no producer for {tensor_name}")


def _shape_map(model: onnx.ModelProto) -> dict[str, tuple[int, ...]]:
    inferred = onnx.shape_inference.infer_shapes(model)
    result: dict[str, tuple[int, ...]] = {}
    for value_info in [
        *inferred.graph.input,
        *inferred.graph.value_info,
        *inferred.graph.output,
    ]:
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims = tuple(dim.dim_value for dim in tensor_type.shape.dim)
        if all(dims):
            result[value_info.name] = dims
    return result


def _verify(original_path: Path, pruned_path: Path) -> None:
    import onnxruntime as ort

    rng = np.random.default_rng(20261025)
    inputs = rng.standard_normal((1, 3, 640, 640), dtype=np.float32)

    original = ort.InferenceSession(
        str(original_path), providers=["CPUExecutionProvider"]
    )
    reference = original.run(
        ["pred_boxes", "pred_logits"], {"images": inputs}
    )
    del original

    pruned = ort.InferenceSession(
        str(pruned_path), providers=["CPUExecutionProvider"]
    )
    actual_outputs = pruned.run(None, {"images": inputs})
    actual_names = [item.name for item in pruned.get_outputs()]
    if actual_names != ["boxes_final", "scores_final"]:
        raise RuntimeError(f"unexpected output names: {actual_names}")

    ref_boxes, ref_logits = reference
    while ref_boxes.ndim > 3 and ref_boxes.shape[1] == 1:
        ref_boxes = np.squeeze(ref_boxes, axis=1)
    while ref_logits.ndim > 3 and ref_logits.shape[1] == 1:
        ref_logits = np.squeeze(ref_logits, axis=1)
    if tuple(ref_boxes.shape) != (1, 300, 4):
        raise RuntimeError(f"unexpected reference box shape: {ref_boxes.shape}")
    if tuple(ref_logits.shape) != (1, 300, 14):
        raise RuntimeError(f"unexpected reference logits shape: {ref_logits.shape}")
    if not np.allclose(ref_boxes, ref_logits[..., :4], rtol=1e-5, atol=1e-5):
        raise RuntimeError("reference pred_boxes differs from concatenated prefix")

    for name, expected, observed in zip(
        ("boxes_final", "scores_final"),
        (ref_logits[..., :4], ref_logits[..., 4:]),
        actual_outputs,
    ):
        max_abs = float(np.max(np.abs(observed - expected)))
        print(
            f"verify {name}: shape={observed.shape} max_abs={max_abs:.7g}",
            flush=True,
        )
        if not np.allclose(observed, expected, rtol=1e-5, atol=1e-5):
            raise RuntimeError(f"{name} mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--preserve-nodes", action="store_true")
    args = parser.parse_args()

    model = onnx.load(str(args.input))
    graph = model.graph
    output_names = [item.name for item in graph.output]
    required_outputs = {"pred_boxes", "pred_logits"}
    if not required_outputs.issubset(output_names):
        raise RuntimeError(f"unexpected input graph outputs: {output_names}")

    boxes_producer = _producer(graph, "pred_boxes")
    if boxes_producer.op_type == "Concat" and len(boxes_producer.input) == 1:
        boxes_producer = _producer(graph, boxes_producer.input[0])
    if boxes_producer.op_type != "Unsqueeze" or len(boxes_producer.input) != 1:
        raise RuntimeError(
            f"unexpected pred_boxes producer chain: {boxes_producer.op_type}"
        )
    boxes_source = boxes_producer.input[0]

    logits_concat = _producer(graph, "pred_logits")
    if logits_concat.op_type != "Concat" or not logits_concat.input:
        raise RuntimeError("unexpected pred_logits producer")
    scores_source = logits_concat.input[-1]
    scores_sigmoid = _producer(graph, scores_source)
    if scores_sigmoid.op_type != "Sigmoid":
        raise RuntimeError("final score tensor is not Sigmoid output")

    shapes = _shape_map(model)
    expected_shapes = {
        boxes_source: (1, 300, 4),
        scores_source: (1, 300, 10),
    }
    for tensor_name, expected in expected_shapes.items():
        observed = shapes.get(tensor_name)
        if observed != expected:
            raise RuntimeError(
                f"{tensor_name}: shape {observed} != {expected}"
            )

    graph.node.extend(
        [
            onnx.helper.make_node(
                "Identity", [boxes_source], ["boxes_final"], "SplitBoxesOutput"
            ),
            onnx.helper.make_node(
                "Identity", [scores_source], ["scores_final"], "SplitScoresOutput"
            ),
        ]
    )

    if args.preserve_nodes:
        del graph.output[:]
        graph.output.extend(
            [
                onnx.helper.make_tensor_value_info(
                    "boxes_final", onnx.TensorProto.FLOAT, [1, 300, 4]
                ),
                onnx.helper.make_tensor_value_info(
                    "scores_final", onnx.TensorProto.FLOAT, [1, 300, 10]
                ),
            ]
        )
        pruned = model
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            augmented = Path(temp_dir) / "augmented.onnx"
            onnx.save_model(model, str(augmented))
            onnx.utils.extract_model(
                str(augmented),
                str(args.output),
                ["images"],
                ["boxes_final", "scores_final"],
            )
        pruned = onnx.load(str(args.output))

    pruned.graph.name = "rtdetr_split_outputs"
    pruned.doc_string = (
        "RT-DETR split output graph: boxes [1,300,4], scores [1,300,10]"
    )
    onnx.checker.check_model(pruned)
    onnx.save_model(pruned, str(args.output))
    print(
        f"pruned {args.input} -> {args.output}: "
        f"nodes={len(pruned.graph.node)} size={args.output.stat().st_size}",
        flush=True,
    )

    if args.verify:
        _verify(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
