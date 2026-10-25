#!/usr/bin/env python3
"""Inspect Squeeze tensors and their immediate graph neighbours in an ONNX model."""

import argparse
from pathlib import Path

import onnx


def _contains_any(value: str, needles: tuple[str, ...]) -> bool:
    return any(needle.lower() in value.lower() for needle in needles)


def _shape_map(model: onnx.ModelProto) -> dict[str, str]:
    inferred = onnx.shape_inference.infer_shapes(model)
    shapes: dict[str, str] = {}
    for value_info in [
        *inferred.graph.input,
        *inferred.graph.value_info,
        *inferred.graph.output,
    ]:
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims = []
        for dim in tensor_type.shape.dim:
            if dim.dim_value:
                dims.append(str(dim.dim_value))
            elif dim.dim_param:
                dims.append(dim.dim_param)
            else:
                dims.append("?")
        shapes[value_info.name] = f"[{','.join(dims)}]"
    return shapes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--needle",
        action="append",
        default=[],
        help="Additional case-insensitive substring to match; may be repeated.",
    )
    args = parser.parse_args()

    model = onnx.load(str(args.model))
    graph = model.graph
    nodes = list(graph.node)
    needles = tuple(args.needle)
    shapes = _shape_map(model)

    hits = []
    for index, node in enumerate(nodes):
        outputs = list(node.output)
        inputs = list(node.input)
        if (not needles and node.op_type == "Squeeze") or (
            needles and _contains_any(" ".join(outputs), needles)
        ):
            hits.append((index, node))
        elif needles and _contains_any(" ".join(inputs), needles):
            hits.append((index, node))

    print(
        f"model={args.model} ir={model.ir_version} "
        f"opset={[(item.domain or 'ai.onnx', item.version) for item in model.opset_import]}"
    )
    print(f"nodes={len(nodes)} hits={len(hits)} needles={needles}")
    for index, node in hits:
        print(
            f"[{index}] op={node.op_type} name={node.name!r} "
            f"inputs={list(node.input)} outputs={list(node.output)}"
        )
        for tensor_name in [*node.input, *node.output]:
            print(f"  shape {tensor_name!r}: {shapes.get(tensor_name, 'unknown')}")
        start = max(0, index - 2)
        end = min(len(nodes), index + 3)
        for neighbour_index in range(start, end):
            neighbour = nodes[neighbour_index]
            print(
                f"  [{neighbour_index}] {neighbour.op_type} "
                f"{neighbour.name!r} {list(neighbour.input)} -> {list(neighbour.output)}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
