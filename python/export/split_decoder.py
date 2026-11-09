"""Split the exported RT-DETR ONNX into backbone+neck and decoder.

The decoder (`/model.26/*`) consumes exactly three neck feature maps, so the
cut is clean: A = images -> [P3, P4, P5], B = [P3, P4, P5] -> y + aux outputs.
"""
import argparse

import onnx
import onnx.utils


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('--feat', nargs=3, required=True,
                    help='neck feature tensors feeding /model.26')
    ap.add_argument('--out-a', required=True)
    ap.add_argument('--out-b', required=True)
    ap.add_argument('--cast-a-fp16', action='store_true',
                    help='cast the backbone outputs to float16 so the decoder '
                         'can consume them without an int8 dequantize step')
    args = ap.parse_args()

    m = onnx.load(args.src, load_external_data=False)
    outputs = [o.name for o in m.graph.output]
    print('src outputs:', outputs)

    onnx.utils.extract_model(args.src, args.out_a, ['images'], args.feat)
    onnx.utils.extract_model(args.src, args.out_b, args.feat, outputs)

    if args.cast_a_fp16:
        cast_model_outputs(args.out_a, args.feat)

    for path in (args.out_a, args.out_b):
        a = onnx.load(path, load_external_data=False)
        print(f'{path}: {len(a.graph.node)} nodes, inputs '
              f'{[i.name for i in a.graph.input]}, outputs '
              f'{[o.name for o in a.graph.output]}')


def cast_model_outputs(path, names):
    """Append Cast(to=FLOAT16) on the given graph outputs."""
    import onnx
    from onnx import TensorProto, helper

    m = onnx.load(path, load_external_data=False)
    g = m.graph
    for name in names:
        out = f'{name}_fp16'
        g.node.append(helper.make_node('Cast', [name], [out],
                                       to=TensorProto.FLOAT16))
        for o in g.output:
            if o.name == name:
                o.name = out
    onnx.save(m, path)
    print(f'cast to fp16: {len(names)} outputs of {path}')



if __name__ == '__main__':
    main()
