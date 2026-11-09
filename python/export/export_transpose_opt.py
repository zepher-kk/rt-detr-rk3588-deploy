"""Export RT-DETR with a transpose-friendly deformable attention.

The stock `multi_scale_deformable_attn_pytorch` builds a 5D tensor
`stack(sampling_value_list).flatten(-2)` -> (bs*heads, dim, queries, levels*points)
and multiplies it by the (bs*heads, 1, queries, levels*points) weights.
On RK3588 that introduces a `(8,32,300,12) -> (8,12,300,32)` Transpose
(perm [0,3,2,1]) which violates the NPU shape constraint (h%8 != 0) and costs
~5.96 ms per decoder layer.

This variant accumulates level by level, so the 12-wide tensor never exists.
Math is identical up to floating point association.

Usage:
  python export_transpose_opt.py --output model/rtdetr_native_tpopt.onnx
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

DETR_SRC = os.environ.get('UAVDETR_SRC', '/mnt/d/learn/detr/UAV-DETR-rt')
if os.path.exists(DETR_SRC):
    sys.path.insert(0, DETR_SRC)

from ultralytics import RTDETR  # noqa: E402
import ultralytics.nn.modules.transformer as tf  # noqa: E402


def msda_v3(value, value_spatial_shapes, sampling_locations, attention_weights):
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, _, _, num_points, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1

    out = None
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        value_l = value_list[level].flatten(2).transpose(1, 2).reshape(
            bs * num_heads, embed_dims, H_, W_)
        grid_l = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampled = F.grid_sample(value_l, grid_l, mode='bilinear',
                                padding_mode='zeros', align_corners=False)
        w = attention_weights[:, :, :, level].transpose(1, 2).reshape(
            bs * num_heads, 1, num_queries, num_points)
        term = (sampled * w).sum(-1)
        out = term if out is None else out + term

    return out.reshape(bs, num_heads * embed_dims, num_queries).transpose(1, 2).contiguous()


def msda_v4(value, value_spatial_shapes, sampling_locations, attention_weights):
    """Like msda_v3, but keeps `value` channel-first and slices it along the
    sequence axis, so the compiler does not need an NCHW->sequence Transpose
    after value_proj."""
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, _, _, num_points, _ = sampling_locations.shape
    # (bs, sumHW, heads, dim) -> (bs, heads, dim, sumHW)
    v = value.permute(0, 2, 3, 1)
    sampling_grids = 2 * sampling_locations - 1

    out = None
    start = 0
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        n = int(H_ * W_)
        v_l = v[..., start:start + n].reshape(
            bs * num_heads, embed_dims, int(H_), int(W_))
        start += n
        grid_l = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampled = F.grid_sample(v_l, grid_l, mode='bilinear',
                                padding_mode='zeros', align_corners=False)
        w = attention_weights[:, :, :, level].transpose(1, 2).reshape(
            bs * num_heads, 1, num_queries, num_points)
        term = (sampled * w).sum(-1)
        out = term if out is None else out + term

    return out.reshape(bs, num_heads * embed_dims, num_queries).transpose(1, 2).contiguous()


def patch_linear():
    original = torch.nn.Linear.forward

    def forward(self, x):
        if x.dim() > 2:
            shape = x.shape
            flat = 1
            for s in shape[:-1]:
                flat *= s
            y = original(self, x.reshape(flat, shape[-1]))
            return y.reshape(list(shape[:-1]) + [self.out_features])
        return original(self, x)

    torch.nn.Linear.forward = forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='model/best.pt')
    ap.add_argument('--output', default='model/rtdetr_native_tpopt.onnx')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--opset', type=int, default=16)
    ap.add_argument('--value-nchw', action='store_true',
                    help='also keep value channel-first in the attention (msda_v4)')
    args = ap.parse_args()

    patch_linear()
    attn = msda_v4 if args.value_nchw else msda_v3
    tf.multi_scale_deformable_attn_pytorch = attn
    print('patched multi_scale_deformable_attn_pytorch ->', attn.__name__)

    model = RTDETR(args.weights)
    torch_model = model.model.cpu().eval()
    if hasattr(torch_model, 'head'):
        torch_model.head.export = True
        torch_model.head.format = 'onnx'

    dummy = torch.randn(1, 3, args.imgsz, args.imgsz)
    torch.onnx.export(
        torch_model, dummy, args.output,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=['images'],
        output_names=['pred_logits', 'pred_boxes'],
    )
    print('exported', args.output)


if __name__ == '__main__':
    main()
