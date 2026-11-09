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
import torch.nn as nn
import torch.nn.functional as F

DETR_SRC = os.environ.get('UAVDETR_SRC', '/mnt/d/learn/detr/UAV-DETR-rt')
if os.path.exists(DETR_SRC):
    sys.path.insert(0, DETR_SRC)

from ultralytics import RTDETR  # noqa: E402
import ultralytics.nn.modules.transformer as tf  # noqa: E402


_GRID_4D = False


def _grid_from_locations(sampling_locations):
    """grid = 2 * locations - 1.

    With `_GRID_4D` the arithmetic is done on a 4-D view
    (bs, queries, heads*levels*points, 2): RKNN 2.5 miscompiles the 6-D
    broadcast form and returns zeros for the Mul that produces the locations.
    """
    if not _GRID_4D:
        return 2 * sampling_locations - 1
    bs, queries, heads, levels, points, _ = sampling_locations.shape
    flat = sampling_locations.reshape(bs, queries, heads * levels * points, 2)
    return (2 * flat - 1).reshape(bs, queries, heads, levels, points, 2)


def msda_v3(value, value_spatial_shapes, sampling_locations, attention_weights):
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, _, _, num_points, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = _grid_from_locations(sampling_locations)

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
    sampling_grids = _grid_from_locations(sampling_locations)

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


def msda_forward_folded(self, query, refer_bbox, value, value_shapes, value_mask=None):
    """MSDeformAttn.forward with the constant 0.5/n_points folded into
    sampling_offsets weights (caller must pre-scale them by 0.125)."""
    bs, len_q = query.shape[:2]
    len_v = value.shape[1]
    value = self.value_proj(value)
    if value_mask is not None:
        value = value.masked_fill(value_mask[..., None], float(0))
    value = value.view(bs, len_v, self.n_heads, self.d_model // self.n_heads)
    sampling_offsets = self.sampling_offsets(query).view(
        bs, len_q, self.n_heads, self.n_levels, self.n_points, 2)
    attention_weights = self.attention_weights(query).view(
        bs, len_q, self.n_heads, self.n_levels * self.n_points)
    attention_weights = F.softmax(attention_weights, -1).view(
        bs, len_q, self.n_heads, self.n_levels, self.n_points)
    num_points = refer_bbox.shape[-1]
    if num_points == 2:
        offset_normalizer = torch.as_tensor(
            value_shapes, dtype=query.dtype, device=query.device).flip(-1)
        add = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        sampling_locations = refer_bbox[:, :, None, :, None, :] + add
    elif num_points == 4:
        if _GRID_4D:
            # 4-D form: RKNN 2.5 returns zeros for the 6-D broadcast Mul
            # (bs,q,H,L,P,2) x (bs,q,1,1,1,2)
            bs_, q_ = sampling_offsets.shape[:2]
            h, l, p = self.n_heads, self.n_levels, self.n_points
            off4 = sampling_offsets.reshape(bs_, q_, h * l * p, 2)
            ref4 = refer_bbox.reshape(bs_, q_, 1, 4)
            loc4 = ref4[..., :2] + off4 * ref4[..., 2:]
            sampling_locations = loc4.reshape(bs_, q_, h, l, p, 2)
        else:
            add = sampling_offsets * refer_bbox[:, :, None, :, None, 2:]
            sampling_locations = refer_bbox[:, :, None, :, None, :2] + add
    else:
        raise ValueError(f'Last dim of reference_points must be 2 or 4, but got {num_points}.')
    output = tf.multi_scale_deformable_attn_pytorch(
        value, value_shapes, sampling_locations, attention_weights)
    return self.output_proj(output)


def fold_offset_scaling(model, n_points=4):
    """Scale sampling_offsets weights by 0.5 / n_points so the forward can skip
    the constant Div/Mul."""
    factor = 0.5 / n_points
    decoder = model.model[-1]
    with torch.no_grad():
        for layer in decoder.decoder.layers:
            ca = layer.cross_attn
            ca.sampling_offsets.weight.mul_(factor)
            ca.sampling_offsets.bias.mul_(factor)
    return factor


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
    ap.add_argument('--num-queries', type=int, default=300,
                    help='decoder queries kept (topk over encoder tokens)')
    ap.add_argument('--decoder-layers', type=int, default=None,
                    help='truncate the decoder to the first N layers')
    ap.add_argument('--fold-offset-scale', action='store_true',
                    help='fold the constant 0.5/n_points into sampling_offsets weights')
    ap.add_argument('--msda-4d-grid', action='store_true',
                    help='do the sampling-location / grid arithmetic in 4-D '
                         '(RKNN returns zeros for the 6-D broadcast Mul)')
    args = ap.parse_args()

    global _GRID_4D
    _GRID_4D = args.msda_4d_grid
    if not _GRID_4D:
        print('WARNING: RKNN 2.5 returns zeros for the 6-D broadcast Mul that\n'
              '         computes sampling_locations, so `Mul` must stay on CPU\n'
              '         (op_target GridSample=gpu,Mul=cpu), or re-export with\n'
              '         --msda-4d-grid to keep Mul on NPU.')
    patch_linear()
    if args.fold_offset_scale:
        tf.MSDeformAttn.forward = msda_forward_folded
        print('patched MSDeformAttn.forward -> folded offset scaling')
    attn = msda_v4 if args.value_nchw else msda_v3
    tf.multi_scale_deformable_attn_pytorch = attn
    print('patched multi_scale_deformable_attn_pytorch ->', attn.__name__)

    model = RTDETR(args.weights)
    torch_model = model.model.cpu().eval()

    if args.fold_offset_scale:
        factor = fold_offset_scaling(torch_model)
        print(f'folded 0.5/n_points = {factor} into sampling_offsets')

    # v4.1: prune queries / decoder layers (same approach as v3 export)
    decoder = torch_model.model[-1]
    if not hasattr(decoder, 'num_queries'):
        raise SystemExit('last module is not an RT-DETR decoder')
    decoder.num_queries = args.num_queries
    print(f'num_queries -> {args.num_queries}')

    if args.decoder_layers is not None:
        n_layers = len(decoder.decoder.layers)
        if not 1 <= args.decoder_layers <= n_layers:
            raise SystemExit(f'decoder layers must be in [1, {n_layers}]')
        decoder.decoder.layers = nn.ModuleList(
            list(decoder.decoder.layers)[:args.decoder_layers])
        decoder.dec_score_head = nn.ModuleList(
            list(decoder.dec_score_head)[:args.decoder_layers])
        decoder.dec_bbox_head = nn.ModuleList(
            list(decoder.dec_bbox_head)[:args.decoder_layers])
        decoder.decoder.num_layers = args.decoder_layers
        decoder.decoder.eval_idx = args.decoder_layers - 1
        print(f'decoder layers -> {args.decoder_layers} (of {n_layers})')

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
