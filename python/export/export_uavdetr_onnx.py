#!/usr/bin/env python3
"""Export UAV-DETR to a static ONNX graph with RKNN-safe operators."""

import argparse
from pathlib import Path


def install_rknn_safe_patches(torch, nn):
    """Replace dynamic GridSample/Linear shapes with static-friendly ops."""

    def custom_grid_sample(
        input,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ):
        n, channels, height_in, width_in = input.shape
        _, height_out, width_out, _ = grid.shape

        ix = grid[..., 0]
        iy = grid[..., 1]
        if align_corners:
            ix = ((ix + 1) / 2) * (width_in - 1)
            iy = ((iy + 1) / 2) * (height_in - 1)
        else:
            ix = ((ix + 1) * width_in - 1) / 2
            iy = ((iy + 1) * height_in - 1) / 2

        ix_nw = torch.floor(ix)
        iy_nw = torch.floor(iy)
        ix_ne = ix_nw + 1
        iy_sw = iy_nw + 1
        ix_se = ix_ne
        iy_se = iy_sw

        nw_weight = (ix_ne - ix) * (iy_sw - iy)
        ne_weight = (ix - ix_nw) * (iy_sw - iy)
        sw_weight = (ix_ne - ix) * (iy - iy_nw)
        se_weight = (ix - ix_nw) * (iy - iy_nw)

        ix_nw = torch.clamp(ix_nw, 0, width_in - 1)
        iy_nw = torch.clamp(iy_nw, 0, height_in - 1)
        ix_ne = torch.clamp(ix_ne, 0, width_in - 1)
        iy_sw = torch.clamp(iy_sw, 0, height_in - 1)
        ix_se = torch.clamp(ix_se, 0, width_in - 1)
        iy_se = torch.clamp(iy_se, 0, height_in - 1)

        idx_nw = (iy_nw * width_in + ix_nw).reshape(n, -1)
        idx_ne = (iy_nw * width_in + ix_ne).reshape(n, -1)
        idx_sw = (iy_sw * width_in + ix_nw).reshape(n, -1)
        idx_se = (iy_se * width_in + ix_se).reshape(n, -1)
        flat_input = input.reshape(n, channels, -1)

        def gather(index):
            expanded = index.unsqueeze(1).expand(-1, channels, -1).long()
            return torch.gather(flat_input, 2, expanded).view(
                n,
                channels,
                height_out,
                width_out,
            )

        return (
            gather(idx_nw) * nw_weight.unsqueeze(1)
            + gather(idx_ne) * ne_weight.unsqueeze(1)
            + gather(idx_sw) * sw_weight.unsqueeze(1)
            + gather(idx_se) * se_weight.unsqueeze(1)
        )

    original_linear_forward = nn.Linear.forward

    def patched_linear_forward(self, input):
        if input.dim() <= 2:
            return original_linear_forward(self, input)
        original_shape = input.shape
        flattened = 1
        for dimension in original_shape[:-1]:
            flattened *= dimension
        output = original_linear_forward(
            self,
            input.reshape(flattened, original_shape[-1]),
        )
        return output.reshape(*original_shape[:-1], self.out_features)

    torch.nn.functional.grid_sample = custom_grid_sample
    nn.Linear.forward = patched_linear_forward


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--num-queries", type=int, default=300)
    parser.add_argument("--decoder-layers", type=int)
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(args.repo.resolve()))
    import torch
    import torch.nn as nn
    from ultralytics import RTDETR

    install_rknn_safe_patches(torch, nn)
    model = RTDETR(str(args.checkpoint.resolve()))
    torch_model = model.model.cpu().eval()
    decoder = torch_model.model[-1]
    if not hasattr(decoder, "num_queries"):
        raise RuntimeError("last model module is not an RT-DETR decoder")
    decoder.num_queries = args.num_queries
    if args.decoder_layers is not None:
        available_layers = len(decoder.decoder.layers)
        if not 1 <= args.decoder_layers <= available_layers:
            raise ValueError(
                f"decoder layers must be in [1, {available_layers}], "
                f"got {args.decoder_layers}"
            )
        decoder.decoder.layers = nn.ModuleList(
            list(decoder.decoder.layers)[:args.decoder_layers]
        )
        decoder.dec_score_head = nn.ModuleList(
            list(decoder.dec_score_head)[:args.decoder_layers]
        )
        decoder.dec_bbox_head = nn.ModuleList(
            list(decoder.dec_bbox_head)[:args.decoder_layers]
        )
        decoder.decoder.num_layers = args.decoder_layers
        decoder.decoder.eval_idx = args.decoder_layers - 1
    if hasattr(torch_model, "head"):
        torch_model.head.export = True
        torch_model.head.format = "onnx"

    dummy_input = torch.randn(1, 3, args.imgsz, args.imgsz)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        torch_model,
        dummy_input,
        str(args.output.resolve()),
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["pred_logits", "pred_boxes"],
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
