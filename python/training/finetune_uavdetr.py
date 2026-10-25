#!/usr/bin/env python3
"""Run a bounded UAV-DETR fine-tune from an existing checkpoint."""

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--lr0", type=float, default=1e-5)
    parser.add_argument("--freeze", type=int, default=10)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(args.repo.resolve()))
    import torch
    from ultralytics import RTDETR

    torch.cuda.empty_cache()
    model = RTDETR(str(args.checkpoint.resolve()))
    model.train(
        data=str(args.data.resolve()),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        fraction=args.fraction,
        workers=args.workers,
        device=args.device,
        lr0=args.lr0,
        lrf=0.1,
        cos_lr=True,
        warmup_epochs=0.1,
        freeze=args.freeze,
        amp=args.amp,
        cache=False,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=0,
        val=False,
        plots=False,
        save=True,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=True,
        seed=0,
        deterministic=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
