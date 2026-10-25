#!/usr/bin/env python3
"""Convert VisDrone detection annotations to normalized YOLO labels."""

import argparse
from pathlib import Path

from PIL import Image


IGNORED_CATEGORIES = {0, 11}


def convert_annotation(
    annotation: Path,
    image_size: tuple[int, int],
) -> list[str]:
    """Return YOLO label rows for one VisDrone annotation file."""
    image_width, image_height = image_size
    rows: list[str] = []

    for line_number, raw_line in enumerate(
        annotation.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        # VisDrone contains a small number of files with one trailing comma.
        normalized_line = line[:-1] if line.endswith(",") else line
        fields = [field.strip() for field in normalized_line.split(",")]
        if len(fields) != 8:
            raise ValueError(
                f"{annotation}:{line_number}: expected 8 fields, got {len(fields)}"
            )

        left, top, width, height = map(float, fields[:4])
        score = int(fields[4])
        category = int(fields[5])
        if score == 0 or category in IGNORED_CATEGORIES:
            continue
        if not 1 <= category <= 10:
            raise ValueError(
                f"{annotation}:{line_number}: unsupported category {category}"
            )

        x1 = max(0.0, min(left, image_width))
        y1 = max(0.0, min(top, image_height))
        x2 = max(x1, min(left + width, image_width))
        y2 = max(y1, min(top + height, image_height))
        box_width = x2 - x1
        box_height = y2 - y1
        if box_width <= 0.0 or box_height <= 0.0:
            continue

        center_x = (x1 + x2) / (2.0 * image_width)
        center_y = (y1 + y2) / (2.0 * image_height)
        normalized_width = box_width / image_width
        normalized_height = box_height / image_height
        rows.append(
            f"{category - 1} {center_x:.6f} {center_y:.6f} "
            f"{normalized_width:.6f} {normalized_height:.6f}"
        )

    return rows


def convert_split(
    root: Path,
    output: Path,
    split: str,
    limit: int | None,
    overwrite: bool,
) -> int:
    """Convert one VisDrone split and return the number of label files."""
    image_dir = root / f"VisDrone2019-DET-{split}" / "images"
    annotation_dir = root / f"VisDrone2019-DET-{split}" / "annotations"
    output_dir = output / "labels" / split
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{output_dir} is not empty; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations = sorted(annotation_dir.glob("*.txt"))
    if limit is not None:
        annotations = annotations[:limit]
    if not annotations:
        raise FileNotFoundError(f"no annotations found in {annotation_dir}")

    converted = 0
    for annotation in annotations:
        image_path = image_dir / f"{annotation.stem}.jpg"
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as image:
            rows = convert_annotation(annotation, image.size)
        (output_dir / f"{annotation.stem}.txt").write_text(
            "\n".join(rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )
        converted += 1
    return converted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for split in args.splits:
        count = convert_split(
            args.root,
            args.output,
            split,
            args.limit,
            args.overwrite,
        )
        print(f"{split}: {count} label files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
