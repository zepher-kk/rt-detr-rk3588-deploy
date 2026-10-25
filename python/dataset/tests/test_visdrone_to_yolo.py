"""Unit tests for the VisDrone-to-YOLO converter."""

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

from visdrone_to_yolo import convert_annotation, convert_split


class VisDroneToYoloTest(unittest.TestCase):
    def test_convert_annotation_filters_and_normalizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            annotation = Path(tmp) / "sample.txt"
            annotation.write_text(
                "\n".join(
                    [
                        "10,20,40,60,1,3,0,0",
                        "0,0,10,10,1,0,0,0",
                        "0,0,10,10,0,3,0,0",
                        "5,5,20,20,1,11,0,0",
                        "5,5,20,20,1,1,0,0,",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            rows = convert_annotation(annotation, (100, 200))

        self.assertEqual(
            rows,
            [
                "2 0.300000 0.250000 0.400000 0.300000",
                "0 0.150000 0.075000 0.200000 0.100000",
            ],
        )

    def test_convert_split_writes_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "visdrone"
            image_dir = root / "VisDrone2019-DET-val" / "images"
            annotation_dir = root / "VisDrone2019-DET-val" / "annotations"
            image_dir.mkdir(parents=True)
            annotation_dir.mkdir(parents=True)
            Image.new("RGB", (100, 50)).save(image_dir / "frame.jpg")
            (annotation_dir / "frame.txt").write_text(
                "10,10,20,10,1,1,0,0\n",
                encoding="utf-8",
            )

            count = convert_split(
                root,
                Path(tmp) / "output",
                "val",
                limit=None,
                overwrite=False,
            )
            label = (Path(tmp) / "output" / "labels" / "val" / "frame.txt").read_text(
                encoding="utf-8"
            )

        self.assertEqual(count, 1)
        self.assertEqual(label, "0 0.200000 0.300000 0.200000 0.200000\n")


if __name__ == "__main__":
    unittest.main()
