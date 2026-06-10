"""Helpers for reading a Roboflow-style YOLO dataset.

Expected on-disk layout (mirrors what Roboflow's ``YOLO*`` export produces):

    <root>/
    +-- data.yaml         # `nc`, `names`, train/val/test paths
    +-- train/
    |   +-- images/
    |   +-- labels/       # one .txt per image, lines: "cid xc yc w h"
    +-- valid/  (optional)
    +-- test/   (optional)

The Roboflow ``data.yaml`` we receive uses paths like ``../train/images``;
we ignore those and rely on the on-disk directory layout instead.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml


_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass
class YoloDataset:
    """A read-only view of a YOLO dataset on disk."""

    root: Path
    class_names: list[str]
    split_dirs: dict[str, Path]

    @classmethod
    def load(cls, root: str | Path) -> "YoloDataset":
        root = Path(root)
        data_yaml = root / "data.yaml"
        if not data_yaml.exists():
            raise FileNotFoundError(f"data.yaml not found in {root}")
        with data_yaml.open("r") as fh:
            data = yaml.safe_load(fh) or {}

        names = data.get("names")
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names.keys(), key=int)]
        if not isinstance(names, list):
            raise ValueError(f"data.yaml['names'] is missing or malformed: {data_yaml}")
        class_names = [str(n) for n in names]

        split_dirs: dict[str, Path] = {}
        for split in ("train", "valid", "val", "test"):
            cand = root / split
            if (cand / "images").is_dir() and (cand / "labels").is_dir():
                key = "valid" if split == "val" else split
                split_dirs[key] = cand
        if not split_dirs:
            raise ValueError(
                f"No usable splits under {root}. Expected at least one of "
                "'train', 'valid', 'val', 'test' with 'images' + 'labels' subdirs."
            )

        return cls(root=root, class_names=class_names, split_dirs=split_dirs)

    def split_images(self, split: str) -> list[Path]:
        d = self.split_dirs[split] / "images"
        return sorted(
            p for p in d.iterdir() if p.suffix.lower() in _IMAGE_EXTS
        )

    def label_for(self, image_path: Path) -> Path:
        # image_path: .../<split>/images/<name>.jpg
        return image_path.parent.parent / "labels" / (image_path.stem + ".txt")

    def iter_labels(self, split: str):
        """Yield ``(image_path, label_path, parsed_rows)`` for every image."""
        for image_path in self.split_images(split):
            label_path = self.label_for(image_path)
            rows: list[tuple[int, float, float, float, float]] = []
            if label_path.exists():
                with label_path.open("r") as fh:
                    for line in fh:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        rows.append((
                            int(parts[0]),
                            float(parts[1]),
                            float(parts[2]),
                            float(parts[3]),
                            float(parts[4]),
                        ))
            yield image_path, label_path, rows

    def class_counts(self, split: str) -> Counter:
        """Annotation counts per source class ID in the given split."""
        cnt: Counter = Counter()
        for _, _, rows in self.iter_labels(split):
            cnt.update(r[0] for r in rows)
        return cnt
