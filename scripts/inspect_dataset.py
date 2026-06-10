"""Summarize the Roboflow YOLO dataset (counts per class, per split).

Usage (from the repo root):

    python scripts/inspect_dataset.py                  # raw summary
    python scripts/inspect_dataset.py --regenerate-csv # rewrite configs/all_class_names.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from irrigation_symbol_recognition.data import YoloDataset  # noqa: E402
from irrigation_symbol_recognition.utils import (  # noqa: E402
    load_dataset_config,
    resolve_path,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("inspect_dataset")


def _print_split_summary(name: str, ds: YoloDataset) -> None:
    counts = ds.class_counts(name)
    total = sum(counts.values())
    used = sum(1 for v in counts.values() if v > 0)
    print(f"\n=== {name} ===")
    print(f"  images:                 {len(ds.split_images(name))}")
    print(f"  annotations:            {total}")
    print(f"  classes with >=1 anno:  {used} / {len(ds.class_names)}")
    print("  top 10 classes:")
    for cid, n in counts.most_common(10):
        print(f"    yolo_id={cid:>4}  name={ds.class_names[cid]:<12}  count={n}")


def _regenerate_csv(ds: YoloDataset, out_path: Path) -> None:
    train_cnt = ds.class_counts("train") if "train" in ds.split_dirs else {}
    test_cnt = ds.class_counts("test") if "test" in ds.split_dirs else {}
    rows = []
    for i, name in enumerate(ds.class_names):
        rows.append({
            "yolo_id": i,
            "name": name,
            "train_count": train_cnt.get(i, 0),
            "test_count": test_cnt.get(i, 0),
            "total_count": train_cnt.get(i, 0) + test_cnt.get(i, 0),
        })
    rows.sort(key=lambda r: (-r["total_count"], r["name"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["yolo_id", "name", "train_count", "test_count", "total_count"])
        w.writeheader()
        w.writerows(rows)
    logger.info("Wrote %d rows to %s", len(rows), out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-config",
        default=str(REPO_ROOT / "configs" / "dataset.yaml"),
    )
    parser.add_argument(
        "--regenerate-csv",
        action="store_true",
        help="Rewrite configs/all_class_names.csv from the on-disk YOLO labels.",
    )
    args = parser.parse_args()

    ds_cfg = load_dataset_config(args.dataset_config)
    root = resolve_path(ds_cfg["dataset"]["root"])
    ds = YoloDataset.load(root)

    print(f"YOLO dataset root: {root}")
    print(f"  nc:     {len(ds.class_names)}")
    print(f"  splits: {sorted(ds.split_dirs)}")

    four_digit = sum(1 for n in ds.class_names if re.fullmatch(r"\d{4}", n))
    print(f"  4-digit class names: {four_digit} / {len(ds.class_names)} ({four_digit/len(ds.class_names):.1%})")

    for split in ("train", "valid", "test"):
        if split in ds.split_dirs:
            _print_split_summary(split, ds)

    if args.regenerate_csv:
        _regenerate_csv(ds, REPO_ROOT / "configs" / "all_class_names.csv")

    print(
        "\n(Tip) Fill in `configs/irrigation_classes.yaml`, then run "
        "`python scripts/prepare_detection_dataset.py` to filter + slice."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
