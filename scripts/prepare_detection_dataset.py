"""End-to-end YOLO dataset preparation for irrigation symbols.

Steps performed (driven by `configs/dataset.yaml` + `configs/irrigation_classes.yaml`
+ `configs/detection.yaml`):

    1. Load the Roboflow YOLO export from `dataset.yaml -> dataset.root`.
    2. Apply the irrigation filter from `irrigation_classes.yaml`
       (class subset + class_groups + non_irrigation_policy).
    3. Carve a page-prefix-grouped `valid` split off `train`.
    4. Write a filtered YOLO dataset (images + per-image .txt labels +
       a fresh data.yaml) at `paths.yolo_dataset_dir`.
    5. Slice every (image, label) pair into 1280x1280 (configurable) tiles
       with overlap, applying the bounding-box-near-cut rule from
       `slicing.py` (the same technique used by the reference pipeline).
       Output goes to `paths.sliced_dataset_dir`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from irrigation_symbol_recognition.data import YoloDataset  # noqa: E402
from irrigation_symbol_recognition.detection.yolo_filter import (  # noqa: E402
    filter_yolo_dataset,
)
from irrigation_symbol_recognition.detection.slicing import slice_data  # noqa: E402
from irrigation_symbol_recognition.utils.config import (  # noqa: E402
    load_dataset_config,
    load_irrigation_config,
    resolve_path,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prepare_detection_dataset")


def _load_detection_config(path: Path) -> dict:
    import yaml

    with path.open("r") as fh:
        return yaml.safe_load(fh) or {}


def _short_label_list(names: list[str], n: int = 20) -> list[str]:
    return names if len(names) <= n else names[:n] + ["..."]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", default=str(REPO_ROOT / "configs" / "dataset.yaml"))
    parser.add_argument("--irrigation-config", default=str(REPO_ROOT / "configs" / "irrigation_classes.yaml"))
    parser.add_argument("--detection-config", default=str(REPO_ROOT / "configs" / "detection.yaml"))
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images instead of symlinking (slower, doubles disk usage).",
    )
    parser.add_argument(
        "--skip-slicing",
        action="store_true",
        help="Only produce the unsliced filtered YOLO dataset; skip the tiling step.",
    )
    args = parser.parse_args()

    ds_cfg = load_dataset_config(args.dataset_config)
    irr_cfg = load_irrigation_config(args.irrigation_config)
    det_cfg = _load_detection_config(Path(args.detection_config))

    if not irr_cfg.irrigation_classes:
        logger.error(
            "irrigation_classes is empty in %s. Fill it in first.", args.irrigation_config
        )
        return 1

    source_root = resolve_path(ds_cfg["dataset"]["root"])
    source_ds = YoloDataset.load(source_root)
    logger.info(
        "Source YOLO export: %s (nc=%d, splits=%s)",
        source_root, len(source_ds.class_names), sorted(source_ds.split_dirs),
    )

    split_cfg = det_cfg["split"]
    paths_cfg = det_cfg["paths"]
    slicing_cfg = det_cfg["slicing"]

    yolo_out = resolve_path(paths_cfg["yolo_dataset_dir"])
    sliced_out = resolve_path(paths_cfg["sliced_dataset_dir"])

    logger.info("Filtering to irrigation classes -> %s", yolo_out)
    summary = filter_yolo_dataset(
        source_ds=source_ds,
        irrigation_cfg=irr_cfg,
        out_dir=yolo_out,
        valid_fraction=float(split_cfg.get("valid_fraction", 0.10)),
        seed=int(split_cfg.get("seed", 42)),
        group_by_page_prefix=bool(split_cfg.get("group_by_page_prefix", True)),
        symlink_images=not args.copy_images,
    )

    report = {
        "num_classes": summary["num_classes"],
        "label_names": _short_label_list(summary["label_names"]),
        "missing_class_names": summary["missing_class_names"],
        "data_yaml": summary["data_yaml"],
        "train": {k: v for k, v in summary["train"].items() if k != "annotations_per_dst_id"},
        "valid": {k: v for k, v in summary["valid"].items() if k != "annotations_per_dst_id"},
        "test":  {k: v for k, v in summary["test"].items()  if k != "annotations_per_dst_id"},
    }
    logger.info("Filter summary:\n%s", json.dumps(report, indent=2))

    if args.skip_slicing:
        logger.info("--skip-slicing was set; done.")
        return 0

    slice_data({
        "dataset_dir": str(yolo_out),
        "output_dir": str(sliced_out),
        "image_size": int(slicing_cfg["slice_size"]),
        "overlap": float(slicing_cfg["overlap"]),
        "fit_box_to_slice": bool(slicing_cfg.get("fit_box_to_slice", False)),
    })

    logger.info("Done. Sliced YOLO dataset is at: %s", sliced_out)
    logger.info("Train YOLO with:\n  python scripts/train_detection.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
