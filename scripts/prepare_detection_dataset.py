"""End-to-end YOLO dataset preparation for irrigation symbols.

Dispatches between two split modes (set via ``configs/detection.yaml -> split.mode``):

* ``page_grouped`` (v1 behaviour):
    1. Load the Roboflow YOLO export.
    2. Apply the irrigation class filter (subset + groups + non-irrigation policy).
    3. Carve a page-prefix-grouped valid split off train.
    4. Write a filtered YOLO dataset + fresh data.yaml.
    5. Slice every (image, label) pair into 1280x1280 tiles with overlap.

* ``stratified_tile`` (v2 behaviour):
    1. Pool Roboflow train + test source images.
    2. Group by page_key, apply class filter / remap in memory.
    3. Greedy class-coverage Set Cover -> drawing-level deployment-test set.
    4. Slice the rest into a scratch directory, capture per-tile metadata.
    5. Build the per-page overlap graph; connected components = atoms.
    6. Stratified assignment of atoms to train/valid/test with per-class
       atom floors so every irrigation class appears in both valid and test.
    7. Materialise into train/valid/test directories + deployment_test/.
    8. Verify the no-leakage invariants before returning.
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
from irrigation_symbol_recognition.detection.stratified_prepare import (  # noqa: E402
    StratifiedPrepareConfig,
    prepare_stratified,
)
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


def _short_class_counts(d: dict, n: int = 20) -> dict:
    items = list(d.items())[:n]
    out = dict(items)
    if len(d) > n:
        out["..."] = f"({len(d) - n} more)"
    return out


def _run_page_grouped(
    *, source_ds, irr_cfg, det_cfg, ds_cfg, args,
) -> int:
    split_cfg = det_cfg["split"]
    paths_cfg = det_cfg["paths"]
    slicing_cfg = det_cfg["slicing"]

    yolo_out = resolve_path(paths_cfg["yolo_dataset_dir"])
    sliced_out = resolve_path(paths_cfg["sliced_dataset_dir"])
    always_drop = list(ds_cfg["dataset"].get("always_drop_categories", []))

    logger.info("[page_grouped] Filtering to irrigation classes -> %s", yolo_out)
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
    return 0


def _run_stratified_tile(
    *, source_ds, irr_cfg, det_cfg, args,
) -> int:
    split_cfg = det_cfg["split"]
    paths_cfg = det_cfg["paths"]
    slicing_cfg = det_cfg["slicing"]
    stratified = split_cfg["stratified_tile"]
    drawing_test = split_cfg.get("drawing_test", {})

    sliced_out = resolve_path(paths_cfg["sliced_dataset_dir"])
    drawing_out = resolve_path(paths_cfg["drawing_test_dir"])

    cfg = StratifiedPrepareConfig(
        slice_size=int(slicing_cfg["slice_size"]),
        overlap=float(slicing_cfg["overlap"]),
        fit_box_to_slice=bool(slicing_cfg.get("fit_box_to_slice", False)),
        train_fraction=float(stratified["train_fraction"]),
        valid_fraction=float(stratified["valid_fraction"]),
        test_fraction=float(stratified["test_fraction"]),
        min_test_atoms_per_class=int(stratified["min_test_atoms_per_class"]),
        min_valid_atoms_per_class=int(stratified["min_valid_atoms_per_class"]),
        deployment_test_max_clusters=(
            int(drawing_test["max_clusters"])
            if drawing_test.get("max_clusters") not in (None, "null", "")
            else None
        ),
        deployment_test_min_clusters_per_class=int(
            drawing_test.get("min_clusters_per_class", 1)
        ),
        seed=int(split_cfg.get("seed", 42)),
        symlink_images=not args.copy_images,
    )

    logger.info("[stratified_tile] Preparing dataset")
    logger.info("  sliced output : %s", sliced_out)
    logger.info("  drawing-test  : %s", drawing_out)
    logger.info("  slice_size=%d  overlap=%.2f  fit_box_to_slice=%s",
                cfg.slice_size, cfg.overlap, cfg.fit_box_to_slice)
    logger.info("  fractions: train=%.2f valid=%.2f test=%.2f",
                cfg.train_fraction, cfg.valid_fraction, cfg.test_fraction)
    logger.info("  per-class floors: test>=%d  valid>=%d",
                cfg.min_test_atoms_per_class, cfg.min_valid_atoms_per_class)
    logger.info("  drawing_test min/max clusters per class: %d / %s",
                cfg.deployment_test_min_clusters_per_class,
                cfg.deployment_test_max_clusters)

    summary = prepare_stratified(
        source_ds=source_ds,
        irrigation_cfg=irr_cfg,
        out_dir=sliced_out,
        deployment_out_dir=drawing_out,
        cfg=cfg,
    )

    # Pretty-print a digest
    digest = {
        "label_names": summary["label_names"],
        "missing_class_names": summary["missing_class_names"],
        "atoms_per_split": summary["atoms_per_split"],
        "tiles_per_split": summary["tiles_per_split"],
        "deployment_test_clusters": summary["deployment_test_clusters"],
        "deployment_test": summary["deployment_test"],
        "data_yaml": summary["data_yaml"],
    }
    logger.info("Stratified prep digest:\n%s", json.dumps(digest, indent=2))

    # Per-split / per-class instance counts (compact)
    logger.info("Per-split instance counts:")
    for split in ("train", "valid", "test"):
        counts = summary["instances_per_split_per_class"][split]
        kept = {k: v for k, v in counts.items() if v > 0}
        logger.info("  %s: %s", split, json.dumps(kept))

    logger.info("Done. Sliced YOLO dataset is at: %s", sliced_out)
    logger.info("Drawing-level test set is at:    %s", drawing_out)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", default=str(REPO_ROOT / "configs" / "dataset.yaml"))
    parser.add_argument("--irrigation-config", default=str(REPO_ROOT / "configs" / "irrigation_classes.yaml"))
    parser.add_argument("--detection-config", default=str(REPO_ROOT / "configs" / "detection.yaml"))
    parser.add_argument("--copy-images", action="store_true",
                        help="Copy images instead of symlinking (slower, doubles disk usage).")
    parser.add_argument("--skip-slicing", action="store_true",
                        help="page_grouped only: skip the tiling step.")
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

    mode = det_cfg["split"].get("mode", "page_grouped")
    if mode == "page_grouped":
        return _run_page_grouped(
            source_ds=source_ds, irr_cfg=irr_cfg, det_cfg=det_cfg,
            ds_cfg=ds_cfg, args=args,
        )
    if mode == "stratified_tile":
        return _run_stratified_tile(
            source_ds=source_ds, irr_cfg=irr_cfg, det_cfg=det_cfg, args=args,
        )
    logger.error("Unknown split.mode: %r", mode)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
