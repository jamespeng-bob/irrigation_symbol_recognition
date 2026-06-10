"""Filter a Roboflow-style YOLO dataset down to the irrigation label space.

Inputs:
  * A YOLO dataset on disk (Roboflow ``YOLO*`` export).
  * An ``IrrigationConfig`` (which class names to keep, optional groupings,
    and the ``non_irrigation_policy``).

Outputs (written to ``out_dir``):
  * ``data.yaml`` describing the filtered label space.
  * ``train/{images,labels}``, ``valid/{images,labels}``, ``test/{images,labels}``.
  * Per-image ``.txt`` files with class IDs remapped into the new 0-indexed
    label space; non-irrigation annotations are either dropped, collapsed
    to ``__background__``, or kept (the "negative" policy keeps the image
    but writes an empty label file).
  * Images are symlinked by default to save disk (set ``symlink_images=False``
    to copy).
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

from ..data.yolo_dataset import YoloDataset
from ..utils.config import IrrigationConfig
from .splits import page_grouped_split


logger = logging.getLogger(__name__)


@dataclass
class _FilterPolicy:
    src_id_to_dst_id: dict[int, int]   # irrigation source IDs -> new contiguous IDs
    background_dst_id: int | None       # set if policy == "background"
    policy: str                         # "drop" | "background" | "negative"


def _build_policy(
    src_class_names: list[str],
    cfg: IrrigationConfig,
) -> tuple[_FilterPolicy, list[str], list[str]]:
    """Return (policy, final_label_names, missing_class_names)."""
    name_to_src_id = {n: i for i, n in enumerate(src_class_names)}
    raw_to_label = cfg.raw_to_label()

    final_labels = list(cfg.label_names)
    if cfg.non_irrigation_policy == "background":
        if "__background__" not in final_labels:
            final_labels.append("__background__")
    label_to_dst_id = {name: i for i, name in enumerate(final_labels)}

    src_id_to_dst_id: dict[int, int] = {}
    missing: list[str] = []
    for raw_name in cfg.irrigation_classes:
        src_id = name_to_src_id.get(raw_name)
        if src_id is None:
            missing.append(raw_name)
            continue
        label = raw_to_label[raw_name]
        src_id_to_dst_id[src_id] = label_to_dst_id[label]

    bg = label_to_dst_id.get("__background__")
    return (
        _FilterPolicy(
            src_id_to_dst_id=src_id_to_dst_id,
            background_dst_id=bg,
            policy=cfg.non_irrigation_policy,
        ),
        final_labels,
        missing,
    )


def _remap_line(
    src_class_id: int,
    rest: str,
    policy: _FilterPolicy,
) -> str | None:
    dst_id = policy.src_id_to_dst_id.get(src_class_id)
    if dst_id is not None:
        return f"{dst_id} {rest}"
    if policy.policy == "background" and policy.background_dst_id is not None:
        return f"{policy.background_dst_id} {rest}"
    return None  # drop / negative


def _link_or_copy(src: Path, dst: Path, *, symlink: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if symlink:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def _write_split(
    *,
    ds: YoloDataset,
    src_split: str,
    image_names_subset: Iterable[str] | None,
    out_img_dir: Path,
    out_lbl_dir: Path,
    policy: _FilterPolicy,
    symlink_images: bool,
) -> dict:
    """Filter labels for one (sub-)split, returning per-class annotation counts."""
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    subset = set(image_names_subset) if image_names_subset is not None else None

    n_images = 0
    n_irrigation_anns = 0
    n_background_anns = 0
    n_dropped_anns = 0
    per_dst_id_counts: dict[int, int] = {}

    for image_path in ds.split_images(src_split):
        if subset is not None and image_path.name not in subset:
            continue
        src_label_path = ds.label_for(image_path)

        out_label_lines: list[str] = []
        if src_label_path.exists():
            with src_label_path.open("r") as fh:
                for line in fh:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) != 2:
                        continue
                    try:
                        src_id = int(parts[0])
                    except ValueError:
                        continue
                    out_line = _remap_line(src_id, parts[1], policy)
                    if out_line is None:
                        n_dropped_anns += 1
                        continue
                    out_label_lines.append(out_line)
                    new_id = int(out_line.split(maxsplit=1)[0])
                    per_dst_id_counts[new_id] = per_dst_id_counts.get(new_id, 0) + 1
                    if policy.background_dst_id is not None and new_id == policy.background_dst_id:
                        n_background_anns += 1
                    else:
                        n_irrigation_anns += 1

        out_label_path = out_lbl_dir / (image_path.stem + ".txt")
        with out_label_path.open("w") as fh:
            fh.write("\n".join(out_label_lines))
            if out_label_lines:
                fh.write("\n")

        out_image_path = out_img_dir / image_path.name
        _link_or_copy(image_path, out_image_path, symlink=symlink_images)
        n_images += 1

    return {
        "num_images": n_images,
        "num_irrigation_annotations": n_irrigation_anns,
        "num_background_annotations": n_background_anns,
        "num_dropped_annotations": n_dropped_anns,
        "annotations_per_dst_id": per_dst_id_counts,
    }


def _write_data_yaml(
    out_root: Path,
    *,
    label_names: list[str],
    has_valid: bool,
    has_test: bool,
) -> None:
    data: dict = {
        "path": str(out_root.resolve()),
        "train": "train/images",
        "nc": len(label_names),
        "names": list(label_names),
    }
    if has_valid:
        data["val"] = "valid/images"
    elif has_test:
        data["val"] = "test/images"   # fall back so ultralytics has a val set
    if has_test:
        data["test"] = "test/images"
    with (out_root / "data.yaml").open("w") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


def filter_yolo_dataset(
    *,
    source_ds: YoloDataset,
    irrigation_cfg: IrrigationConfig,
    out_dir: Path,
    valid_fraction: float = 0.10,
    seed: int = 42,
    group_by_page_prefix: bool = True,
    symlink_images: bool = True,
) -> dict:
    """End-to-end YOLO filtering.

    * Reads the source YOLO ``data.yaml`` for the original class names.
    * Applies ``irrigation_cfg`` (subset + groups + non-irrigation policy).
    * Carves a page-prefix-grouped ``valid`` split off ``train``.
    * Writes the filtered dataset (with a fresh ``data.yaml``) under ``out_dir``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    policy, final_labels, missing = _build_policy(source_ds.class_names, irrigation_cfg)
    if missing:
        logger.warning(
            "irrigation_classes contains %d names not present in the source data.yaml: %s",
            len(missing), missing[:10] + (["..."] if len(missing) > 10 else []),
        )

    train_images = source_ds.split_images("train")
    train_image_names = [p.name for p in train_images]
    train_only_names, valid_only_names = page_grouped_split(
        train_image_names,
        valid_fraction=valid_fraction,
        seed=seed,
        group_by_page=group_by_page_prefix,
    )

    train_report = _write_split(
        ds=source_ds,
        src_split="train",
        image_names_subset=train_only_names,
        out_img_dir=out_dir / "train" / "images",
        out_lbl_dir=out_dir / "train" / "labels",
        policy=policy,
        symlink_images=symlink_images,
    )
    valid_report = _write_split(
        ds=source_ds,
        src_split="train",
        image_names_subset=valid_only_names,
        out_img_dir=out_dir / "valid" / "images",
        out_lbl_dir=out_dir / "valid" / "labels",
        policy=policy,
        symlink_images=symlink_images,
    )

    has_test = "test" in source_ds.split_dirs
    test_report = (
        _write_split(
            ds=source_ds,
            src_split="test",
            image_names_subset=None,
            out_img_dir=out_dir / "test" / "images",
            out_lbl_dir=out_dir / "test" / "labels",
            policy=policy,
            symlink_images=symlink_images,
        )
        if has_test
        else {}
    )

    _write_data_yaml(
        out_dir,
        label_names=final_labels,
        has_valid=valid_report["num_images"] > 0,
        has_test=bool(test_report),
    )

    # Map dst-id -> human-readable label for the report
    dst_id_to_label = {i: name for i, name in enumerate(final_labels)}
    def _decorate(per_id: dict[int, int]) -> dict[str, int]:
        return {dst_id_to_label.get(i, f"id={i}"): n for i, n in sorted(per_id.items())}

    return {
        "label_names": final_labels,
        "num_classes": len(final_labels),
        "missing_class_names": missing,
        "train": {**train_report, "annotations_per_label": _decorate(train_report.get("annotations_per_dst_id", {}))},
        "valid": {**valid_report, "annotations_per_label": _decorate(valid_report.get("annotations_per_dst_id", {}))},
        "test":  {**(test_report or {"num_images": 0}), "annotations_per_label": _decorate((test_report or {}).get("annotations_per_dst_id", {}))} if has_test else {"num_images": 0},
        "data_yaml": str(out_dir / "data.yaml"),
    }
