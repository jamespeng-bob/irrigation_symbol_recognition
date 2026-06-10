"""Tile-based dataset slicing for YOLO detection.

This is a port of the reference implementation in
``electrical_dev-main/detection/data.py``. Behavior is intentionally kept
byte-for-byte equivalent so that any model trained from data produced by
this module is comparable to those trained by the reference pipeline:

* Sliding ``slice_size x slice_size`` crop with ``overlap`` fraction.
* Final row/col anchored to ``(W - S, H - S)`` so the right/bottom edge is
  fully covered; zero-padding is used only when the crop still ends up
  smaller than ``slice_size``.
* For boxes that cross a slice boundary, ``adjust_cut_bbox_in_slice``
  decides whether to keep the (clipped) box, the full original box, or
  drop it entirely (see docstring there).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


logger = logging.getLogger(__name__)


def bbox_intersects_slice(
    bbox_norm: List[float],
    slice_x: int,
    slice_y: int,
    slice_size: int,
    img_w: int,
    img_h: int,
) -> bool:
    """Cheap AABB overlap test between a normalized YOLO box and a slice."""
    x_center, y_center, norm_w, norm_h = bbox_norm

    x_abs = x_center * img_w
    y_abs = y_center * img_h
    w_abs = norm_w * img_w
    h_abs = norm_h * img_h

    bbox_x1 = x_abs - w_abs / 2
    bbox_y1 = y_abs - h_abs / 2
    bbox_x2 = x_abs + w_abs / 2
    bbox_y2 = y_abs + h_abs / 2

    slice_x1, slice_y1 = slice_x, slice_y
    slice_x2, slice_y2 = slice_x + slice_size, slice_y + slice_size

    return not (
        bbox_x2 < slice_x1
        or bbox_x1 > slice_x2
        or bbox_y2 < slice_y1
        or bbox_y1 > slice_y2
    )


def adjust_cut_bbox_in_slice(
    coords_cut: Tuple[float, float, float, float],
    coords_original: Tuple[float, float, float, float],
    slice_size: int,
    fit_box_to_slice: bool,
) -> Tuple[float, float, float, float] | None:
    """Decide what to do with a box that crosses a slice boundary.

    Mirrors the reference's simpler "majority-area" rule (the more elaborate
    SAHI-style logic the reference left commented out is preserved at the
    bottom for posterity).

    Returns ``None`` to indicate the box should be dropped for this slice.
    """
    if fit_box_to_slice:
        return coords_cut

    x1_cut, y1_cut, x2_cut, y2_cut = coords_cut
    x1_o, y1_o, x2_o, y2_o = coords_original

    w_cut, h_cut = x2_cut - x1_cut, y2_cut - y1_cut
    w_o, h_o = x2_o - x1_o, y2_o - y1_o

    area_cut = w_cut * h_cut
    area_original = w_o * h_o
    if area_original <= 0:
        return None

    if area_cut / area_original > 0.5:
        return coords_original

    return None

    # --- Reference's commented-out SAHI-style alternative (kept for posterity) ---
    # ratio = max(w_o, h_o) / min(w_o, h_o)
    # if area_cut / area_original == 1.0:
    #     return coords_original
    # elif 1 <= ratio < 1.05:
    #     if area_cut / area_original > 0.1:
    #         return coords_original
    # elif 1.05 <= ratio < 2.3:
    #     if area_cut / area_original > 0.35:
    #         return coords_original
    #     elif area_cut / area_original > 0.1:
    #         return coords_cut
    # else:
    #     overlap = 0.2
    #     if w_cut / slice_size > overlap or h_cut / slice_size > overlap:
    #         return coords_cut


def adjust_bbox_for_slice(
    bbox_norm: List[float],
    slice_x: int,
    slice_y: int,
    slice_size: int,
    img_w: int,
    img_h: int,
    fit_box_to_slice: bool,
) -> List[float] | None:
    """Translate a normalized full-image YOLO box into normalized slice coords.

    Returns ``None`` if the box should be dropped for this slice (either
    because there is no overlap at all or because ``adjust_cut_bbox_in_slice``
    rejected it).
    """
    x_center, y_center, norm_w, norm_h = bbox_norm

    x_abs = x_center * img_w
    y_abs = y_center * img_h
    w_abs = norm_w * img_w
    h_abs = norm_h * img_h

    bbox_x1 = x_abs - w_abs / 2
    bbox_y1 = y_abs - h_abs / 2
    bbox_x2 = x_abs + w_abs / 2
    bbox_y2 = y_abs + h_abs / 2

    x1_original = bbox_x1 - slice_x
    y1_original = bbox_y1 - slice_y
    x2_original = bbox_x2 - slice_x
    y2_original = bbox_y2 - slice_y

    x1_cut = max(0.0, x1_original)
    y1_cut = max(0.0, y1_original)
    x2_cut = min(float(slice_size), x2_original)
    y2_cut = min(float(slice_size), y2_original)

    if x2_cut <= x1_cut or y2_cut <= y1_cut:
        return None

    coords_rel = adjust_cut_bbox_in_slice(
        (x1_cut, y1_cut, x2_cut, y2_cut),
        (x1_original, y1_original, x2_original, y2_original),
        slice_size,
        fit_box_to_slice=fit_box_to_slice,
    )
    if coords_rel is None:
        return None

    x1_rel, y1_rel, x2_rel, y2_rel = coords_rel

    new_w = x2_rel - x1_rel
    new_h = y2_rel - y1_rel
    new_x_center = (x1_rel + x2_rel) / 2
    new_y_center = (y1_rel + y2_rel) / 2

    new_x_center_norm = new_x_center / slice_size
    new_y_center_norm = new_y_center / slice_size
    new_w_norm = new_w / slice_size
    new_h_norm = new_h / slice_size

    if new_w_norm > 0 and new_h_norm > 0:
        return [new_x_center_norm, new_y_center_norm, new_w_norm, new_h_norm]
    return None


def slice_image_and_labels(
    image_path: Path,
    label_path: Path,
    slice_size: int,
    output_dir: Path,
    split_name: str,
    overlap: float,
    fit_box_to_slice: bool,
) -> int:
    """Slice a single (image, YOLO-label) pair into tiles.

    Returns the number of tiles written.
    """
    assert 0 <= overlap < 1

    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning("Could not load image %s", image_path)
        return 0

    h, w = img.shape[:2]
    image_name = image_path.stem

    labels: list[list[float]] = []
    if label_path.exists():
        with open(label_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    logger.warning("invalid labels in %s", image_path.stem)
                    continue
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                labels.append([class_id, x_center, y_center, width, height])

    step = int(slice_size * (1 - overlap))
    y_positions = list(range(0, h, step))
    x_positions = list(range(0, w, step))

    if y_positions and y_positions[-1] + slice_size < h:
        y_positions.append(h - slice_size)
    if x_positions and x_positions[-1] + slice_size < w:
        x_positions.append(w - slice_size)

    out_img_dir = Path(output_dir) / split_name / "images"
    out_lbl_dir = Path(output_dir) / split_name / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    slice_count = 0
    for y_pos in y_positions:
        for x_pos in x_positions:
            y_end = min(y_pos + slice_size, h)
            x_end = min(x_pos + slice_size, w)

            actual_h = y_end - y_pos
            actual_w = x_end - x_pos

            slice_img = img[y_pos:y_end, x_pos:x_end]

            if actual_h < slice_size or actual_w < slice_size:
                padded_slice = np.zeros((slice_size, slice_size, 3), dtype=slice_img.dtype)
                padded_slice[:actual_h, :actual_w] = slice_img
                slice_img = padded_slice

            slice_filename = f"{image_name}_slice_{x_pos}_{y_pos}.jpg"
            slice_path = out_img_dir / slice_filename
            cv2.imwrite(str(slice_path), slice_img)

            label_filename = f"{image_name}_slice_{x_pos}_{y_pos}.txt"
            label_out = out_lbl_dir / label_filename

            with open(label_out, "w") as f:
                for label in labels:
                    class_id, x_center, y_center, width, height = label
                    bbox_norm = [x_center, y_center, width, height]
                    if bbox_intersects_slice(
                        bbox_norm, x_pos, y_pos, slice_size, w, h
                    ):
                        adjusted_bbox = adjust_bbox_for_slice(
                            bbox_norm,
                            x_pos,
                            y_pos,
                            slice_size,
                            w,
                            h,
                            fit_box_to_slice=fit_box_to_slice,
                        )
                        if adjusted_bbox:
                            f.write(
                                f"{int(class_id)} "
                                f"{adjusted_bbox[0]:.6f} "
                                f"{adjusted_bbox[1]:.6f} "
                                f"{adjusted_bbox[2]:.6f} "
                                f"{adjusted_bbox[3]:.6f}\n"
                            )
            slice_count += 1
    return slice_count


def process_split(
    split_dir: Path,
    split_name: str,
    slice_size: int,
    output_dir: Path,
    overlap: float = 0.0,
    fit_box_to_slice: bool = False,
) -> None:
    """Slice every (image, label) pair in ``split_dir``.

    ``split_dir`` is expected to contain ``images/`` and ``labels/``
    subdirectories (the standard YOLO layout produced by
    ``coco_to_yolo.export_split``).
    """
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    if not images_dir.exists():
        logger.warning("Images directory not found: %s, skipping %s", images_dir, split_name)
        return

    (Path(output_dir) / split_name / "images").mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / split_name / "labels").mkdir(parents=True, exist_ok=True)

    image_files = [
        f for f in images_dir.glob("*.*")
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    ]

    total_slices = 0
    for image_path in image_files:
        label_path = labels_dir / (image_path.stem + ".txt")
        logger.info("Processing %s...", image_path.name)
        slices = slice_image_and_labels(
            image_path,
            label_path,
            slice_size,
            output_dir,
            split_name,
            overlap=overlap,
            fit_box_to_slice=fit_box_to_slice,
        )
        total_slices += slices

    logger.info("%s split: %d source images -> %d tiles", split_name.upper(), len(image_files), total_slices)
    logger.info("  saved to %s", Path(output_dir) / split_name)


def slice_data(detection_data_config: dict) -> None:
    """Slice every split (``train``, ``valid``, ``test``) of a YOLO dataset.

    Expected keys: ``dataset_dir``, ``output_dir``, ``image_size`` (slice
    size in pixels), ``overlap``, and optionally ``fit_box_to_slice``.

    Copies the existing ``data.yaml`` from ``dataset_dir`` to ``output_dir``
    so the sliced dataset is immediately usable by Ultralytics.
    """
    dataset_dir = Path(detection_data_config["dataset_dir"])
    output_dir = Path(detection_data_config["output_dir"])
    slice_size = int(detection_data_config["image_size"])
    overlap = float(detection_data_config["overlap"])
    fit_box_to_slice = bool(detection_data_config.get("fit_box_to_slice", False))

    if not dataset_dir.exists():
        raise ValueError(f"Dataset directory not found: {dataset_dir}")

    logger.info("=" * 60)
    logger.info("YOLO Dataset Slicing")
    logger.info("=" * 60)
    logger.info("Input dataset: %s", dataset_dir)
    logger.info("Output dataset: %s", output_dir)
    logger.info("Slice size: %dx%d  (overlap=%.2f, fit_box_to_slice=%s)", slice_size, slice_size, overlap, fit_box_to_slice)
    logger.info("=" * 60)

    for split_name in ("train", "valid", "test"):
        split_dir = dataset_dir / split_name
        if split_dir.exists():
            process_split(
                split_dir,
                split_name,
                slice_size,
                output_dir,
                overlap=overlap,
                fit_box_to_slice=fit_box_to_slice,
            )

    input_yaml = dataset_dir / "data.yaml"
    output_yaml = output_dir / "data.yaml"
    if input_yaml.exists():
        import yaml

        output_dir.mkdir(parents=True, exist_ok=True)
        with open(input_yaml, "r") as fh:
            data_cfg = yaml.safe_load(fh) or {}
        # The source data.yaml's `path` points at the unsliced dataset; it
        # must be rewritten or Ultralytics would train on the unsliced images.
        data_cfg["path"] = str(output_dir.absolute())
        with open(output_yaml, "w") as fh:
            yaml.safe_dump(data_cfg, fh, sort_keys=False)
        logger.info("Copied data.yaml from %s to %s (path -> %s)", input_yaml, output_yaml, output_dir)

    logger.info("=" * 60)
    logger.info("Dataset slicing complete! Output: %s", output_dir)
    logger.info("=" * 60)
