"""Sliding-window inference + per-class NMS.

Port of ``electrical_dev-main/detection/inference.py``. The only meaningful
difference is that we use the class-aware ``nms_boxes`` from ``.nms``
instead of the reference's electrical-domain ``nmm_boxes`` (which assumes
specific class groupings like text/panel/symbol-with-text that don't apply
to the irrigation label space).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .nms import nms_boxes


logger = logging.getLogger(__name__)


def inference_detection(
    *,
    model: Any,
    image: np.ndarray,
    slice_size: int = 1280,
    model_input_size: int = 640,
    overlap: float = 0.25,
    conf_threshold: float = 0.25,
    batch_size: int = 16,
    nms_iou_threshold: float = 0.5,
) -> list[dict]:
    """Run sliding-window inference and stitch tile predictions together.

    Returns a list of detection dicts with keys ``x1, y1, x2, y2,
    confidence, class_id, class_name`` in **full-image** coordinates.
    """
    h, w = image.shape[:2]
    step = int(slice_size * (1 - overlap))

    y_positions = list(range(0, h, step))
    x_positions = list(range(0, w, step))
    if y_positions and (y_positions[-1] + slice_size < h):
        y_positions.append(h - slice_size)
    if x_positions and (x_positions[-1] + slice_size < w):
        x_positions.append(w - slice_size)

    all_detections: list[dict] = []

    batch_imgs: list[np.ndarray] = []
    batch_meta: list[tuple[int, int, int, int]] = []

    def _flush_batch() -> None:
        if not batch_imgs:
            return
        results = model.predict(
            batch_imgs,
            imgsz=model_input_size,
            conf=conf_threshold,
            verbose=False,
        )
        for result, (x_pos, y_pos, _, _) in zip(results, batch_meta):
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            for (x1s, y1s, x2s, y2s), conf, cls_id in zip(xyxy, confs, clss):
                if x2s <= x1s or y2s <= y1s:
                    continue
                x1_abs = int(x1s) + x_pos
                y1_abs = int(y1s) + y_pos
                x2_abs = int(x2s) + x_pos
                y2_abs = int(y2s) + y_pos
                x1_abs = max(0, min(w - 1, x1_abs))
                y1_abs = max(0, min(h - 1, y1_abs))
                x2_abs = max(0, min(w - 1, x2_abs))
                y2_abs = max(0, min(h - 1, y2_abs))
                all_detections.append({
                    "x1": x1_abs, "y1": y1_abs, "x2": x2_abs, "y2": y2_abs,
                    "confidence": float(conf),
                    "class_id": int(cls_id),
                })
        batch_imgs.clear()
        batch_meta.clear()

    for y_pos in y_positions:
        for x_pos in x_positions:
            y_end = min(y_pos + slice_size, h)
            x_end = min(x_pos + slice_size, w)
            actual_h = y_end - y_pos
            actual_w = x_end - x_pos
            slice_img = image[y_pos:y_end, x_pos:x_end]
            if actual_h < slice_size or actual_w < slice_size:
                padded = np.zeros((slice_size, slice_size, 3), dtype=slice_img.dtype)
                padded[:actual_h, :actual_w] = slice_img
                slice_img = padded
            batch_imgs.append(slice_img)
            batch_meta.append((x_pos, y_pos, actual_w, actual_h))
            if len(batch_imgs) >= batch_size:
                _flush_batch()
    _flush_batch()

    logger.info("Found %d detections before NMS", len(all_detections))
    filtered = nms_boxes(all_detections, iou_threshold=nms_iou_threshold)
    logger.info("Found %d detections after NMS", len(filtered))

    if hasattr(model, "names"):
        names = model.names
        for det in filtered:
            cid = det["class_id"]
            det["class_name"] = names[cid] if cid in names else str(cid)
    return filtered


def run_on_image_file(
    *,
    model: Any,
    image_path: str | Path,
    slice_size: int = 1280,
    model_input_size: int = 640,
    overlap: float = 0.25,
    conf_threshold: float = 0.25,
    batch_size: int = 16,
    nms_iou_threshold: float = 0.5,
) -> list[dict]:
    """Convenience wrapper that loads an image file with OpenCV."""
    import cv2  # imported lazily

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return inference_detection(
        model=model,
        image=image,
        slice_size=slice_size,
        model_input_size=model_input_size,
        overlap=overlap,
        conf_threshold=conf_threshold,
        batch_size=batch_size,
        nms_iou_threshold=nms_iou_threshold,
    )
