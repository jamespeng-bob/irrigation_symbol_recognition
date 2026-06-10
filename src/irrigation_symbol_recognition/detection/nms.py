"""Class-aware NMS for tile-merged YOLO predictions.

The reference inference module runs a domain-specific Non-Maximum Merging
("NMM") pass tuned for electrical drawings (text/panel/symbol-with-text
groups). For irrigation symbols, a straightforward per-class NMS at the
global-image level is the right analogue: it cleans up duplicate boxes
where the same symbol was detected from two overlapping tiles.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def _to_array(boxes: Iterable[dict]) -> np.ndarray:
    return np.array(
        [
            [b["x1"], b["y1"], b["x2"], b["y2"], b["confidence"], b["class_id"]]
            for b in boxes
        ],
        dtype=np.float64,
    )


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of one box ``a`` (4,) against many boxes ``b`` (N,4)."""
    x1 = np.maximum(a[0], b[:, 0])
    y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2])
    y2 = np.minimum(a[3], b[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = np.maximum(0.0, (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)


def nms_boxes(boxes: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    """Per-class NMS over a list of detection dicts.

    Each dict must have keys ``x1, y1, x2, y2, confidence, class_id``;
    extra keys (e.g. ``class_name``) are preserved on the kept detections.
    """
    if not boxes:
        return []

    arr = _to_array(boxes)
    keep_indices: list[int] = []

    for cls in np.unique(arr[:, 5]):
        cls_mask = arr[:, 5] == cls
        cls_indices = np.where(cls_mask)[0]
        cls_arr = arr[cls_mask]
        order = cls_arr[:, 4].argsort()[::-1]  # by confidence desc
        cls_indices = cls_indices[order]
        cls_arr = cls_arr[order]

        suppressed = np.zeros(len(cls_arr), dtype=bool)
        for i in range(len(cls_arr)):
            if suppressed[i]:
                continue
            keep_indices.append(int(cls_indices[i]))
            if i + 1 == len(cls_arr):
                break
            ious = _iou(cls_arr[i, :4], cls_arr[i + 1 :, :4])
            suppressed[i + 1 :] |= ious >= iou_threshold

    keep_indices.sort()
    return [boxes[i] for i in keep_indices]
