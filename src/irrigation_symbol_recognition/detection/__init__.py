"""YOLO detection pipeline (tile-based training / inference).

The slicing and cut-box logic follows the technique used in the colleagues'
reference module under ``electrical_dev-main/detection/``:

* Slide a fixed ``slice_size`` window over the image with ``overlap``.
* Re-anchor the last row/column to ``(W - S, H - S)`` so the right/bottom
  edge is fully covered; pad with zeros only as a safety net.
* For boxes that cross a slice boundary, keep the **full original box** when
  more than 50 percent of the box's area lies inside the slice; otherwise
  drop that annotation for the slice. (Switch to ``fit_box_to_slice=True``
  to instead clip the box at the slice edge.)
"""

from .nms import nms_boxes
from .slicing import (
    adjust_bbox_for_slice,
    adjust_cut_bbox_in_slice,
    bbox_intersects_slice,
    process_split,
    slice_data,
    slice_image_and_labels,
)
from .yolo_filter import filter_yolo_dataset

__all__ = [
    "adjust_bbox_for_slice",
    "adjust_cut_bbox_in_slice",
    "bbox_intersects_slice",
    "filter_yolo_dataset",
    "nms_boxes",
    "process_split",
    "slice_data",
    "slice_image_and_labels",
]
