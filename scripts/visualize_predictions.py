"""Overlay ground-truth labels and YOLO predictions on full-resolution drawings.

For every image in ``--images-dir``, this script:

  1. Runs sliding-window inference (matches `configs/detection.yaml -> inference`).
  2. Reads the ground-truth YOLO label from ``--labels-dir``.
  3. Draws boxes on a copy of the source image at its **original resolution**
     and writes the annotated JPG to ``--out-dir``.

Visual conventions:

  * Ground-truth irrigation symbol .................. solid **green**, with class name
  * Predicted irrigation symbol ..................... solid **red**, with class + confidence
  * Ground-truth ``__background__`` (other symbol) .. thin **gray**, no label
  * Predicted ``__background__`` .................... thin **orange**, no label

So the "interesting" boxes pop out, while the catch-all `__background__`
boxes are visible as context without dominating the view.

Usage (on the server, with GPU)::

    python scripts/visualize_predictions.py \
        --model runs/detect/runs/irrigation_yolo_v1/weights/best.pt \
        --images-dir data/detection/irrigation_yolo/valid/images \
        --labels-dir data/detection/irrigation_yolo/valid/labels \
        --out-dir runs/detect/runs/irrigation_yolo_v1/predictions_viz_valid
"""

from __future__ import annotations

import argparse
import logging
import sys
from itertools import chain
from pathlib import Path

import cv2
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from irrigation_symbol_recognition.detection.inference import run_on_image_file  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("visualize_predictions")


# Colors are BGR (OpenCV).
GT_IRRIG_COLOR  = (0, 200, 0)        # bright green
PR_IRRIG_COLOR  = (40, 40, 220)      # red
GT_BG_COLOR     = (170, 170, 170)    # neutral gray
PR_BG_COLOR     = (60, 140, 220)     # muted orange


def _scaled_thickness(h: int, w: int, *, base: float = 0.0015) -> int:
    return max(2, int(max(h, w) * base))


def _scaled_font(h: int, w: int) -> float:
    return max(0.5, max(h, w) / 2400.0)


def _draw_box(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    label: str,
    color: tuple[int, int, int],
    thickness: int,
    font_scale: float,
) -> None:
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if not label:
        return
    text_thickness = max(1, thickness // 2)
    (tw, th), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness,
    )
    label_y = max(y1 - 4, th + 4)
    cv2.rectangle(
        img,
        (x1, label_y - th - 6),
        (x1 + tw + 6, label_y + 2),
        color,
        thickness=-1,
    )
    cv2.putText(
        img, label,
        (x1 + 3, label_y - 2),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
        text_thickness, cv2.LINE_AA,
    )


def _read_gt(label_path: Path, w: int, h: int, class_names: list[str]) -> list[dict]:
    gts: list[dict] = []
    if not label_path.exists():
        return gts
    with label_path.open("r") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cid = int(parts[0])
            xc, yc, bw, bh = (float(v) for v in parts[1:])
            x1 = int(round((xc - bw / 2) * w))
            y1 = int(round((yc - bh / 2) * h))
            x2 = int(round((xc + bw / 2) * w))
            y2 = int(round((yc + bh / 2) * h))
            gts.append({
                "x1": max(0, x1), "y1": max(0, y1),
                "x2": min(w - 1, x2), "y2": min(h - 1, y2),
                "class_id": cid,
                "class_name": class_names[cid] if cid < len(class_names) else str(cid),
            })
    return gts


def _add_legend(
    img: np.ndarray,
    n_gt_irrig: int, n_pred_irrig: int,
    n_gt_bg: int, n_pred_bg: int,
    image_name: str,
) -> None:
    h, w = img.shape[:2]
    pad = max(20, int(min(h, w) * 0.01))
    line_h = max(28, int(min(h, w) * 0.018))
    fs = _scaled_font(h, w) * 1.1
    th = max(1, int(min(h, w) * 0.0009))
    box_h = pad * 2 + line_h * 5
    box_w = max(int(min(h, w) * 0.35), 600)
    x0, y0 = pad, pad
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
    cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), max(1, th))

    def _line(idx: int, txt: str, swatch: tuple[int, int, int] | None) -> None:
        y = y0 + pad + line_h * idx + line_h // 2
        sx = x0 + pad
        if swatch is not None:
            cv2.rectangle(img, (sx, y - line_h // 2), (sx + line_h, y + line_h // 2), swatch, -1)
            sx += int(line_h * 1.4)
        cv2.putText(img, txt, (sx, y + line_h // 3), cv2.FONT_HERSHEY_SIMPLEX, fs, (20, 20, 20), th * 2, cv2.LINE_AA)

    _line(0, image_name, None)
    _line(1, f"GT irrigation  ({n_gt_irrig})",       GT_IRRIG_COLOR)
    _line(2, f"Pred irrigation  ({n_pred_irrig})",   PR_IRRIG_COLOR)
    _line(3, f"GT __background__  ({n_gt_bg})",      GT_BG_COLOR)
    _line(4, f"Pred __background__  ({n_pred_bg})",  PR_BG_COLOR)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detection-config", default=str(REPO_ROOT / "configs" / "detection.yaml"))
    parser.add_argument("--model", required=True, help="Path to a YOLO .pt (e.g. .../weights/best.pt).")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N images (0 = all).")
    parser.add_argument("--no-background", action="store_true",
                        help="Hide __background__ boxes entirely (default is to draw them faintly).")
    parser.add_argument("--jpeg-quality", type=int, default=92)
    args = parser.parse_args()

    from ultralytics import YOLO  # type: ignore

    with Path(args.detection_config).open("r") as fh:
        det_cfg = yaml.safe_load(fh) or {}
    inf_cfg = det_cfg["inference"]

    model = YOLO(args.model)
    names_map = model.names
    class_names = [names_map[i] for i in sorted(names_map)]
    bg_id = next((i for i, n in enumerate(class_names) if n == "__background__"), None)
    logger.info("Loaded model %s (%d classes)", args.model, len(class_names))

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(chain(
        images_dir.glob("*.jpg"), images_dir.glob("*.jpeg"), images_dir.glob("*.png"),
    ))
    if args.limit > 0:
        image_paths = image_paths[: args.limit]
    logger.info("Visualizing %d images -> %s", len(image_paths), out_dir)

    for k, img_path in enumerate(image_paths, 1):
        real_path = img_path.resolve()
        img = cv2.imread(str(real_path))
        if img is None:
            logger.warning("[%d/%d] could not read %s", k, len(image_paths), real_path)
            continue
        h, w = img.shape[:2]
        thickness_main = _scaled_thickness(h, w, base=0.0015)
        thickness_bg = max(1, thickness_main // 2)
        font_scale = _scaled_font(h, w)
        logger.info(
            "[%d/%d] %s (%dx%d)", k, len(image_paths), img_path.name, w, h,
        )

        preds = run_on_image_file(
            model=model,
            image_path=real_path,
            slice_size=int(inf_cfg["slice_size"]),
            model_input_size=int(inf_cfg["model_input_size"]),
            overlap=float(inf_cfg["overlap"]),
            conf_threshold=float(inf_cfg["conf_threshold"]),
            batch_size=int(inf_cfg["batch_size"]),
            nms_iou_threshold=float(inf_cfg["nms_iou_threshold"]),
        )
        gts = _read_gt(labels_dir / (img_path.stem + ".txt"), w, h, class_names)

        gt_bg    = [g for g in gts   if g["class_name"] == "__background__"]
        gt_irrig = [g for g in gts   if g["class_name"] != "__background__"]
        pr_bg    = [d for d in preds if d["class_name"] == "__background__"]
        pr_irrig = [d for d in preds if d["class_name"] != "__background__"]

        # Draw background layers first (faintly), then irrigation on top so
        # the interesting boxes are always visible.
        if not args.no_background:
            for g in gt_bg:
                _draw_box(img, g["x1"], g["y1"], g["x2"], g["y2"],
                          "", GT_BG_COLOR, thickness_bg, font_scale)
            for d in pr_bg:
                _draw_box(img, int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"]),
                          "", PR_BG_COLOR, thickness_bg, font_scale)

        for g in gt_irrig:
            _draw_box(img, g["x1"], g["y1"], g["x2"], g["y2"],
                      f"GT {g['class_name']}", GT_IRRIG_COLOR,
                      thickness_main, font_scale)
        for d in pr_irrig:
            _draw_box(img, int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"]),
                      f"{d['class_name']} {d['confidence']:.2f}", PR_IRRIG_COLOR,
                      thickness_main, font_scale)

        _add_legend(
            img,
            n_gt_irrig=len(gt_irrig), n_pred_irrig=len(pr_irrig),
            n_gt_bg=len(gt_bg), n_pred_bg=len(pr_bg),
            image_name=img_path.name,
        )

        out_path = out_dir / (img_path.stem + ".jpg")
        cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, int(args.jpeg_quality)])
        logger.info(
            "    GT: %d irrig + %d bg  |  Pred: %d irrig + %d bg  -> %s",
            len(gt_irrig), len(gt_bg), len(pr_irrig), len(pr_bg), out_path,
        )

    logger.info("Done. Outputs at %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
