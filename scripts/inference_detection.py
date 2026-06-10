"""Sliding-window inference for a trained YOLO model.

Pass either a single ``--image`` path or a directory via ``--image-dir``.
Detection JSON files are written to ``--out-dir``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from itertools import chain
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from irrigation_symbol_recognition.detection.inference import (  # noqa: E402
    run_on_image_file,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("inference_detection")


def _load_detection_config(path: Path) -> dict:
    import yaml

    with path.open("r") as fh:
        return yaml.safe_load(fh) or {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detection-config", default=str(REPO_ROOT / "configs" / "detection.yaml"))
    parser.add_argument("--model", required=True, help="Path to YOLO .pt (e.g. runs/.../weights/best.pt).")
    parser.add_argument("--image", default=None, help="Single image path.")
    parser.add_argument("--image-dir", default=None, help="Directory of images.")
    parser.add_argument("--out-dir", required=True, help="Where to write per-image detection JSON.")
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("Provide --image or --image-dir.")

    from ultralytics import YOLO  # type: ignore

    det_cfg = _load_detection_config(Path(args.detection_config))
    inf_cfg = det_cfg["inference"]

    model = YOLO(args.model)
    logger.info("Loaded model %s with %d classes", args.model, len(model.names))

    if args.image:
        image_paths = [Path(args.image)]
    else:
        image_dir = Path(args.image_dir)
        image_paths = list(chain(
            image_dir.glob("*.jpg"),
            image_dir.glob("*.jpeg"),
            image_dir.glob("*.png"),
        ))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        logger.info("Running inference on %s", img_path)
        dets = run_on_image_file(
            model=model,
            image_path=img_path,
            slice_size=int(inf_cfg["slice_size"]),
            model_input_size=int(inf_cfg["model_input_size"]),
            overlap=float(inf_cfg["overlap"]),
            conf_threshold=float(inf_cfg["conf_threshold"]),
            batch_size=int(inf_cfg["batch_size"]),
            nms_iou_threshold=float(inf_cfg["nms_iou_threshold"]),
        )
        out_path = out_dir / (img_path.stem + ".detections.json")
        out_path.write_text(json.dumps({
            "image": str(img_path),
            "num_detections": len(dets),
            "detections": dets,
        }, indent=2))
        logger.info("  %d detections -> %s", len(dets), out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
