"""Launch Ultralytics YOLO training on the sliced irrigation dataset."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from irrigation_symbol_recognition.detection.train import (  # noqa: E402
    _select_device,
    train_with_ultralytics,
)
from irrigation_symbol_recognition.utils.config import resolve_path  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_detection")


def _load_detection_config(path: Path) -> dict:
    import yaml

    with path.open("r") as fh:
        return yaml.safe_load(fh) or {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detection-config", default=str(REPO_ROOT / "configs" / "detection.yaml"))
    parser.add_argument("--dataset-dir", default=None, help="Override sliced YOLO dataset dir.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    det_cfg = _load_detection_config(Path(args.detection_config))
    paths_cfg = det_cfg["paths"]
    training_cfg = det_cfg["training"]

    dataset_dir = resolve_path(args.dataset_dir or paths_cfg["sliced_dataset_dir"])
    if not (dataset_dir / "data.yaml").exists():
        logger.error("data.yaml not found in %s. Run prepare_detection_dataset.py first.", dataset_dir)
        return 1

    device = _select_device(args.device or training_cfg.get("device", "auto"))
    epochs = int(args.epochs if args.epochs is not None else training_cfg["epochs"])
    batch_size = int(args.batch if args.batch is not None else training_cfg["batch_size"])
    name = args.name or paths_cfg.get("runs_name", "irrigation_yolo_v1")
    project = paths_cfg.get("runs_project", "runs")

    logger.info("Starting training on %s (device=%s, epochs=%d, batch=%d)",
                dataset_dir, device, epochs, batch_size)

    model, results = train_with_ultralytics(
        dataset_dir=dataset_dir,
        image_size=int(training_cfg["image_size"]),
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        model_size=str(training_cfg.get("model_size", "m")),
        model_name=str(training_cfg.get("model_name", "") or ""),
        pretrained_path=training_cfg.get("pretrained_path") or None,
        pretrained_model_name=training_cfg.get("model_yaml") or None,
        model_path=training_cfg.get("model_path", "") or "",
        project=project,
        name=name,
    )

    logger.info("Best weights: %s", Path(results.save_dir) / "weights" / "best.pt")
    logger.info("Last weights: %s", Path(results.save_dir) / "weights" / "last.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
