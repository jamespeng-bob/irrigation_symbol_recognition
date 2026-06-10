"""Thin Ultralytics training wrapper, driven by ``configs/detection.yaml``.

Mirrors ``electrical_dev-main/detection/train.py``: a single
``train_with_ultralytics`` entry point that supports starting either from
public weights, a colleague-provided pretrained checkpoint, or an existing
local ``best.pt``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


def _select_device(device_arg: str) -> str:
    if device_arg and device_arg != "auto":
        return device_arg
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_yolo_dataset_yaml(dataset_dir: str | Path) -> str:
    """Locate the ``data.yaml`` that describes a sliced YOLO dataset."""
    dataset_path = Path(dataset_dir).absolute()
    existing_yaml = dataset_path / "data.yaml"
    if not existing_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found under {dataset_path}")
    logger.info("Using data.yaml: %s", existing_yaml)
    return str(existing_yaml)


def train_with_ultralytics(
    *,
    dataset_dir: str | Path,
    image_size: int,
    epochs: int,
    batch_size: int,
    device: str,
    model_size: str = "n",
    model_name: str = "",
    model_path: str = "",
    pretrained_path: str | None = None,
    pretrained_model_name: str | None = None,
    project: str = "runs",
    name: str = "irrigation_yolo",
    extra_train_kwargs: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    """Train an Ultralytics YOLO model on a sliced YOLO dataset.

    ``model_path``      : if set, resume / fine-tune from this checkpoint.
    ``pretrained_path`` : if set, load weights into a fresh ``YOLO(model_yaml)``
                          architecture (used when colleagues ship a ``.pt``
                          that doesn't carry its own config).
    Otherwise the public ``model_name`` weights are used (e.g. ``yolo26m.pt``),
    falling back to ``yolov<model_size>.pt`` if ``model_name`` is empty.
    """
    # Imported lazily so the rest of the package works without ultralytics.
    from ultralytics import YOLO  # type: ignore

    logger.info("Training with ultralytics YOLO")
    logger.info(
        "  model_size=%s image_size=%s epochs=%s batch=%s device=%s",
        model_size, image_size, epochs, batch_size, device,
    )

    dataset_yaml = get_yolo_dataset_yaml(dataset_dir)

    if model_path:
        logger.info("Loading model from: %s", model_path)
        model = YOLO(model_path)
    elif pretrained_path:
        logger.info("Loading pretrained weights from: %s", pretrained_path)
        if not pretrained_model_name:
            raise ValueError("pretrained_model_name is required when pretrained_path is set")
        model = YOLO(pretrained_model_name)
        model.load(pretrained_path)
    else:
        default_weights = model_name or f"yolov{model_size}.pt"
        logger.info("Loading default model: %s", default_weights)
        model = YOLO(default_weights)

    train_kwargs: dict[str, Any] = {
        "data": dataset_yaml,
        "epochs": epochs,
        "imgsz": image_size,
        "batch": batch_size,
        "device": device,
        "project": project,
        "name": name,
        "exist_ok": False,
    }
    if extra_train_kwargs:
        train_kwargs.update(extra_train_kwargs)

    results = model.train(**train_kwargs)

    logger.info("Training complete. Save dir: %s", results.save_dir)
    return model, results
