"""Helpers for loading and validating the YAML configs under `configs/`."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = REPO_ROOT / "configs"


def resolve_path(path: str | Path, *, anchor: Path = REPO_ROOT) -> Path:
    """Resolve a (possibly relative) path against the repo root."""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (anchor / p).resolve()


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as fh:
        return yaml.safe_load(fh) or {}


def load_dataset_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else CONFIGS_DIR / "dataset.yaml"
    return _read_yaml(cfg_path)


@dataclass
class IrrigationConfig:
    """Typed view of `configs/irrigation_classes.yaml`."""

    irrigation_classes: list[str] = field(default_factory=list)
    class_groups: dict[str, list[str]] = field(default_factory=dict)
    class_descriptions: dict[str, str] = field(default_factory=dict)
    non_irrigation_policy: str = "background"

    def __post_init__(self) -> None:
        allowed = {"drop", "background", "negative"}
        if self.non_irrigation_policy not in allowed:
            raise ValueError(
                f"non_irrigation_policy must be one of {sorted(allowed)}; "
                f"got {self.non_irrigation_policy!r}"
            )

        # Every class listed inside class_groups must also appear in
        # irrigation_classes so the user can't accidentally include a class
        # via grouping but forget to add it to the irrigation set.
        listed = set(self.irrigation_classes)
        for label, members in self.class_groups.items():
            missing = [m for m in members if m not in listed]
            if missing:
                raise ValueError(
                    f"class_groups[{label!r}] references classes that are "
                    f"not in irrigation_classes: {missing}"
                )

    @property
    def label_names(self) -> list[str]:
        """Final list of training label names after applying class_groups.

        Order: grouped labels first (sorted), then ungrouped raw class
        names (in the order the user listed them in `irrigation_classes`).
        """
        grouped_members = {m for members in self.class_groups.values() for m in members}
        ungrouped = [c for c in self.irrigation_classes if c not in grouped_members]
        return sorted(self.class_groups.keys()) + ungrouped

    def raw_to_label(self) -> dict[str, str]:
        """Map raw COCO class name -> final training label name."""
        mapping: dict[str, str] = {}
        for label, members in self.class_groups.items():
            for m in members:
                mapping[m] = label
        for c in self.irrigation_classes:
            mapping.setdefault(c, c)
        return mapping


def load_irrigation_config(path: str | Path | None = None) -> IrrigationConfig:
    cfg_path = Path(path) if path else CONFIGS_DIR / "irrigation_classes.yaml"
    raw = _read_yaml(cfg_path)
    return IrrigationConfig(
        irrigation_classes=list(raw.get("irrigation_classes") or []),
        class_groups={k: list(v) for k, v in (raw.get("class_groups") or {}).items()},
        class_descriptions=dict(raw.get("class_descriptions") or {}),
        non_irrigation_policy=raw.get("non_irrigation_policy", "background"),
    )
