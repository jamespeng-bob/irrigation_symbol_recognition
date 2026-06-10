"""Train/valid splitter that groups by source PDF page.

The Roboflow export contains multiple sub-tiles of the same source PDF
page (e.g. ``00_png.rf.<hash>.jpg`` and ``00_png_1_0_0_3600_2700_png.rf.<hash>.jpg``
both come from page ``00``). To avoid leakage between ``train`` and
``valid``, we group every sub-tile under a single "page key" and split
at the page level.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from typing import Iterable


_PAGE_KEY_TRIM = re.compile(r"(_png_\d+_\d+_\d+_\d+_\d+)?(\.rf\.[0-9a-f]+)?$", re.IGNORECASE)


def page_key(file_name: str) -> str:
    """Return a deterministic key identifying the source PDF page.

    The heuristic strips the Roboflow ``.rf.<hash>`` suffix and any
    ``_png_<i>_<x>_<y>_<x2>_<y2>`` sub-tile suffix. Anything left ahead of
    the first ``_png`` token is treated as the page key.
    """
    stem = file_name.rsplit(".", 1)[0]
    stem = _PAGE_KEY_TRIM.sub("", stem)
    # Cut at the first "_png" / "_jpg" boundary so we collapse sub-tiles.
    for marker in ("_png", "_jpg", "_jpeg"):
        idx = stem.find(marker)
        if idx >= 0:
            return stem[:idx]
    return stem


def page_grouped_split(
    file_names: Iterable[str],
    *,
    valid_fraction: float,
    seed: int,
    group_by_page: bool = True,
) -> tuple[list[str], list[str]]:
    """Return ``(train_files, valid_files)``.

    When ``group_by_page=True``, all sub-tiles sharing a page key end up
    in the same split.
    """
    files = list(file_names)
    rng = random.Random(seed)

    if not group_by_page or valid_fraction <= 0:
        files_shuffled = files[:]
        rng.shuffle(files_shuffled)
        cut = int(round(len(files_shuffled) * valid_fraction))
        return files_shuffled[cut:], files_shuffled[:cut]

    groups: dict[str, list[str]] = defaultdict(list)
    for name in files:
        groups[page_key(name)].append(name)

    keys = list(groups.keys())
    rng.shuffle(keys)

    n_total = len(files)
    target_valid = int(round(n_total * valid_fraction))
    valid_files: list[str] = []
    valid_keys: set[str] = set()
    for k in keys:
        if len(valid_files) >= target_valid:
            break
        valid_files.extend(groups[k])
        valid_keys.add(k)

    train_files: list[str] = []
    for k in keys:
        if k not in valid_keys:
            train_files.extend(groups[k])

    return train_files, valid_files
