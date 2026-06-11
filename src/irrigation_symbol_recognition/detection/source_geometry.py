"""Roboflow-source-image geometry parsing.

Roboflow exports drawings with filenames that encode where each image sits
on its underlying PDF page. A typical export contains two kinds of source
images:

  * **Full page**     : ``<page>_png.rf.<hash>.jpg``
                        e.g. ``00_png.rf.21a3ae6d…jpg`` — represents *all* of
                        page ``00``.
  * **Sub-region**    : ``<page>_(png|jpg)_<i>_<x1>_<y1>_<x2>_<y2>_png.rf.<hash>.jpg``
                        e.g. ``00_png_1_0_2700_3600_5400_png.rf.b6128…jpg``
                        — covers page-coordinates ``(0, 2700)`` to
                        ``(3600, 5400)`` of page ``00``.

This module:

  * Extracts the ``page_key`` (the cluster identifier) from a filename.
  * Returns the source image's rectangle in source-page pixel space,
    when parseable.
  * Translates a tile's (x, y) inside a source image to the tile's
    rectangle in source-page pixel space, accounting for any image-vs-page
    rescaling.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


_RF_SUFFIX = re.compile(
    r"\.rf\.[0-9a-f]+\.(?:jpe?g|png)$", re.IGNORECASE,
)
_SUBREGION = re.compile(
    r"^(?P<page>.+?)_(?:png|jpg|jpeg)_\d+"
    r"_(?P<x1>\d+)_(?P<y1>\d+)_(?P<x2>\d+)_(?P<y2>\d+)"
    r"_(?:png|jpg|jpeg)$",
    re.IGNORECASE,
)
_FULL = re.compile(
    r"^(?P<page>.+?)_(?:png|jpg|jpeg)$", re.IGNORECASE,
)


@dataclass(frozen=True)
class SourceImageGeometry:
    """Resolved source-image -> source-page geometry."""

    page_key: str
    page_rect: tuple[int, int, int, int]   # (x1, y1, x2, y2) on the page
    image_dims: tuple[int, int]            # (width, height) of the source image file
    kind: str                              # "full" | "subregion" | "unparsed"

    @property
    def scale_x(self) -> float:
        x1, _, x2, _ = self.page_rect
        return (x2 - x1) / float(self.image_dims[0])

    @property
    def scale_y(self) -> float:
        _, y1, _, y2 = self.page_rect
        return (y2 - y1) / float(self.image_dims[1])

    def tile_page_rect(
        self, tile_x: int, tile_y: int, tile_size: int,
    ) -> tuple[float, float, float, float]:
        """Translate a tile's (x, y, size) in image coords -> page-pixel rectangle."""
        sx, sy = self.scale_x, self.scale_y
        x1 = self.page_rect[0] + tile_x * sx
        y1 = self.page_rect[1] + tile_y * sy
        x2 = self.page_rect[0] + (tile_x + tile_size) * sx
        y2 = self.page_rect[1] + (tile_y + tile_size) * sy
        return (x1, y1, x2, y2)


def parse_source_filename(
    filename: str,
    image_dims: tuple[int, int],
) -> SourceImageGeometry:
    """Resolve a Roboflow filename + image dims to a SourceImageGeometry.

    * "Subregion" filenames yield ``page_rect`` parsed from the filename.
    * "Full" filenames yield ``page_rect = (0, 0, image_w, image_h)`` —
      i.e. we assume the image's pixel coords *are* the page coords, which
      is the right thing to assume for a stand-alone full page.
    * Unparseable filenames fall back to (full image, page_key = stem).

    Important note: when the **same** page_key has BOTH full and sub-region
    source images (page "00" is the only such cluster in v61), the two
    coordinate systems are not directly comparable (the full image is at
    a different scale than the page implied by the sub-regions). In that
    case, the caller should collapse the whole page_key cluster into a
    single atom; the per-tile rectangle returned here only makes sense
    *within* a homogeneous cluster.
    """
    stem = _RF_SUFFIX.sub("", filename)
    m = _SUBREGION.match(stem)
    if m:
        return SourceImageGeometry(
            page_key=m.group("page"),
            page_rect=(
                int(m.group("x1")), int(m.group("y1")),
                int(m.group("x2")), int(m.group("y2")),
            ),
            image_dims=image_dims,
            kind="subregion",
        )
    m = _FULL.match(stem)
    if m:
        return SourceImageGeometry(
            page_key=m.group("page"),
            page_rect=(0, 0, image_dims[0], image_dims[1]),
            image_dims=image_dims,
            kind="full",
        )
    # Unparseable: use stem as cluster key, full image rectangle.
    return SourceImageGeometry(
        page_key=stem,
        page_rect=(0, 0, image_dims[0], image_dims[1]),
        image_dims=image_dims,
        kind="unparsed",
    )


def rectangles_overlap(
    r1: tuple[float, float, float, float],
    r2: tuple[float, float, float, float],
    *, eps: float = 1.0,
) -> bool:
    """Return True iff rectangles share more than `eps` pixels in each axis.

    Using `eps=1.0` prevents two tiles that merely *touch* at a single line
    of pixels from being flagged as overlapping. Set `eps=0` to require
    strict pixel separation.
    """
    return not (
        r1[2] <= r2[0] + eps
        or r2[2] <= r1[0] + eps
        or r1[3] <= r2[1] + eps
        or r2[3] <= r1[1] + eps
    )
