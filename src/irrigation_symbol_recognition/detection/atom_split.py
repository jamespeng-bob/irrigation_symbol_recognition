"""Build no-pixel-leakage 'atoms' from tile metadata, and perform a
class-stratified split + deployment-test cluster selection.

Concepts
--------
* **Cluster** — set of source images sharing a ``page_key``. Two source
  images in the same cluster *may* share pixels (full page + sub-region
  variants of the same PDF page). Two source images in different clusters
  *never* share pixels.
* **Atom** — smallest indivisible unit for the train / valid / test split.
  Each atom is a set of tiles that must travel together to prevent
  pixel-level leakage. An atom is a connected component of the per-page
  tile-rectangle overlap graph.

The pipeline:

1. ``select_deployment_clusters(...)``: pick a small set of
   ``page_key`` clusters to be held out for drawing-level evaluation
   (greedy class-coverage Set Cover).
2. ``build_atoms(...)``: take the tiles from the non-deployment clusters
   and partition them into atoms — each atom is internally pixel-coupled,
   atoms are mutually pixel-disjoint.
3. ``stratified_assign(...)``: assign every atom to ``train`` / ``valid``
   / ``test`` so that each class has at least ``min_*_per_class`` atoms in
   the test and valid splits, while approximating the target fractions.
"""

from __future__ import annotations

import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from .source_geometry import SourceImageGeometry, rectangles_overlap


logger = logging.getLogger(__name__)


# =============================================================================
# Tile model
# =============================================================================

@dataclass(frozen=True)
class TileInfo:
    """All the metadata we need about a tile to split it leakage-safely."""

    index: int                                # stable index in the master list
    source_image: str                         # source-image filename
    page_key: str                             # cluster identifier
    tile_x: int                               # tile origin in image coords
    tile_y: int
    tile_size: int
    page_rect: tuple[float, float, float, float]  # rectangle in page coords
    class_counts: dict[int, int]              # final-label class id -> count


# =============================================================================
# Deployment-test selection (cluster-level)
# =============================================================================

def select_deployment_clusters(
    *,
    class_counts_per_cluster: dict[str, Counter],
    irrigation_label_ids: Sequence[int],
    min_clusters_per_class: int = 1,
    max_clusters: int | None = None,
    seed: int = 42,
) -> set[str]:
    """Greedy class-coverage set-cover over page_key clusters.

    Picks the cluster covering the most uncovered classes per iteration;
    stops when every irrigation class is covered by ``min_clusters_per_class``
    deployment clusters (or when ``max_clusters`` is reached).

    Returns the set of ``page_key`` strings chosen for deployment-test.
    """
    rng = random.Random(seed)
    coverage: dict[int, int] = {c: 0 for c in irrigation_label_ids}

    def fully_covered() -> bool:
        return all(coverage[c] >= min_clusters_per_class for c in irrigation_label_ids
                   if any(class_counts_per_cluster[k].get(c, 0) > 0
                          for k in class_counts_per_cluster))

    selected: set[str] = set()
    cluster_keys = list(class_counts_per_cluster.keys())
    rng.shuffle(cluster_keys)   # tie-break determinism

    while not fully_covered() and (max_clusters is None or len(selected) < max_clusters):
        best_key, best_gain, best_cluster_size = None, -1, None
        for k in cluster_keys:
            if k in selected:
                continue
            counts = class_counts_per_cluster[k]
            gain = sum(
                1 for c in irrigation_label_ids
                if coverage[c] < min_clusters_per_class and counts.get(c, 0) > 0
            )
            cluster_size = sum(counts.values())
            if gain > best_gain or (
                gain == best_gain and (best_cluster_size is None or cluster_size < best_cluster_size)
            ):
                best_key, best_gain, best_cluster_size = k, gain, cluster_size
        if best_key is None or best_gain <= 0:
            break  # no improvement possible
        selected.add(best_key)
        for c in irrigation_label_ids:
            if class_counts_per_cluster[best_key].get(c, 0) > 0:
                coverage[c] += 1

    uncovered = [c for c in irrigation_label_ids if coverage[c] < min_clusters_per_class]
    if uncovered:
        logger.warning(
            "Could not reach min_clusters_per_class=%d for %d classes: %s",
            min_clusters_per_class, len(uncovered), uncovered,
        )

    return selected


# =============================================================================
# Atom construction (overlap-aware connected components)
# =============================================================================

class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def build_atoms(
    tiles: Sequence[TileInfo],
    *,
    cluster_geometries: dict[str, list[SourceImageGeometry]],
) -> list[list[int]]:
    """Group tile indices into pixel-disjoint atoms.

    Within a cluster (same page_key), two tiles are connected iff their
    page rectangles overlap. Different clusters never share an atom.

    Special case: if a cluster's source images use *different* coordinate
    systems (e.g. both a full-page image AND a sub-region exist for the
    same page_key, at different scales), the per-tile rectangles can't be
    compared meaningfully. We collapse the whole cluster into one atom in
    that case — conservative, but correct.
    """
    by_cluster: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(tiles):
        by_cluster[t.page_key].append(i)

    atoms: list[list[int]] = []
    for page_key, indices in by_cluster.items():
        geoms = cluster_geometries.get(page_key, [])
        kinds = {g.kind for g in geoms}
        if "full" in kinds and "subregion" in kinds:
            # Mixed scales — collapse the whole cluster.
            atoms.append(indices)
            continue
        # Multiple "full" source images of the same page at possibly different
        # downscales also can't safely be compared — collapse.
        if "full" in kinds and len({g.image_dims for g in geoms}) > 1:
            atoms.append(indices)
            continue

        uf = _UnionFind(len(indices))
        for i_local in range(len(indices)):
            ri = tiles[indices[i_local]].page_rect
            for j_local in range(i_local + 1, len(indices)):
                rj = tiles[indices[j_local]].page_rect
                if rectangles_overlap(ri, rj):
                    uf.union(i_local, j_local)

        groups: dict[int, list[int]] = defaultdict(list)
        for i_local, tile_index in enumerate(indices):
            groups[uf.find(i_local)].append(tile_index)
        atoms.extend(groups.values())

    return atoms


# =============================================================================
# Stratified atom assignment
# =============================================================================

def _per_atom_class_set(atoms: Sequence[Sequence[int]], tiles: Sequence[TileInfo]) -> list[set[int]]:
    out: list[set[int]] = []
    for atom in atoms:
        cs: set[int] = set()
        for t_idx in atom:
            cs.update(tiles[t_idx].class_counts.keys())
        out.append(cs)
    return out


def stratified_assign(
    *,
    atoms: Sequence[Sequence[int]],
    tiles: Sequence[TileInfo],
    irrigation_label_ids: Sequence[int],
    target_fractions: dict[str, float],
    min_test_atoms_per_class: int,
    min_valid_atoms_per_class: int,
    seed: int = 42,
) -> list[str]:
    """Return an assignment list (len == len(atoms)) of "train"/"valid"/"test".

    Algorithm (deterministic given the seed):

    1. Walk the irrigation classes from rare to common; for each class,
       reserve up to ``min_test_atoms_per_class`` atoms (containing the
       class) into ``test``, preferring atoms that also cover other rare
       classes (a Set-Cover-flavoured heuristic, with weights = number of
       still-uncovered rare classes the atom carries).
    2. Repeat for ``valid`` with ``min_valid_atoms_per_class``.
    3. Distribute remaining atoms to hit ``target_fractions`` (largest
       remainder for ``train``).

    Class coverage minima take priority over fraction targets — if we
    can't reach the per-class floor, we warn but continue.
    """
    n = len(atoms)
    if n == 0:
        return []

    assignment: list[str | None] = [None] * n
    per_atom_classes = _per_atom_class_set(atoms, tiles)
    rng = random.Random(seed)

    # ---- Phase 1: floor for `test` ------------------------------------------
    _enforce_per_class_floor(
        target="test",
        floor=min_test_atoms_per_class,
        assignment=assignment,
        per_atom_classes=per_atom_classes,
        irrigation_label_ids=irrigation_label_ids,
        rng=rng,
    )

    # ---- Phase 2: floor for `valid` -----------------------------------------
    _enforce_per_class_floor(
        target="valid",
        floor=min_valid_atoms_per_class,
        assignment=assignment,
        per_atom_classes=per_atom_classes,
        irrigation_label_ids=irrigation_label_ids,
        rng=rng,
    )

    # ---- Phase 3: distribute remaining atoms by fraction -------------------
    unassigned = [i for i in range(n) if assignment[i] is None]
    rng.shuffle(unassigned)

    counts = Counter(a for a in assignment if a is not None)
    target_test = int(round(n * target_fractions.get("test", 0.10)))
    target_valid = int(round(n * target_fractions.get("valid", 0.10)))
    need_test = max(0, target_test - counts["test"])
    need_valid = max(0, target_valid - counts["valid"])

    take_test = unassigned[:need_test]
    take_valid = unassigned[need_test:need_test + need_valid]
    take_train = unassigned[need_test + need_valid:]

    for i in take_test:
        assignment[i] = "test"
    for i in take_valid:
        assignment[i] = "valid"
    for i in take_train:
        assignment[i] = "train"

    # Sanity: no Nones left.
    assert all(a is not None for a in assignment), "stratified_assign left atoms unassigned"
    return [a for a in assignment]  # type: ignore[list-item]


def _enforce_per_class_floor(
    *,
    target: str,
    floor: int,
    assignment: list[str | None],
    per_atom_classes: list[set[int]],
    irrigation_label_ids: Sequence[int],
    rng: random.Random,
) -> None:
    if floor <= 0:
        return

    n_in_target_per_class: Counter = Counter()
    for i, a in enumerate(assignment):
        if a != target:
            continue
        for c in per_atom_classes[i]:
            n_in_target_per_class[c] += 1

    # Rare first
    class_order = sorted(
        irrigation_label_ids,
        key=lambda c: sum(1 for cs in per_atom_classes if c in cs),
    )
    for c in class_order:
        need = max(0, floor - n_in_target_per_class[c])
        if need == 0:
            continue
        candidates = [
            i for i, cs in enumerate(per_atom_classes)
            if c in cs and assignment[i] is None
        ]
        if not candidates:
            logger.warning(
                "Cannot reach %s floor (%d) for class %d: no unassigned atoms left containing it",
                target, floor, c,
            )
            continue
        # Score each candidate by how many *other rare* classes are still
        # under-floored in this target — pick atoms that cover multiple rare
        # classes at once.
        deficit = {
            cc: max(0, floor - n_in_target_per_class[cc])
            for cc in irrigation_label_ids
        }
        def _score(idx: int) -> int:
            cs = per_atom_classes[idx]
            return sum(1 for cc in cs if deficit.get(cc, 0) > 0)

        # Tie-break randomly (deterministic via rng).
        rng.shuffle(candidates)
        candidates.sort(key=_score, reverse=True)

        taken = 0
        for i in candidates:
            if taken >= need:
                break
            assignment[i] = target
            taken += 1
            for cc in per_atom_classes[i]:
                n_in_target_per_class[cc] += 1
        if taken < need:
            logger.warning(
                "Cannot reach %s floor (%d) for class %d: only %d atoms available",
                target, floor, c, taken,
            )


# =============================================================================
# Diagnostics / reporting
# =============================================================================

def summarize_assignment(
    *,
    assignment: Sequence[str],
    atoms: Sequence[Sequence[int]],
    tiles: Sequence[TileInfo],
    label_names: Sequence[str],
) -> dict:
    """Return a structured summary suitable for logging / JSON dumping."""
    per_split_class_counts: dict[str, Counter] = {
        "train": Counter(), "valid": Counter(), "test": Counter(),
    }
    per_split_atom_count: Counter = Counter()
    per_split_tile_count: Counter = Counter()
    for atom_idx, atom in enumerate(atoms):
        split = assignment[atom_idx]
        per_split_atom_count[split] += 1
        for t_idx in atom:
            per_split_tile_count[split] += 1
            for c, n in tiles[t_idx].class_counts.items():
                per_split_class_counts[split][c] += n

    return {
        "atoms_per_split": dict(per_split_atom_count),
        "tiles_per_split": dict(per_split_tile_count),
        "instances_per_split_per_class": {
            split: {label_names[c]: per_split_class_counts[split].get(c, 0)
                    for c in range(len(label_names))}
            for split in ("train", "valid", "test")
        },
    }
