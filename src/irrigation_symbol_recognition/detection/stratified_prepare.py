"""End-to-end ``stratified_tile`` dataset preparation (v2).

Pipeline:

    1.  Pool the Roboflow ``train`` + ``test`` source images into one set.
    2.  Group them into clusters by ``page_key``.
    3.  Apply the irrigation class filter / remap to every source label
        file once (in memory).
    4.  Greedy class-coverage Set Cover -> pick a small set of clusters as
        the **deployment-test** set. Those clusters' source images are
        written out *unsliced*, full-resolution, into a
        ``deployment_test/`` subdirectory.
    5.  Slice the remaining clusters' source images into tiles using
        ``slicing.py`` (with the configured ``slice_size`` and ``overlap``)
        into a scratch directory, keeping per-tile metadata in memory.
    6.  Build the overlap graph in source-page pixel space; connected
        components = **atoms**.
    7.  Stratified assignment of atoms to ``train`` / ``valid`` / ``test``
        with per-class floors so every irrigation class has at least
        ``min_*_per_class`` atoms in both ``test`` and ``valid``.
    8.  Hard-link / move the tiles from scratch into the final per-split
        directories; write ``data.yaml``.
    9.  Run a battery of leakage-prevention asserts before returning.

Leakage guarantees:

    *   No tile in ``train/valid/test`` shares a ``page_key`` with any
        drawing in ``deployment_test/`` (clusters are partitioned).
    *   No tile in ``train`` shares a ``page_key``+overlapping-rectangle
        with any tile in ``valid`` or ``test`` (atoms are partitioned).
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import yaml

from ..data.yolo_dataset import YoloDataset
from ..utils.config import IrrigationConfig
from .atom_split import (
    TileInfo,
    build_atoms,
    select_deployment_clusters,
    stratified_assign,
    summarize_assignment,
)
from .source_geometry import SourceImageGeometry, parse_source_filename
from .slicing import slice_image_and_labels
from .yolo_filter import _build_policy, _FilterPolicy


logger = logging.getLogger(__name__)


@dataclass
class StratifiedPrepareConfig:
    slice_size: int
    overlap: float
    fit_box_to_slice: bool
    train_fraction: float
    valid_fraction: float
    test_fraction: float
    min_test_atoms_per_class: int
    min_valid_atoms_per_class: int
    deployment_test_max_clusters: int | None
    deployment_test_min_clusters_per_class: int
    seed: int
    symlink_images: bool


# =============================================================================
# Step 1-3: gather source images + apply irrigation class filter
# =============================================================================

def _source_image_paths(source_ds: YoloDataset) -> list[Path]:
    paths: list[Path] = []
    for split in sorted(source_ds.split_dirs):
        paths.extend(source_ds.split_images(split))
    return paths


def _filter_label_lines(
    label_path: Path,
    policy: _FilterPolicy,
) -> list[str]:
    """Return the filtered/remapped YOLO lines for a single source-image label file."""
    out: list[str] = []
    if not label_path.exists():
        return out
    bg_id = policy.background_dst_id
    for line in label_path.read_text().splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        try:
            src_id = int(parts[0])
        except ValueError:
            continue
        dst_id = policy.src_id_to_dst_id.get(src_id)
        if dst_id is None:
            if policy.policy == "background" and bg_id is not None:
                dst_id = bg_id
            else:
                continue
        out.append(f"{dst_id} {parts[1]}")
    return out


def _label_path_for(source_ds: YoloDataset, image_path: Path) -> Path:
    return source_ds.label_for(image_path)


# =============================================================================
# Step 5: slice non-deployment source images into scratch, capture metadata
# =============================================================================

@dataclass
class _SlicedTileRecord:
    scratch_image_path: Path
    scratch_label_path: Path
    info: TileInfo


def _slice_into_scratch(
    *,
    source_image_paths: Sequence[Path],
    source_ds: YoloDataset,
    geometries: dict[str, SourceImageGeometry],
    policy: _FilterPolicy,
    cfg: StratifiedPrepareConfig,
    scratch_dir: Path,
    cluster_geometries: dict[str, list[SourceImageGeometry]],
) -> list[_SlicedTileRecord]:
    """Slice every source image into ``scratch_dir`` and return tile metadata.

    A scratch directory layout of ``scratch_dir/images/`` and
    ``scratch_dir/labels/`` is used. After this returns, the caller decides
    which tile belongs to which final split and moves it accordingly.
    """
    scratch_images = scratch_dir / "images"
    scratch_labels = scratch_dir / "labels"
    scratch_images.mkdir(parents=True, exist_ok=True)
    scratch_labels.mkdir(parents=True, exist_ok=True)

    records: list[_SlicedTileRecord] = []
    next_index = 0

    for image_path in source_image_paths:
        geom = geometries[image_path.name]
        # First write the filtered label file to a temp location so that
        # slice_image_and_labels reads our remapped labels — not the
        # original 4541-class ones.
        filtered_label_lines = _filter_label_lines(
            _label_path_for(source_ds, image_path), policy,
        )
        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, dir=str(scratch_dir),
        ) as tmpf:
            tmpf.write("\n".join(filtered_label_lines))
            if filtered_label_lines:
                tmpf.write("\n")
            tmp_label_path = Path(tmpf.name)

        try:
            # Slice into a per-source-image subdirectory, then move tiles
            # to the flat scratch dir (avoids filename collisions when two
            # source images share the same stem prefix).
            with tempfile.TemporaryDirectory(
                prefix=image_path.stem + "_", dir=str(scratch_dir),
            ) as per_image_scratch:
                slice_image_and_labels(
                    image_path=image_path.resolve(),
                    label_path=tmp_label_path,
                    slice_size=cfg.slice_size,
                    output_dir=Path(per_image_scratch),
                    split_name="_slice",
                    overlap=cfg.overlap,
                    fit_box_to_slice=cfg.fit_box_to_slice,
                )
                img_subdir = Path(per_image_scratch) / "_slice" / "images"
                lbl_subdir = Path(per_image_scratch) / "_slice" / "labels"
                for tile_jpg in sorted(img_subdir.iterdir()):
                    tile_stem = tile_jpg.stem
                    # Parse tile origin (x, y) from filename suffix
                    # "<image_stem>_slice_<x>_<y>"
                    try:
                        rest = tile_stem.rsplit("_slice_", 1)[1]
                        tile_x_str, tile_y_str = rest.split("_")
                        tile_x, tile_y = int(tile_x_str), int(tile_y_str)
                    except (IndexError, ValueError):
                        logger.warning("Could not parse tile origin from %s", tile_stem)
                        continue
                    tile_label = lbl_subdir / (tile_stem + ".txt")
                    # Count classes in this tile (using our remapped IDs)
                    class_counts: Counter = Counter()
                    if tile_label.exists():
                        for line in tile_label.read_text().splitlines():
                            parts = line.strip().split()
                            if len(parts) != 5:
                                continue
                            try:
                                class_counts[int(parts[0])] += 1
                            except ValueError:
                                continue
                    # Move tile to flat scratch dir under a unique name
                    new_image_path = scratch_images / f"{next_index:08d}_{tile_jpg.name}"
                    new_label_path = scratch_labels / f"{next_index:08d}_{tile_jpg.stem}.txt"
                    shutil.move(str(tile_jpg), str(new_image_path))
                    if tile_label.exists():
                        shutil.move(str(tile_label), str(new_label_path))
                    else:
                        new_label_path.write_text("")

                    page_rect = geom.tile_page_rect(tile_x, tile_y, cfg.slice_size)
                    records.append(_SlicedTileRecord(
                        scratch_image_path=new_image_path,
                        scratch_label_path=new_label_path,
                        info=TileInfo(
                            index=next_index,
                            source_image=image_path.name,
                            page_key=geom.page_key,
                            tile_x=tile_x,
                            tile_y=tile_y,
                            tile_size=cfg.slice_size,
                            page_rect=page_rect,
                            class_counts=dict(class_counts),
                        ),
                    ))
                    next_index += 1
        finally:
            tmp_label_path.unlink(missing_ok=True)

    return records


# =============================================================================
# Step 4 + 8: deployment-test materialisation
# =============================================================================

def _materialize_deployment_split(
    *,
    deployment_image_paths: Sequence[Path],
    source_ds: YoloDataset,
    policy: _FilterPolicy,
    out_dir: Path,
    symlink_images: bool,
) -> dict:
    out_images = out_dir / "images"
    out_labels = out_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    per_class_counts: Counter = Counter()
    for image_path in deployment_image_paths:
        dst_image = out_images / image_path.name
        if dst_image.exists() or dst_image.is_symlink():
            dst_image.unlink()
        if symlink_images:
            dst_image.symlink_to(image_path.resolve())
        else:
            shutil.copy2(image_path, dst_image)
        filtered = _filter_label_lines(_label_path_for(source_ds, image_path), policy)
        for line in filtered:
            try:
                cid = int(line.split(maxsplit=1)[0])
                per_class_counts[cid] += 1
            except (IndexError, ValueError):
                continue
        out_label = out_labels / (image_path.stem + ".txt")
        out_label.write_text("\n".join(filtered) + ("\n" if filtered else ""))

    return {
        "num_drawings": len(deployment_image_paths),
        "annotations_per_class_id": dict(per_class_counts),
    }


# =============================================================================
# Step 8: materialise per-split tile directories from scratch records
# =============================================================================

def _materialize_tile_split(
    *,
    records: Sequence[_SlicedTileRecord],
    atoms: Sequence[Sequence[int]],
    assignment: Sequence[str],
    out_dir: Path,
) -> None:
    """Move scratch tiles into ``out_dir/<split>/{images,labels}``.

    Tiles are *derived* data living only in scratch, so we always move
    them (never symlink — that would break the moment scratch is cleaned
    up). ``symlink_images`` from the public config only affects
    deployment-test source images (which exist independently).
    """
    for split in ("train", "valid", "test"):
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    record_by_index = {r.info.index: r for r in records}
    for atom_idx, atom in enumerate(atoms):
        split = assignment[atom_idx]
        for t_idx in atom:
            r = record_by_index[t_idx]
            # Use the tile's natural (unprefixed) filename in the final split
            tile_name = r.scratch_image_path.name.split("_", 1)[1]
            dst_image = out_dir / split / "images" / tile_name
            dst_label = out_dir / split / "labels" / (Path(tile_name).stem + ".txt")
            if dst_image.exists() or dst_image.is_symlink():
                dst_image.unlink()
            shutil.move(str(r.scratch_image_path), str(dst_image))
            shutil.move(str(r.scratch_label_path), str(dst_label))


def _write_data_yaml(
    *,
    out_root: Path,
    label_names: Sequence[str],
    has_valid: bool,
    has_test: bool,
) -> None:
    data: dict = {
        "path": str(out_root.resolve()),
        "train": "train/images",
        "nc": len(label_names),
        "names": list(label_names),
    }
    if has_valid:
        data["val"] = "valid/images"
    elif has_test:
        data["val"] = "test/images"
    if has_test:
        data["test"] = "test/images"
    with (out_root / "data.yaml").open("w") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


# =============================================================================
# Leakage asserts
# =============================================================================

def _verify_no_leakage(
    *,
    out_dir: Path,
    deployment_image_dir: Path,
    cluster_to_split: dict[str, str],
) -> None:
    """Confirm the no-leakage invariants on the materialised outputs."""
    # Invariant 1: tile filenames in each split are disjoint.
    seen: dict[str, str] = {}  # tile_filename -> split_name
    for split in ("train", "valid", "test"):
        for img in (out_dir / split / "images").iterdir():
            if img.name in seen:
                raise AssertionError(
                    f"Leakage: tile {img.name} appears in both {seen[img.name]} and {split}"
                )
            seen[img.name] = split

    # Invariant 2: page_key of every tile must agree with cluster_to_split.
    for split in ("train", "valid", "test"):
        for img in (out_dir / split / "images").iterdir():
            # tile stem suffix is "_slice_<x>_<y>"; the part before "_slice_" is the
            # source image stem (with `.rf.<hash>` stripped already? not yet).
            source_stem = img.stem.rsplit("_slice_", 1)[0]
            # Re-derive page_key from source stem
            from .source_geometry import parse_source_filename
            geom = parse_source_filename(source_stem + ".jpg", (0, 0))
            expected = cluster_to_split.get(geom.page_key)
            if expected and expected != split:
                raise AssertionError(
                    f"Leakage: tile {img.name} is in {split} but page_key "
                    f"{geom.page_key!r} was assigned to {expected}"
                )

    # Invariant 3: no deployment image's page_key collides with any tile-split page_key.
    from .source_geometry import parse_source_filename
    deploy_keys = {
        parse_source_filename(p.name, (0, 0)).page_key
        for p in deployment_image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    for split in ("train", "valid", "test"):
        for img in (out_dir / split / "images").iterdir():
            source_stem = img.stem.rsplit("_slice_", 1)[0]
            geom = parse_source_filename(source_stem + ".jpg", (0, 0))
            if geom.page_key in deploy_keys:
                raise AssertionError(
                    f"Leakage: tile {img.name} is in {split} but its page_key "
                    f"{geom.page_key!r} is also used in deployment_test"
                )


# =============================================================================
# Public entry point
# =============================================================================

def _wipe_subdir(d: Path) -> None:
    """Remove ``d`` (and all its contents) if it exists. Defensive against
    leftover files from a previous run that would otherwise be misread as
    leakage by the verifier."""
    if d.exists():
        shutil.rmtree(d)


def prepare_stratified(
    *,
    source_ds: YoloDataset,
    irrigation_cfg: IrrigationConfig,
    out_dir: Path,
    deployment_out_dir: Path,
    cfg: StratifiedPrepareConfig,
) -> dict:
    """End-to-end v2 pipeline. Returns a summary dict.

    Nukes ``out_dir/{train,valid,test}`` and ``deployment_out_dir`` before
    writing anything so a previous run's contents can't be confused with
    leakage.
    """
    out_dir = Path(out_dir)
    deployment_out_dir = Path(deployment_out_dir)

    # ---- Step 0: clean any previous output so leakage checks are meaningful
    for sub in ("train", "valid", "test"):
        _wipe_subdir(out_dir / sub)
    if (out_dir / "data.yaml").exists():
        (out_dir / "data.yaml").unlink()
    _wipe_subdir(deployment_out_dir / "images")
    _wipe_subdir(deployment_out_dir / "labels")
    out_dir.mkdir(parents=True, exist_ok=True)
    deployment_out_dir.mkdir(parents=True, exist_ok=True)

    policy, final_labels, missing = _build_policy(source_ds.class_names, irrigation_cfg)
    if missing:
        logger.warning(
            "irrigation_classes references %d unknown class names: %s",
            len(missing), missing[:10] + (["..."] if len(missing) > 10 else []),
        )

    # ---- Step 1+2: pool sources and group by page_key -----------------------
    all_image_paths = _source_image_paths(source_ds)
    geometries: dict[str, SourceImageGeometry] = {}
    cluster_geometries: dict[str, list[SourceImageGeometry]] = defaultdict(list)
    cluster_images: dict[str, list[Path]] = defaultdict(list)
    cluster_class_counts: dict[str, Counter] = defaultdict(Counter)

    for image_path in all_image_paths:
        img = cv2.imread(str(image_path.resolve()))
        if img is None:
            logger.warning("Could not read %s; skipping", image_path)
            continue
        h, w = img.shape[:2]
        geom = parse_source_filename(image_path.name, (w, h))
        geometries[image_path.name] = geom
        cluster_geometries[geom.page_key].append(geom)
        cluster_images[geom.page_key].append(image_path)
        # Per-cluster class counts (used for deployment-test set cover):
        for line in _filter_label_lines(_label_path_for(source_ds, image_path), policy):
            try:
                cid = int(line.split(maxsplit=1)[0])
                cluster_class_counts[geom.page_key][cid] += 1
            except (IndexError, ValueError):
                continue

    logger.info(
        "Pooled %d source images across %d page_key clusters",
        len(all_image_paths), len(cluster_images),
    )

    # ---- Step 4: pick deployment-test clusters ------------------------------
    # We pick clusters across the *irrigation* label IDs (i.e. NOT
    # __background__), since deployment-test is about evaluating irrigation
    # detection.
    irrigation_label_ids = [
        i for i, name in enumerate(final_labels) if name != "__background__"
    ]
    deployment_clusters = select_deployment_clusters(
        class_counts_per_cluster=cluster_class_counts,
        irrigation_label_ids=irrigation_label_ids,
        min_clusters_per_class=cfg.deployment_test_min_clusters_per_class,
        max_clusters=cfg.deployment_test_max_clusters,
        seed=cfg.seed,
    )
    logger.info(
        "Deployment-test clusters: %d clusters comprising %d source drawings",
        len(deployment_clusters),
        sum(len(cluster_images[k]) for k in deployment_clusters),
    )

    deployment_image_paths = [
        p for k in deployment_clusters for p in cluster_images[k]
    ]
    deployment_report = _materialize_deployment_split(
        deployment_image_paths=deployment_image_paths,
        source_ds=source_ds,
        policy=policy,
        out_dir=deployment_out_dir,
        symlink_images=cfg.symlink_images,
    )

    # ---- Step 5: slice the non-deployment clusters into scratch -------------
    tile_pool_image_paths = [
        p for k, paths in cluster_images.items()
        if k not in deployment_clusters
        for p in paths
    ]
    logger.info("Slicing %d source drawings into tiles (overlap=%.2f, slice=%dpx)",
                len(tile_pool_image_paths), cfg.overlap, cfg.slice_size)

    # Put the scratch dir alongside the final output (same filesystem) so
    # `shutil.move` becomes a fast rename rather than a copy.
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="irrig_v2_scratch_", dir=str(out_dir.parent)) as scratch_str:
        scratch_dir = Path(scratch_str)
        records = _slice_into_scratch(
            source_image_paths=tile_pool_image_paths,
            source_ds=source_ds,
            geometries=geometries,
            policy=policy,
            cfg=cfg,
            scratch_dir=scratch_dir,
            cluster_geometries=cluster_geometries,
        )
        tiles = [r.info for r in records]
        logger.info("Sliced into %d tiles total", len(tiles))

        # ---- Step 6: build atoms -------------------------------------------
        atoms = build_atoms(
            tiles=tiles,
            cluster_geometries=cluster_geometries,
        )
        logger.info("Built %d atoms from %d tiles", len(atoms), len(tiles))
        atom_sizes = Counter(len(a) for a in atoms)
        logger.info(
            "Atom-size histogram (size: count): %s",
            dict(sorted(atom_sizes.items())[:20]),
        )

        # ---- Step 7: stratified assignment ---------------------------------
        target_fractions = {
            "train": cfg.train_fraction,
            "valid": cfg.valid_fraction,
            "test": cfg.test_fraction,
        }
        assignment = stratified_assign(
            atoms=atoms,
            tiles=tiles,
            irrigation_label_ids=irrigation_label_ids,
            target_fractions=target_fractions,
            min_test_atoms_per_class=cfg.min_test_atoms_per_class,
            min_valid_atoms_per_class=cfg.min_valid_atoms_per_class,
            seed=cfg.seed,
        )

        # ---- Step 8: materialise tile splits -------------------------------
        _materialize_tile_split(
            records=records,
            atoms=atoms,
            assignment=assignment,
            out_dir=out_dir,
        )

        # Build cluster->split mapping for the leakage verifier
        cluster_to_split: dict[str, str] = {}
        for atom_idx, atom in enumerate(atoms):
            split = assignment[atom_idx]
            for t_idx in atom:
                pk = tiles[t_idx].page_key
                if pk in cluster_to_split and cluster_to_split[pk] != split:
                    # A cluster ended up in multiple splits — that's fine
                    # for clusters that are NOT collapsed, but it means
                    # we can't use this mapping for verification.
                    cluster_to_split[pk] = "<multi>"
                else:
                    cluster_to_split[pk] = split

    # ---- Step 9: leakage asserts -------------------------------------------
    _write_data_yaml(
        out_root=out_dir,
        label_names=final_labels,
        has_valid=any(p.is_dir() for p in [out_dir / "valid" / "images"]),
        has_test=any(p.is_dir() for p in [out_dir / "test" / "images"]),
    )
    # Only run filename-based verification for clusters where the entire
    # cluster ended up in a single split (most of them).
    single_split_clusters = {
        k: v for k, v in cluster_to_split.items() if v != "<multi>"
    }
    _verify_no_leakage(
        out_dir=out_dir,
        deployment_image_dir=deployment_out_dir / "images",
        cluster_to_split=single_split_clusters,
    )

    # ---- Summary -----------------------------------------------------------
    summary = summarize_assignment(
        assignment=assignment,
        atoms=atoms,
        tiles=tiles,
        label_names=final_labels,
    )
    summary["label_names"] = list(final_labels)
    summary["deployment_test"] = deployment_report
    summary["deployment_test_clusters"] = sorted(deployment_clusters)
    summary["data_yaml"] = str(out_dir / "data.yaml")
    summary["missing_class_names"] = missing
    return summary
