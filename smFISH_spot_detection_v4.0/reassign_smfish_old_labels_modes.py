#!/usr/bin/env python3
"""smFISH reassignment using the existing (old StarDist) projection labels.

This script is intentionally separate from the future Cellpose true-z pipeline.
It reads ``results/labels/Nucleus_Labels_*.tif`` and never reads Cellpose masks.
``--link-mode none`` treats every label in every projection as an independent
cell, matching the old notebook. ``--link-mode reciprocal`` links reciprocal
best-overlap labels in adjacent projections. ``--benchmark-linking`` measures
the proposed links and saves review tables plus red/green label-overlay crops.

For assignment, each old label image is expanded once and spots are selected by
``z // projection_size``. The output includes every segmented label/cell,
including labels with zero assigned transcripts.

Use this version only for the current first-pass 304/306/307/314/332 analysis.
In addition to the original positive-only table and QC files, it writes wide
tables containing every segmented cell, including cells with zero transcripts.
The output records the selected old-label backend so no-link, linked, and later
Cellpose-based analyses cannot be confused.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

try:
    from skimage.segmentation import expand_labels as _skimage_expand_labels
except ImportError:
    _skimage_expand_labels = None

try:
    from tifffile import imread as _tifffile_imread
    from tifffile import imwrite as _tifffile_imwrite
except ImportError:
    _tifffile_imread = None
    _tifffile_imwrite = None


SCRIPT_VERSION = "1.2.0-old-projection-label-modes"
COMPATIBLE_SAMPLE_VERSIONS = {SCRIPT_VERSION}
LOGGER = logging.getLogger("smfish_projection_reassignment")


DEFAULT_EXPERIMENTS = (
    (
        "304",
        "wnt1",
        "/Volumes/Backup Plus/Experiment_results/304_Analysis_results/Experiment",
    ),
    (
        "306",
        "wnt1",
        "/Volumes/Backup Plus/Experiment_results/306_analysis_results_alter/Experiment",
    ),
    (
        "307",
        "wnt1",
        "/Volumes/Backup Plus/Experiment_results/307_analysis_resutls/307_quantificatoin/Experiment",
    ),
    (
        "314",
        "notum",
        "/Volumes/Backup Plus/Experiment_results/314_analysis_results/Experiment",
    ),
    (
        "332",
        "notum",
        "/Volumes/Backup Plus/Experiment_results/332_notum_analysis/Experiment",
    ),
)


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    gene: str
    root: Path


@dataclass(frozen=True)
class Parameters:
    projection_size: int
    expansion_distance: float
    min_smaller_overlap: float
    min_iou: float
    channel_folder: str
    save_assignment_maps: bool
    link_mode: str
    benchmark_linking: bool
    benchmark_max_crops: int


@dataclass(frozen=True)
class SampleSpec:
    experiment: str
    gene: str
    experiment_dir: Path
    condition: str
    animal: str
    sample_dir: Path
    results_dir: Path

    @property
    def sample_id(self) -> str:
        return f"{self.experiment}__{self.condition}__{self.animal}"


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[tuple[int, int], tuple[int, int]] = {}
        self.rank: dict[tuple[int, int], int] = {}

    def add(self, item: tuple[int, int]) -> None:
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0

    def find(self, item: tuple[int, int]) -> tuple[int, int]:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, first: tuple[int, int], second: tuple[int, int]) -> None:
        self.add(first)
        self.add(second)
        root_a = self.find(first)
        root_b = self.find(second)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("PIL").setLevel(logging.WARNING)


def imread(path: str | Path) -> np.ndarray:
    if _tifffile_imread is not None:
        return np.asarray(_tifffile_imread(path))
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install tifffile to read nuclear-label TIFF files") from exc
    return np.asarray(Image.open(path))


def imwrite(path: str | Path, array: np.ndarray, **kwargs: object) -> None:
    if _tifffile_imwrite is not None:
        _tifffile_imwrite(path, array, **kwargs)
        return
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install tifffile to write assignment TIFF files") from exc
    Image.fromarray(np.asarray(array)).save(path)


def expand_labels(labels: np.ndarray, distance: float) -> np.ndarray:
    if distance <= 0:
        return labels.copy()
    if _skimage_expand_labels is not None:
        return _skimage_expand_labels(labels, distance=distance)
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install scikit-image or scipy for label expansion") from exc
    distances, nearest = distance_transform_edt(
        labels == 0, return_distances=True, return_indices=True
    )
    expanded = labels.copy()
    mask = (labels == 0) & (distances <= distance)
    expanded[mask] = labels[tuple(nearest)][mask]
    return expanded


def remove_output_tree(path: Path) -> None:
    """Tolerate disappearing macOS AppleDouble files on the external drive."""
    def ignore_disappeared_file(function, target, exc_info) -> None:
        error = exc_info[1]
        if isinstance(error, FileNotFoundError):
            LOGGER.debug("Ignoring file that disappeared during cleanup: %s", target)
            return
        raise error

    try:
        shutil.rmtree(path, onerror=ignore_disappeared_file)
    except FileNotFoundError:
        pass
    if path.exists():
        raise OSError(f"Could not fully remove prior output directory: {path}")


def natural_key(path_or_name: str | Path) -> list[object]:
    return [
        int(piece) if piece.isdigit() else piece.lower()
        for piece in re.split(r"(\d+)", Path(path_or_name).name)
    ]


def parse_experiment_argument(value: str) -> ExperimentSpec:
    parts = value.split(",", 2)
    if len(parts) != 3 or not all(part.strip() for part in parts):
        raise argparse.ArgumentTypeError(
            "--experiment must be NAME,GENE,ROOT_OR_EXPERIMENT_DIR"
        )
    name, gene, root = (part.strip() for part in parts)
    return ExperimentSpec(name=name, gene=gene, root=Path(root))


def resolve_experiment_dir(root: Path) -> Path:
    if root.name.lower() == "experiment":
        if root.is_dir():
            return root
        raise FileNotFoundError(f"Experiment directory does not exist: {root}")
    candidate = root / "Experiment"
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(
        f"Expected an Experiment directory or a parent containing one: {root}"
    )


def discover_samples(spec: ExperimentSpec, channel_folder: str) -> list[SampleSpec]:
    try:
        experiment_dir = resolve_experiment_dir(spec.root)
    except FileNotFoundError as exc:
        LOGGER.warning("%s", exc)
        return []

    samples: list[SampleSpec] = []
    for condition_dir in sorted(experiment_dir.iterdir(), key=natural_key):
        if not condition_dir.is_dir() or condition_dir.name.startswith("."):
            continue
        for animal_dir in sorted(condition_dir.iterdir(), key=natural_key):
            if not animal_dir.is_dir() or animal_dir.name.startswith("."):
                continue
            results_dir = animal_dir / channel_folder / "results"
            if results_dir.is_dir():
                samples.append(
                    SampleSpec(
                        experiment=spec.name,
                        gene=spec.gene,
                        experiment_dir=experiment_dir,
                        condition=condition_dir.name,
                        animal=animal_dir.name,
                        sample_dir=animal_dir,
                        results_dir=results_dir,
                    )
                )
    return samples


LABEL_PATTERN = re.compile(r"Nucleus_Labels_(\d+)\.(?:tif|tiff)$", re.IGNORECASE)


def locate_label_files(results_dir: Path) -> list[Path]:
    label_dir = results_dir / "labels"
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Existing label directory is missing: {label_dir}")
    indexed: list[tuple[int, Path]] = []
    for path in label_dir.iterdir():
        if path.name.startswith(".") or not path.is_file():
            continue
        match = LABEL_PATTERN.fullmatch(path.name)
        if match:
            indexed.append((int(match.group(1)), path))
    indexed.sort(key=lambda item: item[0])
    if not indexed:
        fallback = sorted(
            (
                path
                for path in label_dir.iterdir()
                if path.is_file()
                and not path.name.startswith(".")
                and path.suffix.lower() in {".tif", ".tiff"}
            ),
            key=natural_key,
        )
        if not fallback:
            raise FileNotFoundError(f"No TIFF labels found in {label_dir}")
        LOGGER.warning("Using all TIFF files in nonstandard label folder: %s", label_dir)
        return fallback
    indices = [item[0] for item in indexed]
    if indices != list(range(indices[0], indices[0] + len(indices))):
        raise ValueError(f"Noncontiguous projection-label indices in {label_dir}: {indices}")
    return [item[1] for item in indexed]


def locate_spot_file(results_dir: Path) -> Path:
    for filename in (
        "spots_post_decomposition_and_background_removed.npy",
        "spots_post_decomposition.npy",
    ):
        candidate = results_dir / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No post-decomposition spot file in {results_dir}")


def read_label(path: Path) -> np.ndarray:
    labels = np.squeeze(np.asarray(imread(path)))
    if labels.ndim != 2:
        raise ValueError(f"Expected a 2-D label image at {path}; got {labels.shape}")
    if not np.issubdtype(labels.dtype, np.integer):
        if not np.all(labels == np.floor(labels)):
            raise ValueError(f"Non-integer label values in {path}")
    labels = labels.astype(np.int64, copy=False)
    if np.any(labels < 0):
        raise ValueError(f"Negative labels in {path}")
    return labels


def load_spots(path: Path) -> np.ndarray:
    spots = np.asarray(np.load(path, allow_pickle=False))
    if spots.ndim != 2 or spots.shape[1] < 3:
        raise ValueError(f"Expected (n, >=3) spot coordinates at {path}; got {spots.shape}")
    coordinates = spots[:, [0, -2, -1]].astype(np.float64, copy=False)
    if not np.all(np.isfinite(coordinates)):
        raise ValueError(f"NaN/inf spot coordinates in {path}")
    return coordinates


def label_areas(labels: np.ndarray) -> np.ndarray:
    return np.bincount(labels.ravel(), minlength=int(labels.max()) + 1)


def label_centroids(labels: np.ndarray, areas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-label y/x centroids without allocating full coordinate grids."""
    ys, xs = np.nonzero(labels)
    label_ids = labels[ys, xs]
    minlength = len(areas)
    sum_y = np.bincount(label_ids, weights=ys, minlength=minlength)
    sum_x = np.bincount(label_ids, weights=xs, minlength=minlength)
    centroid_y = np.full(minlength, np.nan, dtype=np.float64)
    centroid_x = np.full(minlength, np.nan, dtype=np.float64)
    present = areas > 0
    centroid_y[present] = sum_y[present] / areas[present]
    centroid_x[present] = sum_x[present] / areas[present]
    return centroid_y, centroid_x


def reciprocal_best_links(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    projection_a: int,
    min_smaller_overlap: float,
    min_iou: float,
) -> list[dict[str, object]]:
    if labels_a.shape != labels_b.shape:
        raise ValueError("Adjacent projection labels have different shapes")
    overlap = (labels_a > 0) & (labels_b > 0)
    if not np.any(overlap):
        return []

    a_values = labels_a[overlap]
    b_values = labels_b[overlap]
    base = int(labels_b.max()) + 1
    codes, intersections = np.unique(a_values * base + b_values, return_counts=True)
    a_ids = codes // base
    b_ids = codes % base
    areas_a = label_areas(labels_a)
    areas_b = label_areas(labels_b)
    area_a = areas_a[a_ids]
    area_b = areas_b[b_ids]
    overlap_a = intersections / area_a
    overlap_b = intersections / area_b
    smaller_overlap = intersections / np.minimum(area_a, area_b)
    ious = intersections / (areas_a[a_ids] + areas_b[b_ids] - intersections)
    area_ratios = np.minimum(area_a, area_b) / np.maximum(area_a, area_b)
    centroid_y_a, centroid_x_a = label_centroids(labels_a, areas_a)
    centroid_y_b, centroid_x_b = label_centroids(labels_b, areas_b)
    centroid_distances = np.hypot(
        centroid_y_a[a_ids] - centroid_y_b[b_ids],
        centroid_x_a[a_ids] - centroid_x_b[b_ids],
    )
    normalized_shifts = centroid_distances / np.sqrt(np.minimum(area_a, area_b))

    best_for_a: dict[int, tuple[int, int]] = {}
    best_for_b: dict[int, tuple[int, int]] = {}
    for a_id, b_id, intersection in zip(a_ids, b_ids, intersections, strict=True):
        a_int, b_int, n_int = int(a_id), int(b_id), int(intersection)
        current_a = best_for_a.get(a_int)
        if current_a is None or (n_int, -b_int) > (current_a[1], -current_a[0]):
            best_for_a[a_int] = (b_int, n_int)
        current_b = best_for_b.get(b_int)
        if current_b is None or (n_int, -a_int) > (current_b[1], -current_b[0]):
            best_for_b[b_int] = (a_int, n_int)

    links: list[dict[str, object]] = []
    for (
        a_id,
        b_id,
        intersection,
        fraction_a,
        fraction_b,
        overlap_fraction,
        iou,
        pixels_a,
        pixels_b,
        area_ratio,
        centroid_distance,
        normalized_shift,
    ) in zip(
        a_ids,
        b_ids,
        intersections,
        overlap_a,
        overlap_b,
        smaller_overlap,
        ious,
        area_a,
        area_b,
        area_ratios,
        centroid_distances,
        normalized_shifts,
        strict=True,
    ):
        a_int, b_int = int(a_id), int(b_id)
        reciprocal = (
            best_for_a[a_int][0] == b_int and best_for_b[b_int][0] == a_int
        )
        threshold = float(overlap_fraction) >= min_smaller_overlap or float(iou) >= min_iou
        threshold_scores = []
        if min_smaller_overlap > 0:
            threshold_scores.append(float(overlap_fraction) / min_smaller_overlap)
        if min_iou > 0:
            threshold_scores.append(float(iou) / min_iou)
        threshold_score = max(threshold_scores, default=math.inf)
        suspicious_accepted = bool(
            float(iou) < 0.25
            or float(area_ratio) < 0.50
            or float(normalized_shift) > 0.75
        )
        near_threshold_rejected = bool(not threshold and threshold_score >= 0.75)
        if reciprocal:
            links.append(
                {
                    "projection_a": projection_a,
                    "label_a": a_int,
                    "projection_b": projection_a + 1,
                    "label_b": b_int,
                    "intersection_pixels": int(intersection),
                    "area_a_pixels": int(pixels_a),
                    "area_b_pixels": int(pixels_b),
                    "overlap_a_fraction": float(fraction_a),
                    "overlap_b_fraction": float(fraction_b),
                    "smaller_overlap_fraction": float(overlap_fraction),
                    "iou": float(iou),
                    "area_ratio": float(area_ratio),
                    "centroid_distance_pixels": float(centroid_distance),
                    "centroid_shift_over_sqrt_smaller_area": float(normalized_shift),
                    "accepted_by_threshold": bool(threshold),
                    "threshold_score": float(threshold_score),
                    "review_flag": bool(suspicious_accepted or near_threshold_rejected),
                    "review_reason": (
                        "accepted_suspicious"
                        if threshold and suspicious_accepted
                        else "rejected_near_threshold"
                        if near_threshold_rejected
                        else "accepted_typical"
                        if threshold
                        else "rejected_low_overlap"
                    ),
                }
            )
    return links


def build_cells(
    label_files: Sequence[Path],
    min_smaller_overlap: float,
    min_iou: float,
    link_mode: str,
    benchmark_linking: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    union_find = UnionFind()
    node_rows: list[dict[str, int]] = []
    link_rows: list[dict[str, object]] = []
    expected_shape: tuple[int, int] | None = None
    previous: np.ndarray | None = None
    compute_links = link_mode == "reciprocal" or benchmark_linking

    for projection, path in enumerate(label_files):
        labels = read_label(path)
        if expected_shape is None:
            expected_shape = labels.shape
        elif labels.shape != expected_shape:
            raise ValueError(f"Label shape mismatch at {path}: {labels.shape}")
        areas = label_areas(labels)
        for local_label in np.flatnonzero(areas[1:]) + 1:
            node = (projection, int(local_label))
            union_find.add(node)
            node_rows.append(
                {
                    "projection": projection,
                    "local_label": int(local_label),
                    "nuclear_area_pixels": int(areas[local_label]),
                }
            )
        if previous is not None and compute_links:
            links = reciprocal_best_links(
                previous,
                labels,
                projection - 1,
                min_smaller_overlap,
                min_iou,
            )
            link_rows.extend(links)
            for link in links:
                if bool(link["accepted_by_threshold"]):
                    union_find.union(
                        (int(link["projection_a"]), int(link["label_a"])),
                        (int(link["projection_b"]), int(link["label_b"])),
                    )
        previous = labels

    linked_roots = sorted({union_find.find(node) for node in union_find.parent})
    root_to_linked_cell = {root: index + 1 for index, root in enumerate(linked_roots)}
    node_to_linked_cell = {
        node: root_to_linked_cell[union_find.find(node)] for node in union_find.parent
    }
    node_df = pd.DataFrame(node_rows)
    if node_df.empty:
        raise ValueError("No nuclei were found in the projection labels")
    independent_nodes = sorted(union_find.parent)
    node_to_independent_cell = {
        node: index + 1 for index, node in enumerate(independent_nodes)
    }
    node_df["independent_cell_id"] = [
        node_to_independent_cell[(int(row.projection), int(row.local_label))]
        for row in node_df.itertuples(index=False)
    ]
    node_df["linked_cell_id"] = [
        node_to_linked_cell[(int(row.projection), int(row.local_label))]
        for row in node_df.itertuples(index=False)
    ]
    node_df["global_cell_id"] = (
        node_df["independent_cell_id"]
        if link_mode == "none"
        else node_df["linked_cell_id"]
    )
    link_df = pd.DataFrame(
        link_rows,
        columns=[
            "projection_a",
            "label_a",
            "projection_b",
            "label_b",
            "intersection_pixels",
            "area_a_pixels",
            "area_b_pixels",
            "overlap_a_fraction",
            "overlap_b_fraction",
            "smaller_overlap_fraction",
            "iou",
            "area_ratio",
            "centroid_distance_pixels",
            "centroid_shift_over_sqrt_smaller_area",
            "accepted_by_threshold",
            "threshold_score",
            "review_flag",
            "review_reason",
        ],
    )
    if not link_df.empty:
        link_df.insert(0, "link_id", np.arange(1, len(link_df) + 1))
        link_df["applied_to_assignment"] = (
            link_df["accepted_by_threshold"] & (link_mode == "reciprocal")
        )
        link_df["manual_same_cell"] = ""
        link_df["manual_notes"] = ""

    linked_spans = (
        node_df.groupby("linked_cell_id")["projection"].nunique().to_numpy(dtype=np.int64)
    )
    raw_labels = int(len(node_df))
    linked_cells = int(node_df["linked_cell_id"].nunique())
    accepted_links = (
        link_df.loc[link_df["accepted_by_threshold"]].copy()
        if not link_df.empty
        else link_df.copy()
    )
    benchmark = {
        "link_mode_used": link_mode,
        "raw_projection_label_instances": raw_labels,
        "reciprocal_candidate_pairs": int(len(link_df)),
        "proposed_reciprocal_links": int(len(accepted_links)),
        "independent_cells_if_no_link": raw_labels,
        "cells_if_reciprocal_links_applied": linked_cells,
        "cell_count_reduction_if_linked": raw_labels - linked_cells,
        "cell_count_reduction_fraction_if_linked": (
            (raw_labels - linked_cells) / raw_labels if raw_labels else math.nan
        ),
        "linked_cells_spanning_multiple_projections": int(np.sum(linked_spans > 1)),
        "linked_cells_spanning_more_than_three_projections": int(np.sum(linked_spans > 3)),
        "maximum_projection_span_if_linked": int(np.max(linked_spans)),
        "review_flagged_candidates": int(link_df["review_flag"].sum()) if not link_df.empty else 0,
        "median_iou": (
            float(accepted_links["iou"].median()) if not accepted_links.empty else math.nan
        ),
        "median_smaller_overlap_fraction": (
            float(accepted_links["smaller_overlap_fraction"].median())
            if not accepted_links.empty
            else math.nan
        ),
        "median_area_ratio": (
            float(accepted_links["area_ratio"].median())
            if not accepted_links.empty
            else math.nan
        ),
        "median_centroid_shift_over_sqrt_smaller_area": (
            float(accepted_links["centroid_shift_over_sqrt_smaller_area"].median())
            if not accepted_links.empty
            else math.nan
        ),
    }
    return node_df, link_df, benchmark


def save_link_review_crops(
    label_files: Sequence[Path],
    link_df: pd.DataFrame,
    sample_output_dir: Path,
    maximum_crops: int,
) -> pd.DataFrame:
    """Save deterministic red/green overlays for manual link review.

    Red is the label in projection A, green is the proposed label in projection
    B, and yellow is overlap. Flagged/weak links are selected first; remaining
    capacity is filled with links spread across the full table.
    """
    if link_df.empty or maximum_crops <= 0:
        review = link_df.copy()
        if not review.empty:
            review["review_crop"] = ""
        return review

    ranked = link_df.sort_values(
        [
            "review_flag",
            "iou",
            "area_ratio",
            "centroid_shift_over_sqrt_smaller_area",
            "link_id",
        ],
        ascending=[False, True, True, False, True],
    )
    chosen_ids: list[int] = []
    flagged = ranked.loc[ranked["review_flag"], "link_id"].astype(int).tolist()
    chosen_ids.extend(flagged[:maximum_crops])
    remaining = maximum_crops - len(chosen_ids)
    if remaining > 0:
        candidates = ranked.loc[~ranked["link_id"].isin(chosen_ids), "link_id"].astype(int)
        if len(candidates) <= remaining:
            chosen_ids.extend(candidates.tolist())
        elif remaining == 1:
            chosen_ids.append(int(candidates.iloc[len(candidates) // 2]))
        else:
            positions = np.linspace(0, len(candidates) - 1, remaining).round().astype(int)
            chosen_ids.extend(candidates.iloc[positions].astype(int).tolist())

    review = link_df.loc[link_df["link_id"].isin(chosen_ids)].copy()
    review = review.sort_values(
        ["review_flag", "iou", "area_ratio", "link_id"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    crop_dir = sample_output_dir / "link_benchmark_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    cached_labels: dict[int, np.ndarray] = {}
    crop_names: list[str] = []

    for row in review.itertuples(index=False):
        projection_a = int(row.projection_a)
        projection_b = int(row.projection_b)
        if projection_a not in cached_labels:
            cached_labels[projection_a] = read_label(label_files[projection_a])
        if projection_b not in cached_labels:
            cached_labels[projection_b] = read_label(label_files[projection_b])
        mask_a = cached_labels[projection_a] == int(row.label_a)
        mask_b = cached_labels[projection_b] == int(row.label_b)
        union = mask_a | mask_b
        ys, xs = np.nonzero(union)
        if len(ys) == 0:
            crop_names.append("")
            continue
        margin = 10
        y0 = max(0, int(ys.min()) - margin)
        y1 = min(union.shape[0], int(ys.max()) + margin + 1)
        x0 = max(0, int(xs.min()) - margin)
        x1 = min(union.shape[1], int(xs.max()) + margin + 1)
        crop_a = mask_a[y0:y1, x0:x1]
        crop_b = mask_b[y0:y1, x0:x1]
        overlay = np.zeros((*crop_a.shape, 3), dtype=np.uint8)
        overlay[..., 0] = crop_a.astype(np.uint8) * 255
        overlay[..., 1] = crop_b.astype(np.uint8) * 255
        filename = (
            f"link_{int(row.link_id):05d}_"
            f"p{projection_a:03d}_l{int(row.label_a):05d}_to_"
            f"p{projection_b:03d}_l{int(row.label_b):05d}_"
            f"iou{float(row.iou):.3f}.tif"
        )
        imwrite(crop_dir / filename, overlay, photometric="rgb")
        crop_names.append(str(Path("link_benchmark_crops") / filename))

    review["review_crop"] = crop_names
    review["manual_same_cell"] = ""
    review["manual_notes"] = ""
    return review


def parse_condition(condition: str) -> tuple[float | None, str]:
    normalized = re.sub(r"[^a-z0-9]+", "_", condition.lower()).strip("_")
    hours = re.search(r"(\d+(?:\.\d+)?)[_]*(?:h|hr|hrs|hour|hours)(?:_|$)", normalized)
    days = re.search(r"(\d+(?:\.\d+)?)[_]*(?:d|day|days)(?:_|$)", normalized)
    time_hours = float(hours.group(1)) if hours else 24.0 * float(days.group(1)) if days else None
    held_open = (
        "held" in normalized
        or "hold_open" in normalized
        or "holdopen" in normalized
        or re.search(r"(?:^|_)ho(?:_|$)", normalized) is not None
    )
    if held_open:
        injury = "held_open"
    elif "amputation" in normalized or "amput" in normalized:
        injury = "amputation"
    elif "incision" in normalized or "incis" in normalized:
        injury = "incision"
    elif "dorsal" in normalized and "window" in normalized:
        injury = "dorsal_window"
    else:
        injury = "unparsed"
    return time_hours, injury


def assign_spots(
    sample: SampleSpec,
    label_files: Sequence[Path],
    node_df: pd.DataFrame,
    parameters: Parameters,
    sample_output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    spot_file = locate_spot_file(sample.results_dir)
    spots = load_spots(spot_file)
    projection_indices = np.floor_divide(
        np.rint(spots[:, 0]).astype(np.int64), parameters.projection_size
    )
    ys_all = np.rint(spots[:, 1]).astype(np.int64)
    xs_all = np.rint(spots[:, 2]).astype(np.int64)

    maximum_cell = int(node_df["global_cell_id"].max())
    counts = np.zeros(maximum_cell + 1, dtype=np.int64)
    assigned_spots = 0
    background_spots = 0
    invalid_xy_spots = 0
    invalid_projection_spots = int(
        np.sum((projection_indices < 0) | (projection_indices >= len(label_files)))
    )

    map_dir = sample_output_dir / "assignment_maps"
    if parameters.save_assignment_maps:
        map_dir.mkdir(parents=True, exist_ok=True)

    for projection, label_path in enumerate(label_files):
        labels = read_label(label_path)
        expanded = expand_labels(labels, parameters.expansion_distance)
        local_to_global = np.zeros(int(labels.max()) + 1, dtype=np.int64)
        for row in node_df.loc[node_df["projection"] == projection].itertuples(index=False):
            local_to_global[int(row.local_label)] = int(row.global_cell_id)

        indices = np.flatnonzero(projection_indices == projection)
        ys = ys_all[indices]
        xs = xs_all[indices]
        valid = (ys >= 0) & (ys < labels.shape[0]) & (xs >= 0) & (xs < labels.shape[1])
        invalid_xy_spots += int(np.sum(~valid))
        local_labels = expanded[ys[valid], xs[valid]]
        background_spots += int(np.sum(local_labels == 0))
        foreground = local_labels > 0
        if np.any(foreground):
            global_ids = local_to_global[local_labels[foreground]]
            if np.any(global_ids == 0):
                raise RuntimeError(
                    f"Expanded label lacks global mapping in {sample.sample_id}, "
                    f"projection {projection}"
                )
            counts += np.bincount(global_ids, minlength=len(counts))
            assigned_spots += int(len(global_ids))

    if parameters.save_assignment_maps:
        if int(counts.max()) > np.iinfo(np.uint32).max:
            raise OverflowError("Assignment count map exceeds uint32")
        for projection, label_path in enumerate(label_files):
            labels = read_label(label_path)
            expanded = expand_labels(labels, parameters.expansion_distance)
            local_to_global = np.zeros(int(labels.max()) + 1, dtype=np.int64)
            for row in node_df.loc[node_df["projection"] == projection].itertuples(index=False):
                local_to_global[int(row.local_label)] = int(row.global_cell_id)
            count_map = counts[local_to_global[expanded]].astype(np.uint32, copy=False)
            imwrite(
                map_dir / f"assignment_count_map_{projection:03d}.tif",
                count_map,
                photometric="minisblack",
            )

    cell_df = (
        node_df.groupby("global_cell_id", as_index=False)
        .agg(
            first_projection=("projection", "min"),
            last_projection=("projection", "max"),
            n_projections=("projection", "nunique"),
            summed_nuclear_area_pixels=("nuclear_area_pixels", "sum"),
        )
        .sort_values("global_cell_id")
    )
    cell_df["transcript_count"] = counts[1:]
    cell_df["responding"] = cell_df["transcript_count"] > 0
    time_hours, injury = parse_condition(sample.condition)
    for location, name, value in reversed(
        [
            (0, "experiment", sample.experiment),
            (1, "gene", sample.gene),
            (2, "condition", sample.condition),
            (3, "time_hours", time_hours),
            (4, "injury", injury),
            (5, "animal", sample.animal),
            (6, "sample_id", sample.sample_id),
        ]
    ):
        cell_df.insert(location, name, value)

    total_spots = int(len(spots))
    invalid_spots = invalid_xy_spots + invalid_projection_spots
    accounted_spots = assigned_spots + background_spots + invalid_spots
    spans = cell_df["n_projections"].to_numpy(dtype=np.int64)
    qc = {
        "assignment_backend": (
            "old_projection_labels_no_link"
            if parameters.link_mode == "none"
            else "old_projection_labels_reciprocal_linking"
        ),
        "link_mode": parameters.link_mode,
        "experiment": sample.experiment,
        "gene": sample.gene,
        "condition": sample.condition,
        "time_hours": time_hours,
        "injury": injury,
        "animal": sample.animal,
        "sample_id": sample.sample_id,
        "sample_path": str(sample.sample_dir),
        "spot_file": str(spot_file),
        "projection_size_z_slices": parameters.projection_size,
        "expansion_distance_xy_pixels": parameters.expansion_distance,
        "label_projections": len(label_files),
        "total_spots": total_spots,
        "assigned_spots": assigned_spots,
        "background_spots": background_spots,
        "invalid_xy_spots": invalid_xy_spots,
        "invalid_projection_spots": invalid_projection_spots,
        "invalid_spots": invalid_spots,
        "accounted_spots": accounted_spots,
        "spot_accounting_matches": accounted_spots == total_spots,
        "assignment_fraction_all_spots": assigned_spots / total_spots if total_spots else math.nan,
        "total_cells_in_selected_mode": int(len(cell_df)),
        "responding_cells": int(cell_df["responding"].sum()),
        "responding_fraction": float(cell_df["responding"].mean()),
        "maximum_transcript_count": int(cell_df["transcript_count"].max()),
        "single_projection_cells": int(np.sum(spans == 1)),
        "cells_spanning_multiple_projections": int(np.sum(spans > 1)),
        "cells_spanning_more_than_three_projections": int(np.sum(spans > 3)),
        "median_projections_per_cell": float(np.median(spans)),
        "p95_projections_per_cell": float(np.percentile(spans, 95)),
        "maximum_projections_per_cell": int(np.max(spans)),
    }
    return cell_df, qc


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_write_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=True)
    temporary.replace(path)


def animal_summary(cell_df: pd.DataFrame) -> dict[str, object]:
    first = cell_df.iloc[0]
    counts = cell_df["transcript_count"].to_numpy(dtype=np.int64)
    positive = counts[counts > 0]
    return {
        "experiment": first["experiment"],
        "gene": first["gene"],
        "condition": first["condition"],
        "time_hours": first["time_hours"],
        "injury": first["injury"],
        "animal": first["animal"],
        "sample_id": first["sample_id"],
        "total_cells": int(len(counts)),
        "responding_cells": int(len(positive)),
        "responding_fraction": float(len(positive) / len(counts)),
        "total_assigned_transcripts": int(np.sum(positive)),
        "mean_count_all_cells": float(np.mean(counts)),
        "mean_log1p_all_cells": float(np.log1p(counts).mean()),
        "mean_count_responding_cells": float(np.mean(positive)) if len(positive) else math.nan,
        "median_count_responding_cells": float(np.median(positive)) if len(positive) else math.nan,
        "mean_log1p_responding_cells": float(np.log1p(positive).mean()) if len(positive) else math.nan,
        "fraction_responding_cells_ge_5": float(np.mean(positive >= 5)) if len(positive) else math.nan,
        "maximum_count": int(np.max(positive)) if len(positive) else 0,
    }


def activating_cell_percentage(cell_df: pd.DataFrame) -> dict[str, object]:
    """Return strict transcript-count threshold percentages for one sample."""
    first = cell_df.iloc[0]
    counts = cell_df["transcript_count"].to_numpy(dtype=np.int64)
    row: dict[str, object] = {
        "experiment": first["experiment"],
        "gene": first["gene"],
        "condition": first["condition"],
        "time_hours": first["time_hours"],
        "injury": first["injury"],
        "animal": first["animal"],
        "sample_id": first["sample_id"],
        "total_cells": int(len(counts)),
    }
    for threshold in (0, 1, 2, 5, 10, 100):
        row[f"percent_cells_gt_{threshold}"] = float(
            100.0 * np.mean(counts > threshold)
        )
    return row


def process_sample(
    sample: SampleSpec,
    parameters: Parameters,
    output_root: Path,
    overwrite: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
    sample_output = output_root / sample.experiment / sample.condition / sample.animal
    complete_path = sample_output / "COMPLETE.json"
    if complete_path.is_file() and not overwrite:
        with complete_path.open(encoding="utf-8") as stream:
            complete = json.load(stream)
        if (
            complete.get("script_version") not in COMPATIBLE_SAMPLE_VERSIONS
            or complete.get("parameters") != asdict(parameters)
        ):
            raise ValueError(f"Stale output for {sample.sample_id}; use --overwrite")
        LOGGER.info("Skipping completed sample: %s", sample.sample_id)
        cells = pd.read_csv(sample_output / "cell_counts.csv")
        with (sample_output / "sample_qc.json").open(encoding="utf-8") as stream:
            qc = json.load(stream)
        return cells, qc

    if sample_output.exists() and overwrite:
        remove_output_tree(sample_output)
    elif sample_output.exists():
        raise FileExistsError(f"Partial output for {sample.sample_id}; use --overwrite")
    sample_output.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Processing %s", sample.sample_id)
    labels = locate_label_files(sample.results_dir)
    node_df, link_df, benchmark = build_cells(
        labels,
        parameters.min_smaller_overlap,
        parameters.min_iou,
        parameters.link_mode,
        parameters.benchmark_linking,
    )
    cells, qc = assign_spots(sample, labels, node_df, parameters, sample_output)
    benchmark = {
        "experiment": sample.experiment,
        "condition": sample.condition,
        "animal": sample.animal,
        "sample_id": sample.sample_id,
        **benchmark,
    }
    qc.update(benchmark)
    atomic_write_csv(cells, sample_output / "cell_counts.csv")
    atomic_write_csv(node_df, sample_output / "projection_label_to_cell.csv")
    accepted_link_df = (
        link_df.loc[link_df["accepted_by_threshold"]].copy()
        if not link_df.empty
        else link_df.copy()
    )
    atomic_write_csv(
        accepted_link_df, sample_output / "adjacent_projection_links.csv"
    )
    atomic_write_csv(
        link_df, sample_output / "adjacent_projection_link_candidates.csv"
    )
    atomic_write_json(benchmark, sample_output / "linking_benchmark.json")
    if parameters.benchmark_linking:
        review = save_link_review_crops(
            labels, link_df, sample_output, parameters.benchmark_max_crops
        )
        atomic_write_csv(review, sample_output / "manual_link_review.csv")
    atomic_write_json(qc, sample_output / "sample_qc.json")
    atomic_write_json(
        {
            "script_version": SCRIPT_VERSION,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "parameters": asdict(parameters),
        },
        complete_path,
    )
    return cells, qc


def load_all_completed_sample_outputs(
    output_root: Path,
) -> tuple[list[pd.DataFrame], list[dict[str, object]]]:
    """Load every completed sample, including samples from earlier batches."""
    cell_frames: list[pd.DataFrame] = []
    qc_rows: list[dict[str, object]] = []
    seen_sample_ids: dict[str, Path] = {}
    for cell_path in sorted(output_root.glob("*/*/*/cell_counts.csv")):
        sample_output = cell_path.parent
        complete_path = sample_output / "COMPLETE.json"
        qc_path = sample_output / "sample_qc.json"
        if not complete_path.is_file() or not qc_path.is_file():
            continue
        frame = pd.read_csv(cell_path)
        if frame.empty or "sample_id" not in frame.columns:
            raise ValueError(f"Invalid completed cell table: {cell_path}")
        sample_ids = frame["sample_id"].dropna().astype(str).unique()
        if len(sample_ids) != 1:
            raise ValueError(
                f"Expected one sample_id in {cell_path}, found {sample_ids.tolist()}"
            )
        sample_id = str(sample_ids[0])
        if sample_id in seen_sample_ids:
            raise ValueError(
                f"Duplicate sample_id {sample_id}: "
                f"{seen_sample_ids[sample_id]} and {cell_path}"
            )
        seen_sample_ids[sample_id] = cell_path
        with qc_path.open(encoding="utf-8") as stream:
            qc = json.load(stream)
        cell_frames.append(frame)
        qc_rows.append(qc)
    return cell_frames, qc_rows


def all_cells_output_header(frame: pd.DataFrame) -> tuple[tuple[object, ...], str]:
    """Return 306_output-style sort key and column name for one sample."""
    first = frame.iloc[0]
    condition = str(first["condition"])
    match = re.match(
        r"^\s*(\d+(?:\.\d+)?)\s*(?:hr|h)\s*_(.+?)\s*$",
        condition,
        flags=re.IGNORECASE,
    )
    if match:
        time_value = float(match.group(1))
        treatment = match.group(2)
    else:
        try:
            time_value = float(first.get("time_hours", math.inf))
        except (TypeError, ValueError):
            time_value = math.inf
        treatment = condition
    if math.isfinite(time_value):
        time_label = (
            f"{int(time_value)}h" if time_value.is_integer() else f"{time_value:g}h"
        )
    else:
        time_label = "NAh"
    animal = str(first["animal"])
    animal_match = re.search(r"(\d+)$", animal)
    animal_label = animal_match.group(1) if animal_match else animal
    animal_number = int(animal_match.group(1)) if animal_match else 10**9
    treatment_priority = {"incision": 0, "amputation": 1, "agarose": 2}
    sort_key: tuple[object, ...] = (
        time_value,
        treatment_priority.get(treatment.lower(), 99),
        treatment.lower(),
        animal_number,
        animal.lower(),
    )
    return sort_key, f"{treatment}_{time_label}_{animal_label}"


def aggregate_outputs(
    cell_frames: Sequence[pd.DataFrame],
    qc_rows: Sequence[dict[str, object]],
    output_root: Path,
) -> None:
    all_cells = pd.concat(cell_frames, ignore_index=True)
    atomic_write_csv(all_cells, output_root / "all_cells_long.csv")
    positives: dict[str, pd.Series] = {}
    all_counts: dict[str, pd.Series] = {}
    animals: list[dict[str, object]] = []
    activation_rows: list[dict[str, object]] = []
    experiment_samples: dict[
        str, list[tuple[tuple[object, ...], str, pd.Series]]
    ] = defaultdict(list)
    for sample_id, frame in all_cells.groupby("sample_id", sort=True):
        positives[str(sample_id)] = frame.loc[
            frame["transcript_count"] > 0, "transcript_count"
        ].reset_index(drop=True)
        counts = pd.Series(
            frame["transcript_count"].to_numpy(dtype=np.int64), dtype="Int64"
        ).reset_index(drop=True)
        all_counts[str(sample_id)] = counts
        experiment = str(frame.iloc[0]["experiment"])
        if experiment.endswith(".0") and experiment[:-2].isdigit():
            experiment = experiment[:-2]
        sort_key, output_header = all_cells_output_header(frame)
        experiment_samples[experiment].append((sort_key, output_header, counts))
        animals.append(animal_summary(frame))
        activation_rows.append(activating_cell_percentage(frame))
    atomic_write_csv(pd.DataFrame(positives), output_root / "positive_cells_wide.csv")
    atomic_write_csv(
        pd.DataFrame(all_counts), output_root / "all_cells_wide_including_zero.csv"
    )
    for experiment, entries in sorted(experiment_samples.items()):
        entries.sort(key=lambda item: item[0])
        experiment_columns: dict[str, pd.Series] = {}
        for _, header, counts in entries:
            if header in experiment_columns:
                raise ValueError(
                    f"Duplicate all-cells output header for experiment "
                    f"{experiment}: {header}"
                )
            experiment_columns[header] = counts
        atomic_write_csv(
            pd.DataFrame(experiment_columns),
            output_root / f"{experiment}_output_all_cells_including_zero.csv",
        )
    atomic_write_csv(pd.DataFrame(animals), output_root / "animal_summary.csv")
    atomic_write_csv(
        pd.DataFrame(activation_rows),
        output_root / "activating cell percentage.csv",
    )
    qc_frame = pd.DataFrame(qc_rows)
    atomic_write_csv(qc_frame, output_root / "sample_qc.csv")
    benchmark_columns = [
        "experiment",
        "condition",
        "animal",
        "sample_id",
        "link_mode_used",
        "raw_projection_label_instances",
        "reciprocal_candidate_pairs",
        "proposed_reciprocal_links",
        "independent_cells_if_no_link",
        "cells_if_reciprocal_links_applied",
        "cell_count_reduction_if_linked",
        "cell_count_reduction_fraction_if_linked",
        "linked_cells_spanning_multiple_projections",
        "linked_cells_spanning_more_than_three_projections",
        "maximum_projection_span_if_linked",
        "review_flagged_candidates",
        "median_iou",
        "median_smaller_overlap_fraction",
        "median_area_ratio",
        "median_centroid_shift_over_sqrt_smaller_area",
    ]
    available_benchmark_columns = [
        column for column in benchmark_columns if column in qc_frame.columns
    ]
    if available_benchmark_columns:
        atomic_write_csv(
            qc_frame[available_benchmark_columns],
            output_root / "linking_benchmark.csv",
        )


def run_pipeline(
    experiments: Sequence[ExperimentSpec],
    parameters: Parameters,
    output_root: Path,
    overwrite: bool,
    fail_fast: bool,
    sample_regex: str | None,
) -> int:
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        {
            "script_version": SCRIPT_VERSION,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "assignment_backend": (
                "old_projection_labels_no_link"
                if parameters.link_mode == "none"
                else "old_projection_labels_reciprocal_linking"
            ),
            "experiments": [
                {"name": item.name, "gene": item.gene, "root": str(item.root)}
                for item in experiments
            ],
            "parameters": asdict(parameters),
        },
        output_root / "run_configuration.json",
    )

    pattern = re.compile(sample_regex) if sample_regex else None
    samples: list[SampleSpec] = []
    for experiment in experiments:
        found = discover_samples(experiment, parameters.channel_folder)
        LOGGER.info("Discovered %d samples for experiment %s", len(found), experiment.name)
        samples.extend(found)
    if pattern:
        samples = [sample for sample in samples if pattern.search(sample.sample_id)]
        LOGGER.info("%d samples remain after --sample-regex", len(samples))
    if not samples:
        LOGGER.error("No samples discovered")
        return 2

    cell_frames: list[pd.DataFrame] = []
    qc_rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    for sample in samples:
        try:
            cells, qc = process_sample(sample, parameters, output_root, overwrite)
            cell_frames.append(cells)
            qc_rows.append(qc)
        except Exception as exc:
            LOGGER.exception("Failed sample %s", sample.sample_id)
            errors.append(
                {
                    "sample_id": sample.sample_id,
                    "sample_path": str(sample.sample_dir),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            if fail_fast:
                break

    error_path = output_root / "processing_errors.csv"
    if errors:
        atomic_write_csv(pd.DataFrame(errors), error_path)
    else:
        error_path.unlink(missing_ok=True)
    # Rebuild top-level tables from every completed per-sample output.  This is
    # deliberately cumulative: when --sample-regex is used for several batches,
    # earlier completed samples remain present in the aggregate CSVs.
    completed_cell_frames, completed_qc_rows = load_all_completed_sample_outputs(
        output_root
    )
    if completed_cell_frames:
        aggregate_outputs(completed_cell_frames, completed_qc_rows, output_root)
    atomic_write_json(
        {
            "script_version": SCRIPT_VERSION,
            "assignment_backend": (
                "old_projection_labels_no_link"
                if parameters.link_mode == "none"
                else "old_projection_labels_reciprocal_linking"
            ),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "discovered_samples": len(samples),
            "successful_samples": len(cell_frames),
            "failed_samples": len(errors),
            "all_samples_successful": len(errors) == 0 and len(cell_frames) == len(samples),
        },
        output_root / "batch_qc.json",
    )
    LOGGER.info("Finished: %d successful, %d failed", len(cell_frames), len(errors))
    return 1 if errors else 0


def create_synthetic_experiment(base: Path) -> ExperimentSpec:
    experiment_dir = base / "root" / "Experiment"
    results = experiment_dir / "6h_Amputation" / "Image1" / "565" / "results"
    labels_dir = results / "labels"
    labels_dir.mkdir(parents=True)
    (base / "root" / "Control" / "6h_Amputation" / "Image1" / "565" / "results").mkdir(
        parents=True
    )

    label_0 = np.zeros((40, 50), dtype=np.uint16)
    label_0[10:18, 10:18] = 1
    label_0[25:32, 30:38] = 2
    label_1 = np.zeros_like(label_0)
    label_1[11:19, 11:19] = 1
    label_1[5:11, 35:42] = 2
    label_2 = np.zeros_like(label_0)
    label_2[6:12, 36:43] = 1
    for index, labels in enumerate((label_0, label_1, label_2)):
        imwrite(labels_dir / f"Nucleus_Labels_{index:03d}.tif", labels)
    spots = np.array(
        [[2, 14, 14], [12, 15, 15], [8, 20, 20], [15, 0, 0], [25, 8, 39]],
        dtype=np.int64,
    )
    np.save(results / "spots_post_decomposition_and_background_removed.npy", spots)
    return ExperimentSpec("test", "test_gene", experiment_dir)


def run_self_test() -> int:
    configure_logging(True)
    with tempfile.TemporaryDirectory(prefix="projection_reassign_test_") as temp:
        base = Path(temp)
        parameters = Parameters(
            10, 3.0, 0.30, 0.10, "565", True, "reciprocal", True, 10
        )
        output = base / "output"
        status = run_pipeline(
            [create_synthetic_experiment(base)], parameters, output, True, True, None
        )
        if status != 0:
            raise AssertionError("Synthetic pipeline failed")
        cells = pd.read_csv(output / "all_cells_long.csv")
        qc = pd.read_csv(output / "sample_qc.csv").iloc[0]
        activation = pd.read_csv(
            output / "activating cell percentage.csv"
        ).iloc[0]
        positive = sorted(
            cells.loc[cells["transcript_count"] > 0, "transcript_count"].tolist()
        )
        if positive != [1, 2]:
            raise AssertionError(f"Unexpected positive counts: {positive}")
        all_cells_wide = pd.read_csv(
            output / "all_cells_wide_including_zero.csv"
        )
        experiment_output = pd.read_csv(
            output / "test_output_all_cells_including_zero.csv"
        )
        if all_cells_wide.columns.tolist() != ["test__6h_Amputation__Image1"]:
            raise AssertionError(
                f"Unexpected all-cells columns: {all_cells_wide.columns.tolist()}"
            )
        if experiment_output.columns.tolist() != ["Amputation_6h_1"]:
            raise AssertionError(
                f"Unexpected experiment columns: {experiment_output.columns.tolist()}"
            )
        exported_counts = experiment_output["Amputation_6h_1"].dropna().astype(int)
        if len(exported_counts) != len(cells):
            raise AssertionError("All-cells export omitted one or more cells")
        if int((exported_counts == 0).sum()) != int(
            (cells["transcript_count"] == 0).sum()
        ):
            raise AssertionError("Zero-cell count changed during wide export")
        if int(qc["total_spots"]) != 5 or not bool(qc["spot_accounting_matches"]):
            raise AssertionError("Synthetic spot accounting failed")
        counts_for_percent = cells["transcript_count"].to_numpy(dtype=np.int64)
        for threshold in (0, 1, 2, 5, 10, 100):
            column = f"percent_cells_gt_{threshold}"
            expected = float(100.0 * np.mean(counts_for_percent > threshold))
            if not np.isclose(float(activation[column]), expected):
                raise AssertionError(
                    f"Unexpected {column}: {activation[column]} versus {expected}"
                )

        # An unchanged mode/parameter set must reuse the completed sample.
        reuse_status = run_pipeline(
            [ExperimentSpec("test", "test_gene", base / "root" / "Experiment")],
            parameters,
            output,
            False,
            True,
            None,
        )
        if reuse_status != 0:
            raise AssertionError("Completed sample was not reused")

        # Strict old-label/no-link mode must keep every projection label as a
        # separate cell while still reporting the reciprocal-link benchmark.
        no_link_parameters = Parameters(
            10, 3.0, 0.30, 0.10, "565", False, "none", True, 10
        )
        no_link_output = base / "no_link_output"
        no_link_status = run_pipeline(
            [ExperimentSpec("test", "test_gene", base / "root" / "Experiment")],
            no_link_parameters,
            no_link_output,
            True,
            True,
            None,
        )
        if no_link_status != 0:
            raise AssertionError("Synthetic no-link pipeline failed")
        no_link_cells = pd.read_csv(no_link_output / "all_cells_long.csv")
        no_link_benchmark = pd.read_csv(
            no_link_output / "linking_benchmark.csv"
        ).iloc[0]
        if len(no_link_cells) != 5:
            raise AssertionError(
                f"No-link mode should retain five label instances, got {len(no_link_cells)}"
            )
        if int(no_link_benchmark["cells_if_reciprocal_links_applied"]) != 3:
            raise AssertionError("Unexpected reciprocal benchmark cell count")

        race_dir = base / "overwrite_race"
        race_dir.mkdir()
        sidecar_name = "._cell_counts.csv"
        (race_dir / sidecar_name).touch()
        original_unlink = os.unlink
        triggered = False

        def unlink_with_race(path, *args, **kwargs):
            nonlocal triggered
            if not triggered and Path(os.fspath(path)).name == sidecar_name:
                triggered = True
                original_unlink(path, *args, **kwargs)
                raise FileNotFoundError(os.fspath(path))
            return original_unlink(path, *args, **kwargs)

        try:
            os.unlink = unlink_with_race
            remove_output_tree(race_dir)
        finally:
            os.unlink = original_unlink
        if not triggered or race_dir.exists():
            raise AssertionError("AppleDouble overwrite cleanup failed")
    print("SELF-TEST PASSED: temporary projection reassignment")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        action="append",
        type=parse_experiment_argument,
        help="NAME,GENE,ROOT_OR_EXPERIMENT_DIR; repeat for multiple experiments",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/Volumes/Backup Plus/Experiment_results/smfish_reassignment_old_labels_no_link"
        ),
    )
    parser.add_argument("--projection-size", type=int, default=10)
    parser.add_argument("--expansion-distance", type=float, default=20.0)
    parser.add_argument("--min-smaller-overlap", type=float, default=0.35)
    parser.add_argument("--min-iou", type=float, default=0.15)
    parser.add_argument("--channel-folder", default="565")
    parser.add_argument("--save-assignment-maps", action="store_true")
    parser.add_argument(
        "--link-mode",
        choices=("none", "reciprocal"),
        default="none",
        help=(
            "none reproduces independent old projection labels; reciprocal "
            "applies adjacent-projection reciprocal-overlap links"
        ),
    )
    parser.add_argument(
        "--benchmark-linking",
        action="store_true",
        help="Measure proposed reciprocal links even when --link-mode none is used",
    )
    parser.add_argument(
        "--benchmark-max-crops",
        type=int,
        default=50,
        help="Maximum red/green link-overlay crops saved per sample",
    )
    parser.add_argument("--sample-regex")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return run_self_test()
    if args.projection_size <= 0:
        raise SystemExit("--projection-size must be positive")
    if args.expansion_distance < 0:
        raise SystemExit("--expansion-distance cannot be negative")
    if args.benchmark_max_crops < 0:
        raise SystemExit("--benchmark-max-crops cannot be negative")
    if not 0 <= args.min_smaller_overlap <= 1 or not 0 <= args.min_iou <= 1:
        raise SystemExit("Overlap thresholds must be between 0 and 1")
    configure_logging(args.verbose)
    experiments = args.experiment or [
        ExperimentSpec(name, gene, Path(root)) for name, gene, root in DEFAULT_EXPERIMENTS
    ]
    parameters = Parameters(
        args.projection_size,
        args.expansion_distance,
        args.min_smaller_overlap,
        args.min_iou,
        args.channel_folder,
        args.save_assignment_maps,
        args.link_mode,
        args.benchmark_linking,
        args.benchmark_max_crops,
    )
    return run_pipeline(
        experiments,
        parameters,
        args.output_root,
        args.overwrite,
        args.fail_fast,
        args.sample_regex,
    )


if __name__ == "__main__":
    raise SystemExit(main())
