#!/usr/bin/env python3
"""Assign smFISH spots to the nearest cleaned Cellpose nucleus in calibrated 3-D.

Batch pipeline for experiments 304, 306, 307, 314, and 332. It replaces the
old projection-linking/XY-expansion assignment workflow. For each sample it
uses the cleaned globally stitched Cellpose label stack, assigns each valid
(z,y,x) spot to the nearest nuclear voxel, writes an assignment-count map,
creates a reproducibly random 512x512 five-view example, and produces
biological-replicate-level tables, plots, unpaired Welch tests, and Dunn tests.

Typical run::

    caffeinate -ims python reassign_smfish_3d.py 2>&1 | \
      tee -a smfish_cellpose_assignment.log

Required: numpy, pandas, scipy, matplotlib, tifffile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage, stats

try:
    from tifffile import TiffFile, TiffWriter, imread as tifffile_imread
except ImportError:
    TiffFile = None
    TiffWriter = None
    tifffile_imread = None


SCRIPT_VERSION = "2.0.0"
LOGGER = logging.getLogger("smfish_cellpose_assignment")
FINAL_LABEL_DIRNAME = "cellpose_label_plane_s4_cp-1p75_f0p8_d90_n500_st0p25"
CLEANED_LABEL_DIRNAME = FINAL_LABEL_DIRNAME + "_cleaned"
DEFAULT_OUTPUT_ROOT = Path(
    "/Volumes/Backup Plus/Experiment_results/smFISH_cellpose_assignment"
)
DEFAULT_EXPERIMENTS = (
    ("304", "wnt1", "/Volumes/Backup Plus/Experiment_results/304_Analysis_results", (64.0, 64.0, 361.0)),
    ("306", "wnt1", "/Volumes/Backup Plus/Experiment_results/306_analysis_results_alter", (64.0, 64.0, 361.0)),
    ("307", "wnt1", "/Volumes/Backup Plus/Experiment_results/307_analysis_resutls/307_quantificatoin", (64.0, 64.0, 361.0)),
    ("314", "notum", "/Volumes/Backup Plus/Experiment_results/314_analysis_results", (74.0, 74.0, 500.0)),
    ("332", "notum", "/Volumes/Backup Plus/Experiment_results/332_notum_analysis", (74.0, 74.0, 500.0)),
)
LABEL_INDEX_PATTERN = re.compile(r"Nucleus_Labels_(\d+)")
THRESHOLDS = (1, 5, 10)
INJURY_ORDER = ("incision", "amputation", "held_open", "dorsal_window", "unparsed")
INJURY_COLORS = {
    "incision": "#377eb8", "amputation": "#e41a1c", "held_open": "#4daf4a",
    "dorsal_window": "#984ea3", "unparsed": "#777777",
}


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    gene: str
    root: Path
    voxel_xyz_nm: tuple[float, float, float]

    @property
    def voxel_zyx_um(self) -> tuple[float, float, float]:
        x_nm, y_nm, z_nm = self.voxel_xyz_nm
        return z_nm / 1000.0, y_nm / 1000.0, x_nm / 1000.0


@dataclass(frozen=True)
class Parameters:
    channel_folder: str
    cleaned_label_dirname: str
    crop_size: int
    random_seed: int
    max_link_lines: int
    save_assignment_maps: bool
    save_examples: bool


@dataclass(frozen=True)
class SampleSpec:
    experiment: ExperimentSpec
    condition: str
    animal: str
    sample_dir: Path
    results_dir: Path

    @property
    def sample_id(self) -> str:
        return f"{self.experiment.name}__{self.condition}__{self.animal}"

    @property
    def replicate_name(self) -> str:
        return f"{self.condition}__{self.animal}"


@dataclass
class CellAccumulator:
    voxel_count: int = 0
    sum_z: float = 0.0
    sum_y: float = 0.0
    sum_x: float = 0.0
    first_z: int = 2**31 - 1
    last_z: int = -1
    n_z_planes: int = 0
    max_cross_section_pixels: int = 0
    example_z: int = -1
    example_y: float = math.nan
    example_x: float = math.nan


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def natural_key(path_or_name: str | Path) -> list[object]:
    return [int(p) if p.isdigit() else p.lower()
            for p in re.split(r"(\d+)", Path(path_or_name).name)]


def parse_experiment_argument(value: str) -> ExperimentSpec:
    parts = value.split(",", 5)
    if len(parts) != 6 or not all(part.strip() for part in parts):
        raise argparse.ArgumentTypeError(
            "--experiment must be NAME,GENE,ROOT,X_NM,Y_NM,Z_NM"
        )
    name, gene, root, x_nm, y_nm, z_nm = (part.strip() for part in parts)
    try:
        voxel = (float(x_nm), float(y_nm), float(z_nm))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Voxel sizes must be numeric") from exc
    if any(v <= 0 for v in voxel):
        raise argparse.ArgumentTypeError("Voxel sizes must be positive")
    return ExperimentSpec(name, gene, Path(root), voxel)


def parse_condition(condition: str) -> tuple[float | None, str]:
    normalized = condition.lower().replace("-", "_").replace(" ", "_")
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hour|hours)", normalized)
    time_hours = float(match.group(1)) if match else None
    if "held" in normalized or "hold_open" in normalized or "holdopen" in normalized:
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


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=True)
    temporary.replace(path)


def read_label(path: Path) -> np.ndarray:
    if tifffile_imread is not None:
        labels = np.asarray(tifffile_imread(path))
    else:
        from PIL import Image
        labels = np.asarray(Image.open(path))
    labels = np.squeeze(labels)
    if labels.ndim != 2:
        raise ValueError(f"Expected 2-D labels at {path}; got {labels.shape}")
    if not np.issubdtype(labels.dtype, np.integer) and not np.all(labels == np.floor(labels)):
        raise TypeError(f"Non-integer labels at {path}")
    labels = labels.astype(np.int64, copy=False)
    if np.any(labels < 0):
        raise ValueError(f"Negative labels at {path}")
    return labels


def label_plane_index(path: Path) -> int:
    match = LABEL_INDEX_PATTERN.search(path.stem)
    if match is None:
        raise ValueError(f"Cannot parse z index from {path.name}")
    return int(match.group(1))


def discover_samples(experiment: ExperimentSpec, channel_folder: str) -> list[SampleSpec]:
    if not experiment.root.exists():
        LOGGER.warning("Experiment root does not exist; skipping: %s", experiment.root)
        return []
    samples = []
    for results_dir in experiment.root.rglob("results"):
        if not results_dir.is_dir() or results_dir.parent.name != channel_folder:
            continue
        sample_dir = results_dir.parent.parent
        condition_dir = sample_dir.parent
        if condition_dir.parent.name.lower() != "experiment":
            continue
        samples.append(SampleSpec(experiment, condition_dir.name, sample_dir.name,
                                  sample_dir, results_dir))
    unique = {sample.results_dir.resolve(): sample for sample in samples}
    return sorted(unique.values(),
                  key=lambda s: (natural_key(s.condition), natural_key(s.animal)))


def locate_cleaned_label_files(sample: SampleSpec, dirname: str) -> tuple[Path, list[Path]]:
    label_dir = sample.results_dir / dirname
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Cleaned Cellpose label directory is missing: {label_dir}")
    summary_path = label_dir / "cleaning_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Cleaning completion record is missing: {summary_path}")
    with summary_path.open(encoding="utf-8") as stream:
        summary = json.load(stream)
    if summary.get("status") != "completed":
        raise RuntimeError(f"Cleaned label stack is not completed: {summary_path}")
    files = sorted(label_dir.glob("Nucleus_Labels_*.tif*"), key=natural_key)
    if not files:
        raise FileNotFoundError(f"No cleaned label TIFFs in {label_dir}")
    indices = [label_plane_index(path) for path in files]
    if indices != list(range(indices[0], indices[0] + len(indices))):
        raise ValueError(f"Cleaned label z indices are not contiguous in {label_dir}")
    expected = read_label(files[0]).shape
    for path in files[1:]:
        if read_label(path).shape != expected:
            raise ValueError(f"Cleaned label shape mismatch: {path}")
    return label_dir, files


def locate_spot_file(results_dir: Path) -> Path:
    for name in ("spots_post_decomposition_and_background_removed.npy",
                 "spots_post_decomposition.npy"):
        path = results_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No final spot NPY found in {results_dir}")


def load_spots(path: Path) -> np.ndarray:
    spots = np.asarray(np.load(path, allow_pickle=False))
    if spots.ndim != 2 or spots.shape[1] < 3:
        raise ValueError(f"Expected spots shaped (n, >=3) at {path}; got {spots.shape}")
    coordinates = spots[:, [0, -2, -1]].astype(np.float64, copy=False)
    if not np.all(np.isfinite(coordinates)):
        raise ValueError(f"Spot coordinates contain NaN/inf: {path}")
    return coordinates


def locate_raw_smfish_file(sample: SampleSpec) -> Path:
    candidates = [path for path in sample.results_dir.parent.glob("*.tif*")
                  if path.is_file() and not path.name.startswith("._")]
    if not candidates:
        raise FileNotFoundError(f"No raw smFISH TIFF in {sample.results_dir.parent}")
    def score(path: Path) -> tuple[int, int, str]:
        name = path.name.lower()
        match_score = sum(token.lower() in name for token in
                          (sample.condition, sample.animal, sample.experiment.gene, "565"))
        return match_score, path.stat().st_size, path.name
    return max(candidates, key=score)


def update_cell_accumulators(accumulators: dict[int, CellAccumulator],
                             labels: np.ndarray, z_index: int) -> None:
    ys, xs = np.nonzero(labels)
    if not len(ys):
        return
    ids = labels[ys, xs].astype(np.int64, copy=False)
    length = int(ids.max()) + 1
    counts = np.bincount(ids, minlength=length)
    sums_y = np.bincount(ids, weights=ys, minlength=length)
    sums_x = np.bincount(ids, weights=xs, minlength=length)
    for cell_id in np.flatnonzero(counts[1:]) + 1:
        area = int(counts[cell_id])
        acc = accumulators.setdefault(int(cell_id), CellAccumulator())
        acc.voxel_count += area
        acc.sum_z += z_index * area
        acc.sum_y += float(sums_y[cell_id])
        acc.sum_x += float(sums_x[cell_id])
        acc.first_z = min(acc.first_z, z_index)
        acc.last_z = max(acc.last_z, z_index)
        acc.n_z_planes += 1
        if area > acc.max_cross_section_pixels:
            acc.max_cross_section_pixels = area
            acc.example_z = z_index
            acc.example_y = float(sums_y[cell_id] / area)
            acc.example_x = float(sums_x[cell_id] / area)


def assign_spots_nearest_3d(sample: SampleSpec, label_files: Sequence[Path],
                            spots: np.ndarray
                            ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], np.ndarray]:
    """Exact nearest discrete nuclear voxel, streamed one z-plane at a time."""
    height, width = read_label(label_files[0]).shape
    z_indices = np.asarray([label_plane_index(path) for path in label_files], float)
    z_min, z_max = float(z_indices.min()), float(z_indices.max())
    z_um, y_um, x_um = sample.experiment.voxel_zyx_um
    spot_z, spot_y, spot_x = spots[:, 0], spots[:, 1], spots[:, 2]
    valid_xy = ((spot_y >= 0) & (spot_y <= height - 1) &
                (spot_x >= 0) & (spot_x <= width - 1))
    valid_z = (spot_z >= z_min) & (spot_z <= z_max)
    valid = valid_xy & valid_z
    query_y = np.clip(np.rint(spot_y[valid]).astype(np.int64), 0, height - 1)
    query_x = np.clip(np.rint(spot_x[valid]).astype(np.int64), 0, width - 1)
    n_valid = int(valid.sum())
    best_distance = np.full(n_valid, np.inf)
    best_cell = np.zeros(n_valid, dtype=np.int64)
    best_z = np.full(n_valid, -1, dtype=np.int64)
    best_y = np.full(n_valid, -1, dtype=np.int64)
    best_x = np.full(n_valid, -1, dtype=np.int64)
    accumulators: dict[int, CellAccumulator] = {}
    planes_with_nuclei = 0
    for position, label_path in enumerate(label_files):
        labels = read_label(label_path)
        z_index = label_plane_index(label_path)
        update_cell_accumulators(accumulators, labels, z_index)
        if not np.any(labels > 0):
            continue
        planes_with_nuclei += 1
        nearest = ndimage.distance_transform_edt(
            labels == 0, sampling=(y_um, x_um),
            return_distances=False, return_indices=True,
        )
        nearest_y = nearest[0, query_y, query_x]
        nearest_x = nearest[1, query_y, query_x]
        candidate_cell = labels[nearest_y, nearest_x].astype(np.int64, copy=False)
        dy = (spot_y[valid] - nearest_y) * y_um
        dx = (spot_x[valid] - nearest_x) * x_um
        dz = (spot_z[valid] - z_index) * z_um
        candidate_distance = np.sqrt(dx * dx + dy * dy + dz * dz)
        better = candidate_distance < best_distance - 1e-12
        tie = np.isclose(candidate_distance, best_distance, rtol=0, atol=1e-12)
        tie &= (best_cell == 0) | (candidate_cell < best_cell)
        choose = better | tie
        best_distance[choose] = candidate_distance[choose]
        best_cell[choose] = candidate_cell[choose]
        best_z[choose] = z_index
        best_y[choose] = nearest_y[choose]
        best_x[choose] = nearest_x[choose]
        LOGGER.info("%s | nearest 3-D pass %d/%d z=%d", sample.sample_id,
                    position + 1, len(label_files), z_index)
    if not accumulators or not planes_with_nuclei:
        raise RuntimeError(f"No cleaned nuclei remain in {sample.sample_id}")
    if np.any(best_cell == 0):
        raise RuntimeError(f"Valid spots could not be assigned in {sample.sample_id}")
    assigned_cell = np.zeros(len(spots), dtype=np.int64)
    nearest_z = np.full(len(spots), -1, dtype=np.int64)
    nearest_y_all = np.full(len(spots), -1, dtype=np.int64)
    nearest_x_all = np.full(len(spots), -1, dtype=np.int64)
    distance_all = np.full(len(spots), np.nan)
    assigned_cell[valid], nearest_z[valid] = best_cell, best_z
    nearest_y_all[valid], nearest_x_all[valid] = best_y, best_x
    distance_all[valid] = best_distance
    counts = np.bincount(assigned_cell[valid],
                         minlength=max(max(accumulators), int(assigned_cell.max())) + 1)
    time_hours, injury = parse_condition(sample.condition)
    rows = []
    for cell_id in sorted(accumulators):
        acc = accumulators[cell_id]
        count = int(counts[cell_id]) if cell_id < len(counts) else 0
        rows.append({
            "experiment": sample.experiment.name, "gene": sample.experiment.gene,
            "condition": sample.condition, "time_hours": time_hours, "injury": injury,
            "animal": sample.animal, "sample_id": sample.sample_id,
            "replicate_name": sample.replicate_name, "global_cell_id": cell_id,
            "first_z": acc.first_z, "last_z": acc.last_z,
            "n_z_planes": acc.n_z_planes, "nuclear_voxel_count": acc.voxel_count,
            "max_cross_section_pixels": acc.max_cross_section_pixels,
            "centroid_z": acc.sum_z / acc.voxel_count,
            "centroid_y": acc.sum_y / acc.voxel_count,
            "centroid_x": acc.sum_x / acc.voxel_count,
            "example_z": acc.example_z, "example_y": acc.example_y,
            "example_x": acc.example_x, "transcript_count": count,
            "responding": count > 0,
        })
    cell_df = pd.DataFrame(rows)
    spot_df = pd.DataFrame({
        "experiment": sample.experiment.name, "condition": sample.condition,
        "animal": sample.animal, "sample_id": sample.sample_id,
        "spot_index": np.arange(len(spots)), "spot_z": spot_z, "spot_y": spot_y,
        "spot_x": spot_x, "coordinate_valid": valid,
        "assigned_cell_id": assigned_cell, "nearest_nucleus_z": nearest_z,
        "nearest_nucleus_y": nearest_y_all, "nearest_nucleus_x": nearest_x_all,
        "assignment_distance_um": distance_all,
    })
    distances = distance_all[valid]
    mean_distance = float(np.mean(distances)) if len(distances) else math.nan
    median_distance = float(np.median(distances)) if len(distances) else math.nan
    p95_distance = float(np.percentile(distances, 95)) if len(distances) else math.nan
    maximum_distance = float(np.max(distances)) if len(distances) else math.nan
    qc = {
        "experiment": sample.experiment.name, "gene": sample.experiment.gene,
        "condition": sample.condition, "time_hours": time_hours, "injury": injury,
        "animal": sample.animal, "sample_id": sample.sample_id,
        "sample_path": str(sample.sample_dir),
        "voxel_xyz_nm": list(sample.experiment.voxel_xyz_nm),
        "label_planes": len(label_files), "label_z_min": int(z_min),
        "label_z_max": int(z_max), "label_shape_yx": [height, width],
        "total_cleaned_cells": len(cell_df), "total_spots": len(spots),
        "assigned_spots": int(valid.sum()), "invalid_xy_spots": int((~valid_xy).sum()),
        "invalid_z_spots": int((valid_xy & ~valid_z).sum()),
        "unassigned_valid_spots": int(np.sum(valid & (assigned_cell == 0))),
        "spot_accounting_matches": int(valid.sum() + (~valid).sum()) == len(spots),
        "responding_cells": int(cell_df["responding"].sum()),
        "responding_fraction": float(cell_df["responding"].mean()),
        "maximum_transcript_count": int(cell_df["transcript_count"].max()),
        "mean_assignment_distance_um": mean_distance,
        "median_assignment_distance_um": median_distance,
        "p95_assignment_distance_um": p95_distance,
        "maximum_assignment_distance_um": maximum_distance,
        "fraction_assignment_distance_gt_5um": (
            float(np.mean(distances > 5)) if len(distances) else math.nan),
        "fraction_assignment_distance_gt_10um": (
            float(np.mean(distances > 10)) if len(distances) else math.nan),
        "fraction_assignment_distance_gt_20um": (
            float(np.mean(distances > 20)) if len(distances) else math.nan),
    }
    return cell_df, spot_df, qc, counts.astype(np.int64)


def write_tiff_stack(path: Path, planes: Iterable[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if TiffWriter is not None:
        with TiffWriter(temporary, bigtiff=True) as writer:
            for plane in planes:
                writer.write(np.asarray(plane, dtype=np.uint32), photometric="minisblack",
                             compression="zlib")
    else:
        from PIL import Image
        images = [Image.fromarray(np.asarray(plane, dtype=np.uint32)) for plane in planes]
        if not images:
            raise ValueError("Cannot write empty TIFF stack")
        images[0].save(
            temporary,
            format="TIFF",
            save_all=True,
            append_images=images[1:],
        )
    temporary.replace(path)


def write_assignment_count_map(label_files: Sequence[Path], counts: np.ndarray,
                               path: Path) -> None:
    max_label = max(int(read_label(label_path).max()) for label_path in label_files)
    if max_label >= len(counts):
        counts = np.pad(counts, (0, max_label + 1 - len(counts)))
    write_tiff_stack(path, (counts[read_label(label_path)].astype(np.uint32, copy=False)
                            for label_path in label_files))


def stable_rng(sample_id: str, seed: int) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}|{sample_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def choose_example_cell(cell_df: pd.DataFrame,
                        rng: np.random.Generator) -> tuple[pd.Series, str]:
    eligible = cell_df[cell_df["transcript_count"] > 10]
    rule = "random_expression_gt_10"
    if eligible.empty:
        eligible = cell_df[cell_df["transcript_count"] == cell_df["transcript_count"].max()]
        rule = "random_maximum_expression"
    return eligible.iloc[int(rng.integers(0, len(eligible)))], rule


def crop_bounds(center: float, size: int, length: int) -> tuple[int, int]:
    if length <= size:
        return 0, length
    start = max(0, min(int(round(center)) - size // 2, length - size))
    return start, start + size


def normalize_raw_stack(array: np.ndarray, label_shape: tuple[int, int]) -> np.ndarray:
    raw = np.squeeze(np.asarray(array))
    if raw.ndim == 2:
        if raw.shape != label_shape:
            raise ValueError(f"Raw shape {raw.shape} != label shape {label_shape}")
        return raw[np.newaxis]
    if raw.ndim != 3:
        raise ValueError(f"Expected 2-D/3-D raw TIFF; got {raw.shape}")
    height, width = label_shape
    for z_axis in range(3):
        other = [axis for axis in range(3) if axis != z_axis]
        y_axis, x_axis = other
        if raw.shape[y_axis] == height and raw.shape[x_axis] == width:
            return np.moveaxis(raw, (z_axis, y_axis, x_axis), (0, 1, 2))
        if raw.shape[y_axis] == width and raw.shape[x_axis] == height:
            return np.moveaxis(raw, (z_axis, y_axis, x_axis), (0, 1, 2)).swapaxes(1, 2)
    raise ValueError(f"Cannot align raw shape {raw.shape} to labels {label_shape}")


def raw_crop_max_projection(raw_path: Path, label_shape: tuple[int, int],
                            y_slice: slice, x_slice: slice) -> np.ndarray:
    height, width = label_shape
    if TiffFile is not None:
        with TiffFile(raw_path) as tif:
            series = tif.series[0]
            pages = list(series.pages)
            if pages and all(tuple(page.shape) == (height, width) for page in pages):
                projection = None
                for page in pages:
                    crop = np.asarray(page.asarray())[y_slice, x_slice]
                    projection = crop.copy() if projection is None else np.maximum(projection, crop)
                if projection is None:
                    raise ValueError(f"No TIFF pages in {raw_path}")
                return projection
            raw = normalize_raw_stack(series.asarray(), label_shape)
            return np.max(raw[:, y_slice, x_slice], axis=0)
    from PIL import Image
    image = Image.open(raw_path)
    projection = None
    frame = 0
    while True:
        try:
            image.seek(frame)
        except EOFError:
            break
        array = np.asarray(image)
        if array.shape != (height, width):
            raw = normalize_raw_stack(array, label_shape)
            return np.max(raw[:, y_slice, x_slice], axis=0)
        crop = array[y_slice, x_slice]
        projection = crop.copy() if projection is None else np.maximum(projection, crop)
        frame += 1
    if projection is None:
        raise ValueError(f"No TIFF frames in {raw_path}")
    return projection


def display_limits(image: np.ndarray) -> tuple[float, float]:
    values = np.asarray(image, float)
    values = values[np.isfinite(values)]
    if not values.size:
        return 0.0, 1.0
    low, high = np.percentile(values, (1, 99.8))
    return float(low), float(high if high > low else low + 1)


def label_rgb(labels: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*labels.shape, 3), dtype=np.float32)
    for cell_id in np.unique(labels):
        if not cell_id:
            continue
        digest = hashlib.sha256(str(int(cell_id)).encode()).digest()
        color = np.asarray([64 + digest[0] % 192, 64 + digest[1] % 192,
                            64 + digest[2] % 192]) / 255.0
        rgb[labels == cell_id] = color
    return rgb


def outline(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask)


def draw_raw(ax: plt.Axes, raw: np.ndarray, title: str) -> None:
    vmin, vmax = display_limits(raw)
    ax.imshow(raw, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_axis_off()


def save_single_panel(path: Path, draw: Callable[[plt.Axes], None]) -> None:
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    draw(ax)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_example_images(sample: SampleSpec, label_files: Sequence[Path],
                          cell_df: pd.DataFrame, spot_df: pd.DataFrame,
                          counts: np.ndarray, example_dir: Path,
                          parameters: Parameters) -> dict[str, object]:
    example_dir.mkdir(parents=True, exist_ok=True)
    rng = stable_rng(sample.sample_id, parameters.random_seed)
    chosen, selection_rule = choose_example_cell(cell_df, rng)
    selected_cell = int(chosen["global_cell_id"])
    z_index = int(chosen["example_z"])
    by_z = {label_plane_index(path): path for path in label_files}
    labels_full = read_label(by_z[z_index])
    height, width = labels_full.shape
    y0, y1 = crop_bounds(float(chosen["example_y"]), parameters.crop_size, height)
    x0, x1 = crop_bounds(float(chosen["example_x"]), parameters.crop_size, width)
    labels = labels_full[y0:y1, x0:x1]
    raw_path = locate_raw_smfish_file(sample)
    raw = raw_crop_max_projection(raw_path, (height, width), slice(y0, y1), slice(x0, x1))
    selected_outline = outline(labels == selected_cell)
    all_nuclei_outline = outline(labels > 0)
    if int(labels.max()) >= len(counts):
        counts = np.pad(counts, (0, int(labels.max()) + 1 - len(counts)))
    count_map = counts[labels]
    visible = spot_df[
        spot_df["coordinate_valid"] & spot_df["spot_y"].between(y0, y1 - 1) &
        spot_df["spot_x"].between(x0, x1 - 1)
    ].copy()
    if len(visible) > parameters.max_link_lines:
        positions = np.sort(rng.choice(len(visible), parameters.max_link_lines, replace=False))
        visible_links = visible.iloc[positions]
    else:
        visible_links = visible

    def add_outline(ax: plt.Axes) -> None:
        yy, xx = np.nonzero(selected_outline)
        ax.scatter(xx, yy, s=2, c="#00ffff", linewidths=0)

    def draw_smfish(ax: plt.Axes) -> None:
        draw_raw(ax, raw, "smFISH max projection")
        add_outline(ax)

    def draw_detection(ax: plt.Axes) -> None:
        draw_raw(ax, raw, "smFISH detection")
        ax.scatter(visible["spot_x"] - x0, visible["spot_y"] - y0, s=11,
                   facecolors="none", edgecolors="#ff3030", linewidths=0.7)
        add_outline(ax)

    def draw_segmentation(ax: plt.Axes) -> None:
        ax.imshow(label_rgb(labels))
        ax.set_title(f"Cleaned nuclei segmentation, z={z_index}")
        add_outline(ax)
        ax.set_axis_off()

    def draw_assignment(ax: plt.Axes) -> None:
        positive = count_map[count_map > 0]
        vmax = max(1, float(np.percentile(positive, 99))) if positive.size else 1
        shown = ax.imshow(count_map, cmap="magma", vmin=0, vmax=vmax)
        ax.set_title("Nuclei assignment-count map")
        yy_all, xx_all = np.nonzero(all_nuclei_outline)
        ax.scatter(xx_all, yy_all, s=0.35, c="#d0d0d0", linewidths=0, alpha=0.7)
        add_outline(ax)
        ax.set_axis_off()
        plt.colorbar(shown, ax=ax, fraction=0.046, pad=0.04, label="spots / cell")

    def draw_links(ax: plt.Axes) -> None:
        draw_raw(ax, raw, "3-D spot-to-nearest-nucleus links")
        for row in visible_links.itertuples(index=False):
            color = "#00ffff" if int(row.assigned_cell_id) == selected_cell else "#ffd92f"
            ax.plot([row.spot_x - x0, row.nearest_nucleus_x - x0],
                    [row.spot_y - y0, row.nearest_nucleus_y - y0],
                    color=color, linewidth=0.55, alpha=0.65)
        ax.scatter(visible_links["spot_x"] - x0, visible_links["spot_y"] - y0,
                   s=9, c="#ff3030", linewidths=0)
        add_outline(ax)
        ax.set_xlim(-0.5, raw.shape[1] - 0.5)
        ax.set_ylim(raw.shape[0] - 0.5, -0.5)

    panels = [
        ("01_smFISH.png", draw_smfish),
        ("02_smFISH_detection.png", draw_detection),
        ("03_nuclei_segmentation.png", draw_segmentation),
        ("04_nuclei_assignment_map.png", draw_assignment),
        ("05_spot_to_nearest_nucleus_links.png", draw_links),
    ]
    for filename, draw in panels:
        save_single_panel(example_dir / filename, draw)
    fig, axes = plt.subplots(1, 5, figsize=(24, 5), constrained_layout=True)
    for ax, (_, draw) in zip(axes, panels, strict=True):
        draw(ax)
    fig.suptitle(f"{sample.sample_id} | cell {selected_cell} | expression="
                 f"{int(chosen['transcript_count'])}")
    fig.savefig(example_dir / "00_example_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    metadata = {
        "sample_id": sample.sample_id, "condition": sample.condition,
        "animal": sample.animal, "raw_smfish_file": str(raw_path),
        "selected_cell_id": selected_cell,
        "selected_cell_expression": int(chosen["transcript_count"]),
        "selection_rule": selection_rule, "selected_z_plane": z_index,
        "crop_y0_y1": [y0, y1], "crop_x0_x1": [x0, x1],
        "crop_size_requested": parameters.crop_size,
        "spots_visible_in_crop": len(visible), "links_drawn": len(visible_links),
    }
    atomic_write_json(metadata, example_dir / "example_metadata.json")
    return metadata


def plot_distance_qc(spot_df: pd.DataFrame, path: Path, sample_id: str) -> None:
    values = spot_df.loc[spot_df["coordinate_valid"], "assignment_distance_um"].dropna()
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.hist(values, bins=50, color="#4c78a8", edgecolor="white", linewidth=0.3)
    ax.set(xlabel="3-D distance to assigned nucleus (µm)", ylabel="spots", title=sample_id)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def sample_summary_from_cells(cell_df: pd.DataFrame) -> dict[str, object]:
    first = cell_df.iloc[0]
    counts = cell_df["transcript_count"].to_numpy(float)
    nonzero = counts[counts > 0]
    row = {
        "experiment": first["experiment"], "gene": first["gene"],
        "condition": first["condition"], "time_hours": first["time_hours"],
        "injury": first["injury"], "animal": first["animal"],
        "sample_id": first["sample_id"], "replicate_name": first["replicate_name"],
        "total_cells": len(counts), "nonzero_cells": len(nonzero),
        "percent_nonzero_cells": 100 * len(nonzero) / len(counts),
        "total_assigned_spots": int(np.sum(counts)),
        "mean_count_all_cells": float(np.mean(counts)),
        "median_count_all_cells": float(np.median(counts)),
        "mean_count_nonzero_cells": float(np.mean(nonzero)) if len(nonzero) else math.nan,
        "median_count_nonzero_cells": float(np.median(nonzero)) if len(nonzero) else math.nan,
        "maximum_count": int(np.max(counts)),
    }
    for threshold in THRESHOLDS:
        row[f"percent_cells_gt_{threshold}"] = float(100 * np.mean(counts > threshold))
    return row


def wide_counts(all_cells: pd.DataFrame, nonzero_only: bool) -> pd.DataFrame:
    columns = {}
    ordered = all_cells.sort_values(["condition", "animal", "global_cell_id"])
    for replicate, group in ordered.groupby("replicate_name", sort=False):
        values = group["transcript_count"]
        if nonzero_only:
            values = values[values > 0]
        columns[str(replicate)] = values.reset_index(drop=True)
    return pd.DataFrame(columns)


def welch_df(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 2 or len(second) < 2:
        return math.nan
    v1, v2 = np.var(first, ddof=1), np.var(second, ddof=1)
    numerator = (v1 / len(first) + v2 / len(second)) ** 2
    denominator = ((v1 / len(first)) ** 2 / (len(first) - 1) +
                   (v2 / len(second)) ** 2 / (len(second) - 1))
    return float(numerator / denominator) if denominator > 0 else math.nan


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    result = np.full(len(p_values), np.nan)
    finite = [i for i, value in enumerate(p_values) if np.isfinite(value)]
    ordered = sorted(finite, key=lambda i: p_values[i])
    previous = 0.0
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, float(p_values[index]) * (len(ordered) - rank))
        previous = max(previous, adjusted)
        result[index] = previous
    return result.tolist()


def unpaired_ttests(summary: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    rows = []
    times = sorted(summary["time_hours"].dropna().unique())
    for metric in metrics:
        indices = []
        for time_hours in times:
            subset = summary[summary["time_hours"] == time_hours]
            first = subset.loc[subset["injury"] == "incision", metric].dropna().to_numpy(float)
            second = subset.loc[subset["injury"] == "amputation", metric].dropna().to_numpy(float)
            statistic = p_value = math.nan
            if len(first) >= 2 and len(second) >= 2:
                variance_first = float(np.var(first, ddof=1))
                variance_second = float(np.var(second, ddof=1))
                if variance_first == 0 and variance_second == 0:
                    difference = float(np.mean(first) - np.mean(second))
                    statistic = 0.0 if difference == 0 else math.copysign(math.inf, difference)
                    p_value = 1.0 if difference == 0 else 0.0
                else:
                    test = stats.ttest_ind(
                        first, second, equal_var=False, alternative="two-sided"
                    )
                    statistic, p_value = float(test.statistic), float(test.pvalue)
            rows.append({
                "metric": metric, "time_hours": time_hours,
                "group_1": "incision", "group_2": "amputation",
                "n_group_1": len(first), "n_group_2": len(second),
                "mean_group_1": float(np.mean(first)) if len(first) else math.nan,
                "mean_group_2": float(np.mean(second)) if len(second) else math.nan,
                "mean_difference_group_1_minus_group_2": (
                    float(np.mean(first) - np.mean(second)) if len(first) and len(second)
                    else math.nan),
                "t_statistic": statistic, "welch_df": welch_df(first, second),
                "p_value": p_value, "test": "two-sided unpaired Welch t-test",
                "statistical_unit": "biological replicate / image",
            })
            indices.append(len(rows) - 1)
        adjusted = holm_adjust([rows[index]["p_value"] for index in indices])
        for index, value in zip(indices, adjusted, strict=True):
            rows[index]["p_value_holm_across_times_for_metric"] = value
    return pd.DataFrame(rows)


def plot_biological_replicate_boxplots(summary: pd.DataFrame,
                                       metrics: Sequence[str], titles: Sequence[str],
                                       ylabels: Sequence[str], path: Path) -> None:
    times = sorted(summary["time_hours"].dropna().unique())
    injuries = [item for item in INJURY_ORDER if item in set(summary["injury"])]
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.4 * len(metrics), 5),
                             squeeze=False, constrained_layout=True)
    offsets = np.linspace(-0.25, 0.25, max(1, len(injuries)))
    rng = np.random.default_rng(20260831)
    for panel, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels, strict=True)):
        ax = axes[0, panel]
        handles = []
        for injury_index, injury in enumerate(injuries):
            color = INJURY_COLORS[injury]
            for time_index, time_hours in enumerate(times):
                values = summary.loc[(summary["injury"] == injury) &
                                     (summary["time_hours"] == time_hours), metric].dropna().to_numpy(float)
                if not len(values):
                    continue
                position = time_index + offsets[injury_index]
                box = ax.boxplot([values], positions=[position], widths=0.22,
                                 patch_artist=True, showfliers=False,
                                 medianprops={"color": "black"})
                box["boxes"][0].set_facecolor(color)
                box["boxes"][0].set_alpha(0.42)
                ax.scatter(np.full(len(values), position) + rng.uniform(-0.035, 0.035, len(values)),
                           values, color=color, edgecolor="black", linewidth=0.35,
                           s=34, zorder=3)
            handles.append(plt.Line2D([0], [0], marker="o", color="none",
                                      markerfacecolor=color, markeredgecolor="black",
                                      label=injury))
        ax.set_xticks(range(len(times)), [f"{time:g} h" for time in times])
        ax.set(xlabel="time", ylabel=ylabel, title=title)
        if handles:
            ax.legend(handles=handles, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def dunn_tests(summary: pd.DataFrame, metrics: Sequence[str]
               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    omnibus_rows, pairwise_rows = [], []
    for metric in metrics:
        clean = summary[["condition", metric]].dropna()
        grouped = [(str(name), group[metric].to_numpy(float))
                   for name, group in clean.groupby("condition", sort=True) if len(group)]
        if len(grouped) < 2:
            omnibus_rows.append({"metric": metric, "n_groups": len(grouped),
                                  "h_statistic": math.nan, "p_value": math.nan})
            continue
        arrays = [values for _, values in grouped]
        all_values = np.concatenate(arrays)
        if np.all(all_values == all_values[0]):
            h_value, p_omnibus = 0.0, 1.0
        else:
            try:
                omnibus = stats.kruskal(*arrays, nan_policy="omit")
                h_value, p_omnibus = float(omnibus.statistic), float(omnibus.pvalue)
            except ValueError:
                h_value, p_omnibus = 0.0, 1.0
        omnibus_rows.append({
            "metric": metric, "n_groups": len(grouped),
            "total_biological_replicates": sum(len(values) for values in arrays),
            "h_statistic": h_value, "p_value": p_omnibus,
            "test": "Kruskal-Wallis", "statistical_unit": "biological replicate / image",
        })
        ranks = stats.rankdata(all_values, method="average")
        n_total = len(all_values)
        _, ties = np.unique(all_values, return_counts=True)
        tie_correction = 1 - float(np.sum(ties**3 - ties) / (n_total**3 - n_total))
        variance = n_total * (n_total + 1) / 12 * tie_correction
        means, start = [], 0
        for values in arrays:
            means.append(float(np.mean(ranks[start:start + len(values)])))
            start += len(values)
        indices = []
        for first_index in range(len(grouped)):
            for second_index in range(first_index + 1, len(grouped)):
                first_name, first_values = grouped[first_index]
                second_name, second_values = grouped[second_index]
                se = math.sqrt(variance * (1 / len(first_values) + 1 / len(second_values))) if variance > 0 else 0
                z_value = abs(means[first_index] - means[second_index]) / se if se else 0
                p_value = float(2 * stats.norm.sf(z_value)) if se else 1.0
                pairwise_rows.append({
                    "metric": metric, "group_1": first_name, "group_2": second_name,
                    "n_group_1": len(first_values), "n_group_2": len(second_values),
                    "mean_rank_group_1": means[first_index],
                    "mean_rank_group_2": means[second_index],
                    "z_statistic": z_value, "p_value": p_value,
                    "test": "two-sided Dunn test",
                    "statistical_unit": "biological replicate / image",
                })
                indices.append(len(pairwise_rows) - 1)
        adjusted = holm_adjust([pairwise_rows[index]["p_value"] for index in indices])
        for index, value in zip(indices, adjusted, strict=True):
            pairwise_rows[index]["p_value_holm_within_metric"] = value
    return pd.DataFrame(omnibus_rows), pd.DataFrame(pairwise_rows)


def aggregate_experiment(experiment: ExperimentSpec, cell_frames: Sequence[pd.DataFrame],
                         qc_rows: Sequence[dict[str, object]],
                         experiment_output: Path) -> None:
    if not cell_frames:
        return
    results_dir, qc_dir = experiment_output / "results", experiment_output / "qc"
    all_cells = pd.concat(cell_frames, ignore_index=True)
    summary = pd.DataFrame([sample_summary_from_cells(group)
                            for _, group in all_cells.groupby("sample_id", sort=True)])
    summary = summary.sort_values(["time_hours", "injury", "condition", "animal"],
                                  na_position="last")
    atomic_write_csv(all_cells, results_dir / "all_cells_long.csv")
    atomic_write_csv(wide_counts(all_cells, False),
                     results_dir / "all_cells_including_zero_wide.csv")
    atomic_write_csv(wide_counts(all_cells, True), results_dir / "nonzero_cells_wide.csv")
    atomic_write_csv(summary, results_dir / "biological_replicate_summary.csv")
    atomic_write_csv(pd.DataFrame(qc_rows), qc_dir / "sample_qc.csv")
    mean_metrics = ("mean_count_all_cells", "mean_count_nonzero_cells")
    atomic_write_csv(unpaired_ttests(summary, mean_metrics),
                     results_dir / "mean_count_unpaired_welch_ttests.csv")
    plot_biological_replicate_boxplots(
        summary, mean_metrics,
        ("Mean per-cell count: all cells", "Mean per-cell count: nonzero cells"),
        ("mean spots / cell", "mean spots / nonzero cell"),
        results_dir / "mean_count_biological_replicate_boxplots.png")
    threshold_metrics = tuple(f"percent_cells_gt_{value}" for value in THRESHOLDS)
    threshold_table = summary[["experiment", "gene", "condition", "time_hours",
                               "injury", "animal", "sample_id", "replicate_name",
                               *threshold_metrics]]
    atomic_write_csv(threshold_table, results_dir / "cell_percentage_gt_1_5_10.csv")
    atomic_write_csv(unpaired_ttests(summary, threshold_metrics),
                     results_dir / "cell_percentage_gt_1_5_10_unpaired_welch_ttests.csv")
    plot_biological_replicate_boxplots(
        summary, threshold_metrics,
        tuple(f"Cells with count > {value}" for value in THRESHOLDS),
        tuple("cells (%)" for _ in THRESHOLDS),
        results_dir / "cell_percentage_gt_1_5_10_boxplots.png")
    omnibus, dunn = dunn_tests(summary, (*mean_metrics, *threshold_metrics))
    atomic_write_csv(omnibus, results_dir / "kruskal_wallis_biological_replicates.csv")
    atomic_write_csv(dunn, results_dir / "dunn_test_biological_replicates.csv")
    if qc_rows:
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        x = np.arange(len(qc_rows))
        ax.scatter(x, [row.get("median_assignment_distance_um", math.nan) for row in qc_rows],
                   label="median", color="#4c78a8")
        ax.scatter(x, [row.get("p95_assignment_distance_um", math.nan) for row in qc_rows],
                   label="95th percentile", color="#f58518")
        ax.set_xticks(x, [str(row.get("replicate_name", row["animal"])) for row in qc_rows],
                      rotation=90)
        ax.set(ylabel="assignment distance (µm)",
               title=f"Experiment {experiment.name}: nearest-nucleus distance QC")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.savefig(qc_dir / "assignment_distance_by_sample.png", dpi=200,
                    bbox_inches="tight")
        plt.close(fig)


def process_sample(sample: SampleSpec, parameters: Parameters,
                   experiment_output: Path, overwrite: bool
                   ) -> tuple[pd.DataFrame, dict[str, object]]:
    result_dir = experiment_output / "results" / "samples" / sample.condition / sample.animal
    qc_dir = experiment_output / "qc" / "samples" / sample.condition / sample.animal
    example_dir = experiment_output / "example" / sample.condition / sample.animal
    completion = qc_dir / "COMPLETE.json"
    if completion.is_file() and not overwrite:
        LOGGER.info("Skipping completed sample: %s", sample.sample_id)
        cell_df = pd.read_csv(result_dir / "cell_counts.csv")
        with (qc_dir / "sample_qc.json").open(encoding="utf-8") as stream:
            return cell_df, json.load(stream)
    if overwrite:
        for directory in (result_dir, qc_dir, example_dir):
            if directory.exists():
                shutil.rmtree(directory)
    result_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    if parameters.save_examples:
        example_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Processing %s", sample.sample_id)
    label_dir, label_files = locate_cleaned_label_files(
        sample, parameters.cleaned_label_dirname)
    spot_file = locate_spot_file(sample.results_dir)
    spots = load_spots(spot_file)
    cell_df, spot_df, qc, counts = assign_spots_nearest_3d(sample, label_files, spots)
    qc.update({"replicate_name": sample.replicate_name,
               "cleaned_label_dir": str(label_dir), "spot_file": str(spot_file)})
    atomic_write_csv(cell_df, result_dir / "cell_counts.csv")
    atomic_write_csv(spot_df, result_dir / "spot_assignments_nearest_3d.csv")
    if parameters.save_assignment_maps:
        write_assignment_count_map(label_files, counts,
                                   result_dir / "assignment_count_map_3d.tif")
    if parameters.save_examples:
        metadata = create_example_images(sample, label_files, cell_df, spot_df, counts,
                                         example_dir, parameters)
        qc.update({
            "example_selected_cell_id": metadata["selected_cell_id"],
            "example_selected_cell_expression": metadata["selected_cell_expression"],
            "example_selection_rule": metadata["selection_rule"],
        })
    plot_distance_qc(spot_df, qc_dir / "assignment_distance_histogram.png",
                     sample.sample_id)
    atomic_write_json(qc, qc_dir / "sample_qc.json")
    atomic_write_json({
        "status": "completed", "script_version": SCRIPT_VERSION,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": asdict(parameters),
    }, completion)
    return cell_df, qc


def run_pipeline(experiments: Sequence[ExperimentSpec], parameters: Parameters,
                 output_root: Path, overwrite: bool, fail_fast: bool,
                 sample_regex: str | None) -> int:
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json({
        "script_version": SCRIPT_VERSION,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "experiments": [{"name": item.name, "gene": item.gene,
                         "root": str(item.root),
                         "voxel_xyz_nm": list(item.voxel_xyz_nm)} for item in experiments],
        "parameters": asdict(parameters),
        "assignment_rule": (
            "nearest cleaned nuclear voxel in calibrated 3-D; no maximum distance cutoff"),
        "statistics": {
            "unit": "biological replicate / image",
            "t_test": "two-sided unpaired Welch",
            "thresholds": [f"transcript_count > {value}" for value in THRESHOLDS],
            "multiple_testing": "Holm",
            "nonparametric": "Kruskal-Wallis followed by Dunn with Holm correction",
        },
    }, output_root / "run_configuration.json")
    pattern = re.compile(sample_regex) if sample_regex else None
    total_success, all_errors = 0, []
    for experiment in experiments:
        experiment_output = output_root / experiment.name
        for subdirectory in ("example", "results", "qc"):
            (experiment_output / subdirectory).mkdir(parents=True, exist_ok=True)
        samples = discover_samples(experiment, parameters.channel_folder)
        if pattern:
            samples = [sample for sample in samples if pattern.search(sample.sample_id)]
        LOGGER.info("Discovered %d samples for experiment %s", len(samples), experiment.name)
        cell_frames, qc_rows, experiment_errors = [], [], []
        for sample in samples:
            try:
                cell_df, qc = process_sample(sample, parameters,
                                             experiment_output, overwrite)
                cell_frames.append(cell_df)
                qc_rows.append(qc)
                total_success += 1
            except Exception as exc:
                LOGGER.exception("Failed sample %s", sample.sample_id)
                error = {"experiment": experiment.name, "sample_id": sample.sample_id,
                         "sample_path": str(sample.sample_dir),
                         "error_type": type(exc).__name__, "error": str(exc)}
                experiment_errors.append(error)
                all_errors.append(error)
                if fail_fast:
                    break
        if experiment_errors:
            atomic_write_csv(pd.DataFrame(experiment_errors),
                             experiment_output / "qc" / "processing_errors.csv")
        aggregate_experiment(experiment, cell_frames, qc_rows, experiment_output)
        if fail_fast and experiment_errors:
            break
    if all_errors:
        atomic_write_csv(pd.DataFrame(all_errors), output_root / "processing_errors_all.csv")
    LOGGER.info("Finished: %d successful, %d failed. Output: %s",
                total_success, len(all_errors), output_root)
    return 1 if all_errors else (0 if total_success else 2)


def save_2d_tiff(path: Path, array: np.ndarray) -> None:
    if tifffile_imread is not None:
        from tifffile import imwrite
        imwrite(path, array)
    else:
        from PIL import Image
        Image.fromarray(array).save(path)


def create_synthetic_experiment(base: Path) -> ExperimentSpec:
    experiment = ExperimentSpec("test", "test_gene", base / "synthetic_experiment",
                                (100.0, 100.0, 500.0))
    results = (experiment.root / "Experiment" / "6h_Amputation" /
               "Image1" / "565" / "results")
    label_dir = results / CLEANED_LABEL_DIRNAME
    label_dir.mkdir(parents=True)
    for z in range(3):
        plane = np.zeros((64, 80), dtype=np.uint16)
        if z <= 1:
            plane[10:20, 10:20] = 1
        if z >= 1:
            plane[40:50, 55:65] = 2
        save_2d_tiff(label_dir / f"Nucleus_Labels_{z:04d}.tif", plane)
    atomic_write_json({"status": "completed", "n_planes": 3},
                      label_dir / "cleaning_summary.json")
    spots = np.asarray([[0, 12, 12], [1, 18, 18], [2, 45, 60], [1, 38, 55]], float)
    np.save(results / "spots_post_decomposition_and_background_removed.npy", spots)
    raw = np.zeros((3, 64, 80), dtype=np.uint16)
    for z, y, x in spots.astype(int):
        raw[z, y, x] = 5000
    write_tiff_stack(results.parent / "synthetic_565.tif", raw)
    return experiment


def run_self_test() -> int:
    configure_logging(True)
    with tempfile.TemporaryDirectory(prefix="smfish_cellpose_assignment_test_") as temp:
        base = Path(temp)
        experiment = create_synthetic_experiment(base)
        output = base / "output"
        parameters = Parameters("565", CLEANED_LABEL_DIRNAME, 32, 17, 100, True, True)
        status = run_pipeline([experiment], parameters, output, True, True, None)
        if status:
            raise AssertionError(f"Synthetic pipeline returned {status}")
        cells = pd.read_csv(output / "test" / "results" / "all_cells_long.csv")
        if sorted(cells["transcript_count"].astype(int)) != [2, 2]:
            raise AssertionError(f"Unexpected counts: {cells['transcript_count'].tolist()}")
        assignments = pd.read_csv(output / "test" / "results" / "samples" /
                                  "6h_Amputation" / "Image1" /
                                  "spot_assignments_nearest_3d.csv")
        if not assignments["coordinate_valid"].all():
            raise AssertionError("Synthetic valid coordinates failed")
        example = output / "test" / "example" / "6h_Amputation" / "Image1" / "00_example_overview.png"
        if not example.is_file():
            raise AssertionError("Example overview missing")
        print("SELF-TEST PASSED")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", action="append", type=parse_experiment_argument,
                        help="Override: NAME,GENE,ROOT,X_NM,Y_NM,Z_NM (repeatable)")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--channel-folder", default="565")
    parser.add_argument("--cleaned-label-dirname", default=CLEANED_LABEL_DIRNAME)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--random-seed", type=int, default=20260831)
    parser.add_argument("--max-link-lines", type=int, default=1000)
    parser.add_argument("--sample-regex")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--no-assignment-maps", action="store_true")
    parser.add_argument("--no-examples", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return run_self_test()
    if args.crop_size <= 0 or args.max_link_lines <= 0:
        parser.error("--crop-size and --max-link-lines must be positive")
    configure_logging(args.verbose)
    experiments = args.experiment or [ExperimentSpec(name, gene, Path(root), voxel)
                                      for name, gene, root, voxel in DEFAULT_EXPERIMENTS]
    parameters = Parameters(args.channel_folder, args.cleaned_label_dirname,
                            args.crop_size, args.random_seed, args.max_link_lines,
                            not args.no_assignment_maps, not args.no_examples)
    return run_pipeline(experiments, parameters, args.output_root.expanduser(),
                        args.overwrite, args.fail_fast, args.sample_regex)


if __name__ == "__main__":
    raise SystemExit(main())
