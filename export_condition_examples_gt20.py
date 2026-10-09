#!/usr/bin/env python3
"""Export one 512x512 raw-image example per experiment and condition.

For every condition in experiments 304, 306, 307, 314, and 332, this script
draws a biological-replicate number from 1, 2, or 3. If that replicate is
missing, it falls back downward (3 -> 2 -> 1). Within the selected replicate it
randomly chooses a cell with ``transcript_count > 20``; if none exists, it
randomly chooses among the maximum-expression cells. It centers a 512x512 crop
on that nucleus and exports full 512x512 z-stack TIFFs:

1. original 565 intensity, yellow spot mask, and magenta-smFISH/yellow-spot merge;
2. original 405 intensity, inverted-Glasbey segmentation, and their merge;
3. original uint32 assignment-count map and its grayscale/yellow-spot merge.

Random choices are reproducible for a given ``--random-seed``.

Required packages: numpy, pandas, tifffile, colorcet.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import colorcet as cc
from tifffile import TiffFile, imread as tifffile_imread, imwrite as tifffile_imwrite


EXPERIMENTS = ("304", "306", "307", "314", "332")
CLEANED_LABEL_DIRNAME = (
    "cellpose_label_plane_s4_cp-1p75_f0p8_d90_n500_st0p25_cleaned"
)
LABEL_INDEX_PATTERN = re.compile(r"Nucleus_Labels_(\d+)")

DEFAULT_ASSIGNMENT_ROOT = Path(
    "/Volumes/Backup Plus/Experiment_results/smFISH_cellpose_assignment"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/Volumes/Backup Plus/Experiment_results/smFISH_condition_examples_gt20"
)
DEFAULT_EXPERIMENT_ROOTS = {
    "304": Path("/Volumes/Backup Plus/Experiment_results/304_Analysis_results"),
    "306": Path("/Volumes/Backup Plus/Experiment_results/306_analysis_results_alter"),
    "307": Path(
        "/Volumes/Backup Plus/Experiment_results/307_analysis_resutls/307_quantificatoin"
    ),
    "314": Path("/Volumes/Backup Plus/Experiment_results/314_analysis_results"),
    "332": Path("/Volumes/Backup Plus/Experiment_results/332_notum_analysis"),
}


@dataclass(frozen=True)
class Selection:
    experiment: str
    condition: str
    animal: str
    cell_id: int
    transcript_count: int
    example_z: int
    center_y: float
    center_x: float
    replicate_draw: int
    selection_rule: str
    cell_counts_path: Path
    spot_assignments_path: Path
    sample_dir: Path


def natural_key(value: str | Path) -> list[object]:
    return [
        int(piece) if piece.isdigit() else piece.lower()
        for piece in re.split(r"(\d+)", Path(value).name)
    ]


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "condition"


def atomic_write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=True)
    temporary.replace(path)


def atomic_write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(rows[0]) if rows else [
        "experiment", "condition", "status", "message"
    ]
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def discover_conditions(assignment_root: Path, experiment: str) -> list[Path]:
    samples_root = assignment_root / experiment / "results" / "samples"
    if not samples_root.is_dir():
        raise FileNotFoundError(f"Assignment samples folder is missing: {samples_root}")
    return sorted(
        [path for path in samples_root.iterdir() if path.is_dir()],
        key=natural_key,
    )


def stable_rng(experiment: str, condition: str, random_seed: int) -> np.random.Generator:
    digest = hashlib.sha256(
        f"{random_seed}|{experiment}|{condition}".encode("utf-8")
    ).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def trailing_number(value: str) -> int | None:
    match = re.search(r"(\d+)\s*$", value)
    return int(match.group(1)) if match else None


def choose_cell_for_condition(
    assignment_root: Path,
    experiment_root: Path,
    experiment: str,
    condition: str,
    threshold: int,
    random_seed: int,
) -> Selection:
    condition_root = (
        assignment_root / experiment / "results" / "samples" / condition
    )
    animal_dirs = sorted(
        [path for path in condition_root.iterdir() if path.is_dir()],
        key=natural_key,
    )
    usable = [
        path
        for path in animal_dirs
        if (path / "cell_counts.csv").is_file()
        and (path / "spot_assignments_nearest_3d.csv").is_file()
    ]
    if not usable:
        raise FileNotFoundError(f"No completed biological replicates in {condition_root}")

    by_number = {
        number: path
        for path in usable
        if (number := trailing_number(path.name)) is not None
    }
    rng = stable_rng(experiment, condition, random_seed)
    replicate_draw = int(rng.integers(1, 4))
    animal_dir = None
    for number in range(replicate_draw, 0, -1):
        if number in by_number:
            animal_dir = by_number[number]
            break
    if animal_dir is None:
        # Image1 normally exists. This final guard supports unusual sample names
        # without changing the requested 3 -> 2 -> 1 fallback when numbered
        # replicates are present.
        animal_dir = usable[0]

    cell_counts_path = animal_dir / "cell_counts.csv"
    cells = pd.read_csv(cell_counts_path)
    required = {
        "global_cell_id",
        "transcript_count",
        "example_z",
    }
    missing = required - set(cells.columns)
    if missing:
        raise ValueError(f"{cell_counts_path} is missing columns: {sorted(missing)}")
    if cells.empty:
        raise ValueError(f"No cells in {cell_counts_path}")

    eligible = cells[cells["transcript_count"] > threshold]
    if len(eligible):
        pool = eligible
        selection_rule = f"random_cell_with_transcript_count_gt_{threshold}"
    else:
        maximum = cells["transcript_count"].max()
        pool = cells[cells["transcript_count"] == maximum]
        selection_rule = "random_cell_among_maximum_expression_ties"
    row = pool.iloc[int(rng.integers(0, len(pool)))]
    animal = animal_dir.name
    cell_id = int(row["global_cell_id"])
    y_column = "example_y" if "example_y" in row.index else "centroid_y"
    x_column = "example_x" if "example_x" in row.index else "centroid_x"
    if y_column not in row.index or x_column not in row.index:
        raise ValueError(
            f"{cell_counts_path} has no example_y/example_x or centroid_y/centroid_x"
        )
    sample_dir = experiment_root / "Experiment" / condition / animal
    return Selection(
        experiment=experiment,
        condition=condition,
        animal=animal,
        cell_id=cell_id,
        transcript_count=int(row["transcript_count"]),
        example_z=int(row["example_z"]),
        center_y=float(row[y_column]),
        center_x=float(row[x_column]),
        replicate_draw=replicate_draw,
        selection_rule=selection_rule,
        cell_counts_path=cell_counts_path,
        spot_assignments_path=cell_counts_path.parent
        / "spot_assignments_nearest_3d.csv",
        sample_dir=sample_dir,
    )


def locate_channel_tiff(selection: Selection, channel: str) -> Path:
    channel_dir = selection.sample_dir / channel
    if not channel_dir.is_dir():
        raise FileNotFoundError(f"Channel folder is missing: {channel_dir}")
    candidates = [
        path
        for path in channel_dir.glob("*.tif*")
        if path.is_file() and not path.name.startswith("._")
    ]
    if not candidates:
        raise FileNotFoundError(f"No original {channel} TIFF in {channel_dir}")

    def score(path: Path) -> tuple[int, int, str]:
        name = path.name.lower()
        tokens = (selection.condition, selection.animal, channel)
        return (
            sum(token.lower() in name for token in tokens),
            path.stat().st_size,
            path.name,
        )

    return max(candidates, key=score)


def label_plane_index(path: Path) -> int:
    match = LABEL_INDEX_PATTERN.search(path.stem)
    if match is None:
        raise ValueError(f"Cannot parse z index from {path.name}")
    return int(match.group(1))


def locate_cleaned_label_files(selection: Selection) -> list[Path]:
    label_dir = selection.sample_dir / "565" / "results" / CLEANED_LABEL_DIRNAME
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Cleaned label folder is missing: {label_dir}")
    files = sorted(label_dir.glob("Nucleus_Labels_*.tif*"), key=natural_key)
    if not files:
        raise FileNotFoundError(f"No cleaned label planes in {label_dir}")
    indices = [label_plane_index(path) for path in files]
    if indices != list(range(indices[0], indices[0] + len(indices))):
        raise ValueError(f"Cleaned label z indices are not contiguous in {label_dir}")
    return files


def normalize_stack(array: np.ndarray) -> np.ndarray:
    stack = np.squeeze(np.asarray(array))
    if stack.ndim == 2:
        return stack[np.newaxis, ...]
    if stack.ndim != 3:
        raise ValueError(f"Expected a 2-D or 3-D single-channel TIFF; got {stack.shape}")
    z_axis = int(np.argmin(stack.shape))
    return np.moveaxis(stack, z_axis, 0)


def stack_shape(path: Path) -> tuple[int, int, int]:
    with TiffFile(path) as tif:
        series = tif.series[0]
        pages = list(series.pages)
        if pages and all(len(tuple(page.shape)) == 2 for page in pages):
            height, width = tuple(int(value) for value in pages[0].shape)
            return len(pages), height, width
        stack = normalize_stack(series.asarray())
        return tuple(int(value) for value in stack.shape)


def crop_bounds(center: float, crop_size: int, length: int) -> tuple[int, int]:
    if length < crop_size:
        raise ValueError(
            f"Image dimension {length} is smaller than requested crop {crop_size}"
        )
    start = int(round(center)) - crop_size // 2
    start = max(0, min(start, length - crop_size))
    return start, start + crop_size


def read_crop_stack(
    path: Path,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> np.ndarray:
    with TiffFile(path) as tif:
        series = tif.series[0]
        pages = list(series.pages)
        if pages and all(len(tuple(page.shape)) == 2 for page in pages):
            return np.stack(
                [np.asarray(page.asarray())[y0:y1, x0:x1] for page in pages],
                axis=0,
            )
        stack = normalize_stack(series.asarray())
        return stack[:, y0:y1, x0:x1]


def read_label_crop_stack(
    label_files: Sequence[Path], y0: int, y1: int, x0: int, x1: int
) -> tuple[np.ndarray, list[int]]:
    planes = []
    indices = []
    expected_shape = None
    for path in label_files:
        plane = np.squeeze(np.asarray(tifffile_imread(path)))
        if plane.ndim != 2:
            raise ValueError(f"Expected 2-D labels at {path}; got {plane.shape}")
        if expected_shape is None:
            expected_shape = plane.shape
        elif plane.shape != expected_shape:
            raise ValueError(f"Cleaned label shape mismatch at {path}")
        planes.append(plane[y0:y1, x0:x1].astype(np.int64, copy=False))
        indices.append(label_plane_index(path))
    return np.stack(planes, axis=0), indices


def contrast_to_uint8(image: np.ndarray) -> tuple[np.ndarray, float, float]:
    values = np.asarray(image, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if not finite.size:
        return np.zeros(values.shape, dtype=np.uint8), 0.0, 1.0
    low, high = np.percentile(finite, (1.0, 99.8))
    if high <= low:
        high = low + 1.0
    scaled = np.clip((values - low) / (high - low), 0, 1)
    return np.rint(scaled * 255).astype(np.uint8), float(low), float(high)


def atomic_write_tiff(path: Path, array: np.ndarray, axes: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.tif")
    kwargs: dict[str, object] = {
        "compression": "zlib",
        "metadata": {"axes": axes},
        "bigtiff": np.asarray(array).nbytes >= 2**32,
    }
    if axes.endswith("S"):
        kwargs["photometric"] = "rgb"
    else:
        kwargs["photometric"] = "minisblack"
    tifffile_imwrite(temporary, np.asarray(array), **kwargs)
    temporary.replace(path)


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def glasbey_inverted_stack(labels: np.ndarray) -> np.ndarray:
    palette = np.asarray(
        [hex_to_rgb(value) for value in reversed(cc.palette["glasbey"])],
        dtype=np.uint8,
    )
    rgb = palette[np.mod(labels, len(palette))]
    rgb[labels == 0] = 0
    return rgb


def make_spot_mask(
    spots: pd.DataFrame,
    z_count: int,
    crop_size: int,
    y0: int,
    x0: int,
) -> tuple[np.ndarray, int]:
    required = {"coordinate_valid", "spot_z", "spot_y", "spot_x"}
    missing = required - set(spots.columns)
    if missing:
        raise ValueError(f"Spot assignments are missing columns: {sorted(missing)}")
    valid_values = spots["coordinate_valid"]
    if pd.api.types.is_bool_dtype(valid_values):
        valid_mask = valid_values
    else:
        valid_mask = valid_values.astype(str).str.strip().str.lower().isin(
            {"true", "1", "yes"}
        )
    visible = spots[
        valid_mask
        & spots["spot_z"].between(0, z_count - 1)
        & spots["spot_y"].between(y0, y0 + crop_size - 1)
        & spots["spot_x"].between(x0, x0 + crop_size - 1)
    ]
    mask = np.zeros((z_count, crop_size, crop_size), dtype=np.uint8)
    offsets = [
        (dy, dx)
        for dy in range(-2, 3)
        for dx in range(-2, 3)
        if dy * dy + dx * dx <= 4
    ]
    for row in visible.itertuples(index=False):
        z = int(np.rint(row.spot_z))
        y = int(np.rint(row.spot_y)) - y0
        x = int(np.rint(row.spot_x)) - x0
        for dy, dx in offsets:
            yy, xx = y + dy, x + dx
            if 0 <= yy < crop_size and 0 <= xx < crop_size:
                mask[z, yy, xx] = 255
    return mask, len(visible)


def yellow_from_mask(mask: np.ndarray) -> np.ndarray:
    zeros = np.zeros_like(mask)
    return np.stack((mask, mask, zeros), axis=-1)


def magenta_from_gray(gray: np.ndarray) -> np.ndarray:
    zeros = np.zeros_like(gray)
    return np.stack((gray, zeros, gray), axis=-1)


def grayscale_rgb(gray: np.ndarray) -> np.ndarray:
    return np.repeat(gray[..., np.newaxis], 3, axis=-1)


def overlay_yellow(base_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    merged = np.asarray(base_rgb, dtype=np.uint8).copy()
    merged[mask > 0] = (255, 255, 0)
    return merged


def alpha_merge_segmentation(
    grayscale: np.ndarray, segmentation: np.ndarray, labels: np.ndarray, alpha: float = 0.45
) -> np.ndarray:
    merged = grayscale.copy().astype(np.float32)
    foreground = labels > 0
    merged[foreground] = (
        (1.0 - alpha) * merged[foreground] + alpha * segmentation[foreground]
    )
    return np.rint(np.clip(merged, 0, 255)).astype(np.uint8)


def export_selection(
    selection: Selection,
    output_root: Path,
    crop_size: int,
) -> dict[str, object]:
    smfish_path = locate_channel_tiff(selection, "565")
    nuclei_path = locate_channel_tiff(selection, "405")
    label_files = locate_cleaned_label_files(selection)
    smfish_shape = stack_shape(smfish_path)
    nuclei_shape = stack_shape(nuclei_path)
    if smfish_shape != nuclei_shape:
        raise ValueError(
            f"405/565 z-stack shapes differ for {selection.sample_dir}: "
            f"{nuclei_shape} versus {smfish_shape}"
        )
    z_count, height, width = smfish_shape
    label_indices = [label_plane_index(path) for path in label_files]
    if label_indices != list(range(z_count)):
        raise ValueError(
            f"Cleaned labels must map one-to-one to raw zero-based z planes. "
            f"Raw z=0..{z_count - 1}; label indices start/end="
            f"{label_indices[0]}/{label_indices[-1]}"
        )
    y0, y1 = crop_bounds(selection.center_y, crop_size, height)
    x0, x1 = crop_bounds(selection.center_x, crop_size, width)
    smfish = read_crop_stack(smfish_path, y0, y1, x0, x1)
    nuclei = read_crop_stack(nuclei_path, y0, y1, x0, x1)
    labels, read_label_indices = read_label_crop_stack(
        label_files, y0, y1, x0, x1
    )
    expected_crop_shape = (z_count, crop_size, crop_size)
    for name, array in (("smFISH", smfish), ("nuclei", nuclei), ("labels", labels)):
        if array.shape != expected_crop_shape:
            raise ValueError(f"Unexpected {name} crop stack shape: {array.shape}")
    if read_label_indices != label_indices:
        raise RuntimeError("Label z ordering changed while reading")

    spots = pd.read_csv(selection.spot_assignments_path)
    spot_mask, visible_spots = make_spot_mask(
        spots, z_count, crop_size, y0, x0
    )
    spot_yellow = yellow_from_mask(spot_mask)

    cells = pd.read_csv(selection.cell_counts_path)
    max_cell_id = max(int(labels.max()), int(cells["global_cell_id"].max()))
    counts_by_cell = np.zeros(max_cell_id + 1, dtype=np.uint32)
    for row in cells.itertuples(index=False):
        counts_by_cell[int(row.global_cell_id)] = int(row.transcript_count)
    assignment = counts_by_cell[labels].astype(np.uint32, copy=False)

    smfish_u8, smfish_low, smfish_high = contrast_to_uint8(smfish)
    nuclei_u8, nuclei_low, nuclei_high = contrast_to_uint8(nuclei)
    assignment_u8, assignment_low, assignment_high = contrast_to_uint8(assignment)
    smfish_magenta = magenta_from_gray(smfish_u8)
    smfish_spot_merged = overlay_yellow(smfish_magenta, spot_mask)
    nuclei_gray_rgb = grayscale_rgb(nuclei_u8)
    segmentation_glasbey = glasbey_inverted_stack(labels)
    nuclei_segmentation_merged = alpha_merge_segmentation(
        nuclei_gray_rgb, segmentation_glasbey, labels, alpha=0.45
    )
    assignment_gray_rgb = grayscale_rgb(assignment_u8)
    spot_assignment_merged = overlay_yellow(assignment_gray_rgb, spot_mask)

    condition_name = safe_name(selection.condition)
    prefix = f"{selection.experiment}_{condition_name}"
    output_dir = output_root / selection.experiment / selection.condition
    output_dir.mkdir(parents=True, exist_ok=True)
    # Remove only this script's obsolete PNG previews from the prior version.
    for obsolete_png in output_dir.glob(f"{prefix}_*.png"):
        obsolete_png.unlink()
    outputs = {
        "smfish_original": output_dir / f"{prefix}_01_smFISH_original_raw_zstack.tif",
        "smfish_magenta": output_dir / f"{prefix}_02_smFISH_magenta_zstack.tif",
        "spot_yellow": output_dir / f"{prefix}_03_spot_detection_yellow_zstack.tif",
        "smfish_spot_merge": output_dir
        / f"{prefix}_04_smFISH_magenta_spot_yellow_merged_zstack.tif",
        "nuclei_original": output_dir / f"{prefix}_05_nuclei_original_raw_zstack.tif",
        "segmentation": output_dir
        / f"{prefix}_06_nuclei_segmentation_glasbey_inverted_zstack.tif",
        "nuclei_segmentation_merge": output_dir
        / f"{prefix}_07_nuclei_grayscale_segmentation_merged_zstack.tif",
        "assignment_raw": output_dir
        / f"{prefix}_08_assignment_map_grayscale_raw_uint32_zstack.tif",
        "spot_assignment_merge": output_dir
        / f"{prefix}_09_spot_yellow_assignment_grayscale_merged_zstack.tif",
    }
    atomic_write_tiff(outputs["smfish_original"], smfish, "ZYX")
    atomic_write_tiff(outputs["smfish_magenta"], smfish_magenta, "ZYXS")
    atomic_write_tiff(outputs["spot_yellow"], spot_yellow, "ZYXS")
    atomic_write_tiff(outputs["smfish_spot_merge"], smfish_spot_merged, "ZYXS")
    atomic_write_tiff(outputs["nuclei_original"], nuclei, "ZYX")
    atomic_write_tiff(outputs["segmentation"], segmentation_glasbey, "ZYXS")
    atomic_write_tiff(
        outputs["nuclei_segmentation_merge"], nuclei_segmentation_merged, "ZYXS"
    )
    atomic_write_tiff(outputs["assignment_raw"], assignment, "ZYX")
    atomic_write_tiff(
        outputs["spot_assignment_merge"], spot_assignment_merged, "ZYXS"
    )
    metadata = {
        "experiment": selection.experiment,
        "condition": selection.condition,
        "animal": selection.animal,
        "selected_cell_id": selection.cell_id,
        "selected_cell_transcript_count": selection.transcript_count,
        "replicate_random_draw_1_to_3": selection.replicate_draw,
        "replicate_fallback_rule": "drawn replicate, then decrement toward Image1",
        "selection_rule": selection.selection_rule,
        "selected_cell_reference_z": selection.example_z,
        "crop_size": crop_size,
        "z_planes": z_count,
        "crop_y0_y1": [y0, y1],
        "crop_x0_x1": [x0, x1],
        "smfish_source": str(smfish_path),
        "nuclei_source": str(nuclei_path),
        "cleaned_segmentation_sources": [str(path) for path in label_files],
        "cell_counts_source": str(selection.cell_counts_path),
        "spot_assignments_source": str(selection.spot_assignments_path),
        "visible_spots_in_crop": visible_spots,
        "smfish_display_percentile_range": [smfish_low, smfish_high],
        "nuclei_display_percentile_range": [nuclei_low, nuclei_high],
        "assignment_display_percentile_range": [assignment_low, assignment_high],
        "smfish_color": "magenta RGB=(I,0,I)",
        "spot_detection_color": "yellow RGB=(255,255,0)",
        "segmentation_color": "Colorcet glasbey palette in reversed order",
        "nuclei_segmentation_alpha": 0.45,
        "raw_tiff_values_preserved": True,
        "output_axes": {"grayscale": "ZYX", "rgb": "ZYXS"},
        "outputs": {name: path.name for name, path in outputs.items()},
    }
    atomic_write_json(metadata, output_dir / f"{prefix}_selection_metadata.json")
    return metadata


def run_export(
    assignment_root: Path,
    output_root: Path,
    experiment_roots: dict[str, Path],
    threshold: int,
    crop_size: int,
    random_seed: int,
) -> int:
    output_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    errors = 0
    for experiment in EXPERIMENTS:
        try:
            condition_dirs = discover_conditions(assignment_root, experiment)
        except Exception as exc:
            print(f"ERROR {experiment}: {type(exc).__name__}: {exc}", flush=True)
            summary_rows.append(
                {
                    "experiment": experiment,
                    "condition": "",
                    "status": "error",
                    "message": str(exc),
                }
            )
            errors += 1
            continue
        for condition_dir in condition_dirs:
            condition = condition_dir.name
            try:
                selection = choose_cell_for_condition(
                    assignment_root,
                    experiment_roots[experiment],
                    experiment,
                    condition,
                    threshold,
                    random_seed,
                )
                metadata = export_selection(selection, output_root, crop_size)
                print(
                    f"WROTE {experiment} {condition} | {selection.animal} | "
                    f"replicate draw={selection.replicate_draw} | "
                    f"cell={selection.cell_id} | expression={selection.transcript_count} | "
                    f"rule={selection.selection_rule}",
                    flush=True,
                )
                summary_rows.append(
                    {
                        "experiment": experiment,
                        "condition": condition,
                        "status": "completed",
                        "message": "",
                        "animal": selection.animal,
                        "replicate_random_draw": selection.replicate_draw,
                        "cell_selection_rule": selection.selection_rule,
                        "selected_cell_id": selection.cell_id,
                        "transcript_count": selection.transcript_count,
                        "crop_y0": metadata["crop_y0_y1"][0],
                        "crop_y1": metadata["crop_y0_y1"][1],
                        "crop_x0": metadata["crop_x0_x1"][0],
                        "crop_x1": metadata["crop_x0_x1"][1],
                    }
                )
            except Exception as exc:
                print(
                    f"ERROR {experiment} {condition}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                summary_rows.append(
                    {
                        "experiment": experiment,
                        "condition": condition,
                        "status": "error",
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                )
                errors += 1
    # Normalize columns because completed rows contain additional selection data.
    all_fields = [
        "experiment",
        "condition",
        "status",
        "message",
        "animal",
        "replicate_random_draw",
        "cell_selection_rule",
        "selected_cell_id",
        "transcript_count",
        "crop_y0",
        "crop_y1",
        "crop_x0",
        "crop_x1",
    ]
    normalized = [{field: row.get(field, "") for field in all_fields} for row in summary_rows]
    atomic_write_csv(normalized, output_root / "selection_summary.csv")
    completed = sum(row["status"] == "completed" for row in summary_rows)
    print(
        f"FINISHED: completed={completed}, errors={errors}, output={output_root}",
        flush=True,
    )
    return 1 if errors else 0


def write_test_stack(path: Path, stack: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile_imwrite(
        path,
        np.asarray(stack),
        photometric="minisblack",
        metadata={"axes": "ZYX"},
    )


def run_self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="condition_example_gt20_test_") as temporary:
        base = Path(temporary)
        assignment_root = base / "assignments"
        output_root = base / "output"
        experiment_roots = {
            experiment: base / f"experiment_{experiment}" for experiment in EXPERIMENTS
        }
        for experiment in EXPERIMENTS:
            condition = "6h_Amputation"
            animal = "Image1"
            result_dir = (
                assignment_root
                / experiment
                / "results"
                / "samples"
                / condition
                / animal
            )
            result_dir.mkdir(parents=True)
            counts = [5, 20] if experiment == "304" else [5, 25]
            pd.DataFrame(
                {
                    "global_cell_id": [1, 2],
                    "transcript_count": counts,
                    "example_z": [1, 1],
                    "example_y": [20.0, 40.0],
                    "example_x": [20.0, 50.0],
                }
            ).to_csv(result_dir / "cell_counts.csv", index=False)
            pd.DataFrame(
                {
                    "coordinate_valid": [True, True],
                    "spot_z": [1.0, 0.0],
                    "spot_y": [40.0, 15.0],
                    "spot_x": [50.0, 15.0],
                    "assigned_cell_id": [2, 1],
                }
            ).to_csv(result_dir / "spot_assignments_nearest_3d.csv", index=False)
            sample_dir = experiment_roots[experiment] / "Experiment" / condition / animal
            stack_565 = np.zeros((3, 72, 96), dtype=np.uint16)
            stack_405 = np.zeros_like(stack_565)
            stack_565[1, 40, 50] = 5000
            stack_405[:, 35:45, 45:55] = 3000
            write_test_stack(sample_dir / "565" / f"{animal}_565.tif", stack_565)
            write_test_stack(sample_dir / "405" / f"{animal}_405.tif", stack_405)
            label_dir = (
                sample_dir / "565" / "results" / CLEANED_LABEL_DIRNAME
            )
            label_dir.mkdir(parents=True)
            for z_index in range(3):
                label = np.zeros((72, 96), dtype=np.uint16)
                label[15:25, 15:25] = 1
                label[35:45, 45:55] = 2
                tifffile_imwrite(
                    label_dir / f"Nucleus_Labels_{z_index:04d}.tif",
                    label,
                    photometric="minisblack",
                )

        status = run_export(
            assignment_root,
            output_root,
            experiment_roots,
            threshold=20,
            crop_size=32,
            random_seed=20260901,
        )
        if status != 0:
            raise AssertionError(f"Self-test returned {status}")
        for experiment in EXPERIMENTS:
            folder = output_root / experiment / "6h_Amputation"
            tiffs = sorted(folder.glob("*.tif"))
            if len(tiffs) != 9:
                raise AssertionError(f"Expected nine TIFFs for {experiment}; got {tiffs}")
            for image_path in tiffs:
                array = np.asarray(tifffile_imread(image_path))
                if array.shape not in {(3, 32, 32), (3, 32, 32, 3)}:
                    raise AssertionError(
                        f"Wrong z-stack crop shape {array.shape}: {image_path}"
                    )
            assignment_path = next(
                folder.glob("*_08_assignment_map_grayscale_raw_uint32_zstack.tif")
            )
            if tifffile_imread(assignment_path).dtype != np.uint32:
                raise AssertionError("Raw assignment map must remain uint32")
            magenta = tifffile_imread(next(folder.glob("*_02_smFISH_magenta_zstack.tif")))
            yellow = tifffile_imread(
                next(folder.glob("*_03_spot_detection_yellow_zstack.tif"))
            )
            merged = tifffile_imread(
                next(folder.glob("*_04_smFISH_magenta_spot_yellow_merged_zstack.tif"))
            )
            if np.any(magenta[..., 1] != 0) or not np.array_equal(
                magenta[..., 0], magenta[..., 2]
            ):
                raise AssertionError("smFISH color stack is not magenta")
            yellow_mask = yellow[..., 0] > 0
            if not np.any(yellow_mask):
                raise AssertionError("Yellow spot-detection stack is empty")
            if not np.all(yellow[yellow_mask] == (255, 255, 0)):
                raise AssertionError("Spot-detection color is not yellow")
            if not np.all(merged[yellow_mask] == (255, 255, 0)):
                raise AssertionError("Merged spot overlay must remain yellow")
            if list(folder.glob("*.png")):
                raise AssertionError("PNG output is forbidden")
        summary = pd.read_csv(output_root / "selection_summary.csv")
        if len(summary) != 5 or not (summary["status"] == "completed").all():
            raise AssertionError("Unexpected self-test selection summary")
        metadata_path = (
            output_root
            / "304"
            / "6h_Amputation"
            / "304_6h_Amputation_selection_metadata.json"
        )
        with metadata_path.open(encoding="utf-8") as stream:
            metadata_304 = json.load(stream)
        if metadata_304["selected_cell_transcript_count"] != 20:
            raise AssertionError("No->20 fallback must choose the maximum-expression cell")
        if metadata_304["selection_rule"] != "random_cell_among_maximum_expression_ties":
            raise AssertionError("Maximum-expression fallback rule was not recorded")
        print("SELF-TEST PASSED", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment-root", type=Path, default=DEFAULT_ASSIGNMENT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--threshold", type=int, default=20)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--random-seed", type=int, default=20260901)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return run_self_test()
    if args.threshold < 0:
        raise SystemExit("--threshold must be >= 0")
    if args.crop_size <= 0:
        raise SystemExit("--crop-size must be > 0")
    return run_export(
        args.assignment_root.expanduser(),
        args.output_root.expanduser(),
        DEFAULT_EXPERIMENT_ROOTS,
        args.threshold,
        args.crop_size,
        args.random_seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
