#!/usr/bin/env python3
"""Remove fragmented Cellpose edge artifacts from an existing 3-D label stack.

The cleaner is intended for stitched ``Nucleus_Labels_*.tif`` planes produced
by ``cellpose_sam_nuclei_3d.py``.  It never changes surviving global label IDs
and never overwrites the input directory.

Two complementary tests are applied at matched physical scales:

1. 3-D reliability: remove global objects that never reach a plausible nuclear
   cross-section or occur in too few z-planes.
2. 2-D fragment clusters: remove plane-specific groups made from many adjacent,
   undersized labels.  This catches the characteristic striped Cellpose masks
   produced in dark image-edge regions, including fragments accidentally
   stitched to an otherwise valid global ID.

Voxel sizes are inferred from experiment IDs 304, 306, 307, 314, and 332 in
the input path (or supplied explicitly).  Physical thresholds are converted to
pixel/plane thresholds for each stack, so 64-nm and 74-nm XY data are treated
consistently.  Use ``--all-experiments`` to search the configured experiment
roots and clean every final Cellpose output, or ``--input-dir`` for one stack.
The output directory contains cleaned label TIFFs plus CSV/JSON audit files.
Run with ``--dry-run`` first to inspect the reported removal counts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from scipy import ndimage
except ImportError as exc:  # pragma: no cover - dependency check on user's Mac
    raise SystemExit("Install scipy in this environment: python -m pip install scipy") from exc

try:
    from tifffile import imread, imwrite
except ImportError:  # permit dependency-light unit tests of array functions
    imread = None
    imwrite = None


SCRIPT_VERSION = "1.2.0"
LABEL_PATTERN = re.compile(r"Nucleus_Labels_(\d+)")
DEFAULT_LABEL_DIRNAME = "cellpose_label_plane_s4_cp-1p75_f0p8_d90_n500_st0p25"

EXPERIMENT_ROOTS: dict[str, Path] = {
    "304": Path("/Volumes/Backup Plus/Experiment_results/304_Analysis_results"),
    "306": Path(
        "/Volumes/Backup Plus/Experiment_results/306_analysis_results_alter"
    ),
    "307": Path(
        "/Volumes/Backup Plus/Experiment_results/"
        "307_analysis_resutls/307_quantificatoin"
    ),
    "314": Path("/Volumes/Backup Plus/Experiment_results/314_analysis_results"),
    "332": Path("/Volumes/Backup Plus/Experiment_results/332_notum_analysis"),
}

# Input convention is (X, Y, Z), matching the microscope voxel metadata.
EXPERIMENT_VOXEL_SIZE_XYZ_NM: dict[str, tuple[float, float, float]] = {
    "304": (64.0, 64.0, 361.0),
    "306": (64.0, 64.0, 361.0),
    "307": (64.0, 64.0, 361.0),
    "314": (74.0, 74.0, 500.0),
    "332": (74.0, 74.0, 500.0),
}
EXPERIMENT_ID_PATTERN = re.compile(
    rf"(?<!\d)({'|'.join(EXPERIMENT_VOXEL_SIZE_XYZ_NM)})(?!\d)"
)

# Preserve the original 64-nm-data criteria, expressed in physical units.
# 3000 px * (0.064 um)^2 = 12.288 um^2
# 1500 px * (0.064 um)^2 =  6.144 um^2
# 2 planes * 0.361 um       =  0.722 um sampled z thickness
DEFAULT_MIN_MAX_PLANE_AREA_UM2 = 12.288
DEFAULT_MIN_Z_THICKNESS_UM = 0.722
DEFAULT_SMALL_OBJECT_MAX_AREA_UM2 = 12.288
DEFAULT_CLUSTER_MIN_AREA_UM2 = 6.144
DEFAULT_BORDER_MAX_AREA_UM2 = 12.288


@dataclass(frozen=True)
class Settings:
    experiment_id: str | None
    voxel_size_xyz_nm: tuple[float, float, float]
    pixel_area_um2: float
    voxel_volume_um3: float
    min_max_plane_area: int
    min_max_plane_area_um2: float
    min_z_planes: int
    min_z_thickness_um: float
    small_object_max_area: int
    small_object_max_area_um2: float
    cluster_min_labels: int
    cluster_min_pixels: int
    cluster_min_area_um2: float
    remove_small_border_slivers: bool
    border_max_area: int
    border_max_area_um2: float
    border_min_aspect_ratio: float
    compression: str


@dataclass(frozen=True)
class GlobalObjectStats:
    global_id: int
    n_z_planes: int
    max_plane_area: int
    total_voxels: int


def infer_experiment_id(path: Path) -> str | None:
    """Return the nearest recognized experiment ID in a path, if present."""

    for part in reversed(path.parts):
        match = EXPERIMENT_ID_PATTERN.search(part)
        if match:
            return match.group(1)
    return None


def area_um2_to_pixels(area_um2: float, pixel_area_um2: float) -> int:
    if area_um2 <= 0:
        return 0
    # Nearest whole pixel keeps the calibrated physical cutoff closest.
    return max(1, int(math.floor(area_um2 / pixel_area_um2 + 0.5)))


def thickness_um_to_planes(thickness_um: float, voxel_z_um: float) -> int:
    if thickness_um <= 0:
        return 1
    # Subtract a tiny tolerance so exact ratios (for example 0.722/0.361)
    # do not round upward because of floating-point representation.
    return max(1, int(math.ceil(thickness_um / voxel_z_um - 1e-12)))


def build_settings(args: argparse.Namespace, input_dir: Path) -> Settings:
    experiment_id = args.experiment_id or infer_experiment_id(input_dir)
    if args.voxel_size_nm is not None:
        voxel_size_xyz_nm = tuple(float(value) for value in args.voxel_size_nm)
    else:
        if experiment_id is None:
            known = ", ".join(EXPERIMENT_VOXEL_SIZE_XYZ_NM)
            raise SystemExit(
                "Could not infer an experiment ID from --input-dir. "
                f"Use --experiment-id ({known}) or --voxel-size-nm X Y Z."
            )
        voxel_size_xyz_nm = EXPERIMENT_VOXEL_SIZE_XYZ_NM[experiment_id]

    voxel_x_um, voxel_y_um, voxel_z_um = (
        value / 1000.0 for value in voxel_size_xyz_nm
    )
    pixel_area_um2 = voxel_x_um * voxel_y_um
    voxel_volume_um3 = pixel_area_um2 * voxel_z_um

    min_max_plane_area = (
        args.min_max_plane_area
        if args.min_max_plane_area is not None
        else area_um2_to_pixels(args.min_max_plane_area_um2, pixel_area_um2)
    )
    min_z_planes = (
        args.min_z_planes
        if args.min_z_planes is not None
        else thickness_um_to_planes(args.min_z_thickness_um, voxel_z_um)
    )
    small_object_max_area = (
        args.small_object_max_area
        if args.small_object_max_area is not None
        else area_um2_to_pixels(args.small_object_max_area_um2, pixel_area_um2)
    )
    cluster_min_pixels = (
        args.cluster_min_pixels
        if args.cluster_min_pixels is not None
        else area_um2_to_pixels(args.cluster_min_area_um2, pixel_area_um2)
    )
    border_max_area = (
        args.border_max_area
        if args.border_max_area is not None
        else area_um2_to_pixels(args.border_max_area_um2, pixel_area_um2)
    )

    return Settings(
        experiment_id=experiment_id,
        voxel_size_xyz_nm=voxel_size_xyz_nm,
        pixel_area_um2=pixel_area_um2,
        voxel_volume_um3=voxel_volume_um3,
        min_max_plane_area=min_max_plane_area,
        min_max_plane_area_um2=min_max_plane_area * pixel_area_um2,
        min_z_planes=min_z_planes,
        min_z_thickness_um=min_z_planes * voxel_z_um,
        small_object_max_area=small_object_max_area,
        small_object_max_area_um2=small_object_max_area * pixel_area_um2,
        cluster_min_labels=args.cluster_min_labels,
        cluster_min_pixels=cluster_min_pixels,
        cluster_min_area_um2=cluster_min_pixels * pixel_area_um2,
        remove_small_border_slivers=args.remove_small_border_slivers,
        border_max_area=border_max_area,
        border_max_area_um2=border_max_area * pixel_area_um2,
        border_min_aspect_ratio=args.border_min_aspect_ratio,
        compression=args.compression,
    )


def natural_label_key(path: Path) -> tuple[int, str]:
    match = LABEL_PATTERN.search(path.name)
    return (int(match.group(1)) if match else sys.maxsize, path.name)


def discover_label_planes(input_dir: Path, pattern: str) -> list[Path]:
    files = sorted(
        (
            path
            for path in input_dir.glob(pattern)
            if path.is_file() and not path.name.startswith("._")
        ),
        key=natural_label_key,
    )
    if not files:
        raise FileNotFoundError(f"No label TIFFs matched {pattern!r} in {input_dir}")
    indices = [natural_label_key(path)[0] for path in files]
    if any(index == sys.maxsize for index in indices):
        raise ValueError("Every label filename must contain Nucleus_Labels_<integer>")
    if indices != list(range(indices[0], indices[0] + len(indices))):
        raise ValueError(f"Label plane numbers are not contiguous: {indices[:20]}")
    return files


def read_labels(path: Path) -> np.ndarray:
    if imread is None:
        raise SystemExit("Install tifffile: python -m pip install tifffile")
    labels = np.asarray(imread(path))
    labels = np.squeeze(labels)
    if labels.ndim != 2:
        raise ValueError(f"Expected one 2-D label plane in {path}; got {labels.shape}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError(f"Label TIFF must be integer-valued: {path} has {labels.dtype}")
    if np.any(labels < 0):
        raise ValueError(f"Negative labels found in {path}")
    return labels.astype(np.uint32, copy=False)


def label_counts(labels: np.ndarray) -> np.ndarray:
    return np.bincount(labels.ravel().astype(np.int64, copy=False))


def collect_global_stats(
    files: Sequence[Path],
) -> tuple[dict[int, GlobalObjectStats], tuple[int, int]]:
    n_planes: defaultdict[int, int] = defaultdict(int)
    max_area: defaultdict[int, int] = defaultdict(int)
    total_voxels: defaultdict[int, int] = defaultdict(int)
    expected_shape: tuple[int, int] | None = None

    for plane_number, path in enumerate(files, start=1):
        labels = read_labels(path)
        if expected_shape is None:
            expected_shape = tuple(int(value) for value in labels.shape)
        elif labels.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch at {path}: {labels.shape} versus {expected_shape}"
            )
        counts = label_counts(labels)
        present = np.flatnonzero(counts[1:] > 0) + 1
        for global_id_raw in present:
            global_id = int(global_id_raw)
            area = int(counts[global_id])
            n_planes[global_id] += 1
            total_voxels[global_id] += area
            max_area[global_id] = max(max_area[global_id], area)
        print(
            f"Pass 1/2: {plane_number}/{len(files)} {path.name} "
            f"({len(present)} objects)",
            flush=True,
        )

    if expected_shape is None:  # guarded by discovery, retained for type safety
        raise ValueError("No labels were read")
    stats = {
        global_id: GlobalObjectStats(
            global_id=global_id,
            n_z_planes=n_planes[global_id],
            max_plane_area=max_area[global_id],
            total_voxels=total_voxels[global_id],
        )
        for global_id in sorted(n_planes)
    }
    return stats, expected_shape


def global_removal_reasons(
    stats: dict[int, GlobalObjectStats], settings: Settings
) -> dict[int, tuple[str, ...]]:
    reasons: dict[int, tuple[str, ...]] = {}
    for global_id, record in stats.items():
        current: list[str] = []
        if (
            settings.min_max_plane_area > 0
            and record.max_plane_area < settings.min_max_plane_area
        ):
            current.append("max_plane_area_below_threshold")
        if settings.min_z_planes > 1 and record.n_z_planes < settings.min_z_planes:
            current.append("z_span_below_threshold")
        if current:
            reasons[global_id] = tuple(current)
    return reasons


def plane_fragment_reasons(
    labels: np.ndarray, settings: Settings
) -> dict[int, set[str]]:
    """Return plane-local global IDs flagged by cluster/border geometry."""

    counts = label_counts(labels)
    present = np.flatnonzero(counts[1:] > 0) + 1
    reasons: defaultdict[int, set[str]] = defaultdict(set)

    small_ids = present[counts[present] <= settings.small_object_max_area]
    if len(small_ids):
        small_mask = np.isin(labels, small_ids)
        components, n_components = ndimage.label(
            small_mask, structure=np.ones((3, 3), dtype=np.uint8)
        )
        component_counts = np.bincount(components.ravel())
        component_boxes = ndimage.find_objects(components)
        for component_id in range(1, n_components + 1):
            if component_counts[component_id] < settings.cluster_min_pixels:
                continue
            slices = component_boxes[component_id - 1]
            if slices is None:
                continue
            component_ids = np.unique(
                labels[slices][components[slices] == component_id]
            )
            component_ids = component_ids[component_ids > 0]
            if len(component_ids) < settings.cluster_min_labels:
                continue
            for global_id_raw in component_ids:
                reasons[int(global_id_raw)].add("adjacent_small_mask_cluster")

    if settings.remove_small_border_slivers and len(present):
        height, width = labels.shape
        boxes = ndimage.find_objects(labels)
        for global_id_raw in present:
            global_id = int(global_id_raw)
            if counts[global_id] > settings.border_max_area:
                continue
            slices = boxes[global_id - 1]
            if slices is None:
                continue
            y_slice, x_slice = slices
            box_height = y_slice.stop - y_slice.start
            box_width = x_slice.stop - x_slice.start
            aspect_ratio = max(
                box_width / max(1, box_height),
                box_height / max(1, box_width),
            )
            touches_border = (
                x_slice.start == 0
                or y_slice.start == 0
                or x_slice.stop == width
                or y_slice.stop == height
            )
            if touches_border and aspect_ratio >= settings.border_min_aspect_ratio:
                reasons[global_id].add("small_elongated_border_mask")

    return dict(reasons)


def write_label_plane(path: Path, labels: np.ndarray, compression: str) -> None:
    if imwrite is None:
        raise SystemExit("Install tifffile: python -m pip install tifffile")
    kwargs: dict[str, object] = {"photometric": "minisblack", "metadata": None}
    if compression == "zlib":
        kwargs["compression"] = "zlib"
        kwargs["compressionargs"] = {"level": 1}
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    imwrite(temporary, labels.astype(np.uint32, copy=False), **kwargs)
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[dict[str, object]], fields: Sequence[str]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_write_json(path: Path, data: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")
    temporary.replace(path)


def clean_stack(
    files: Sequence[Path],
    input_dir: Path,
    output_dir: Path,
    settings: Settings,
    dry_run: bool,
) -> dict[str, object]:
    print(
        "Physical calibration: "
        f"experiment={settings.experiment_id or 'custom'}, "
        f"voxel XYZ={settings.voxel_size_xyz_nm} nm, "
        f"min cross-section={settings.min_max_plane_area} px "
        f"({settings.min_max_plane_area_um2:.4f} um^2), "
        f"min z={settings.min_z_planes} planes "
        f"({settings.min_z_thickness_um:.3f} um), "
        f"cluster min={settings.cluster_min_pixels} px "
        f"({settings.cluster_min_area_um2:.4f} um^2)",
        flush=True,
    )
    stats, shape = collect_global_stats(files)
    global_reasons = global_removal_reasons(stats, settings)
    global_remove_ids = np.asarray(sorted(global_reasons), dtype=np.uint32)

    if not dry_run:
        if output_dir.resolve() == input_dir.resolve():
            raise ValueError("Output directory must differ from input directory")
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. Choose a new directory."
            )
        output_dir.mkdir(parents=True, exist_ok=True)

    global_rows = [
        {
            "global_id": global_id,
            "n_z_planes": stats[global_id].n_z_planes,
            "sampled_z_thickness_um": (
                stats[global_id].n_z_planes
                * settings.voxel_size_xyz_nm[2]
                / 1000.0
            ),
            "max_plane_area": stats[global_id].max_plane_area,
            "max_plane_area_um2": (
                stats[global_id].max_plane_area * settings.pixel_area_um2
            ),
            "total_voxels": stats[global_id].total_voxels,
            "total_volume_um3": (
                stats[global_id].total_voxels * settings.voxel_volume_um3
            ),
            "reasons": ";".join(global_reasons[global_id]),
        }
        for global_id in sorted(global_reasons)
    ]
    fragment_rows: list[dict[str, object]] = []
    plane_rows: list[dict[str, object]] = []
    total_removed_pixels = 0
    total_cluster_fragment_pixels = 0

    for position, path in enumerate(files):
        labels = read_labels(path)
        counts = label_counts(labels)
        plane_reasons = plane_fragment_reasons(labels, settings)
        plane_remove_ids = np.asarray(sorted(plane_reasons), dtype=np.uint32)

        remove_mask = np.zeros(labels.shape, dtype=bool)
        if len(global_remove_ids):
            remove_mask |= np.isin(labels, global_remove_ids)
        cluster_mask = np.zeros(labels.shape, dtype=bool)
        if len(plane_remove_ids):
            cluster_mask = np.isin(labels, plane_remove_ids)
            remove_mask |= cluster_mask

        removed_pixels = int(np.count_nonzero(remove_mask))
        cluster_pixels = int(np.count_nonzero(cluster_mask))
        total_removed_pixels += removed_pixels
        total_cluster_fragment_pixels += cluster_pixels
        cleaned = labels.copy()
        cleaned[remove_mask] = 0

        plane_number = natural_label_key(path)[0]
        for global_id in sorted(plane_reasons):
            fragment_rows.append(
                {
                    "plane_number": plane_number,
                    "filename": path.name,
                    "global_id": global_id,
                    "plane_area": int(counts[global_id]),
                    "plane_area_um2": (
                        int(counts[global_id]) * settings.pixel_area_um2
                    ),
                    "reasons": ";".join(sorted(plane_reasons[global_id])),
                }
            )
        plane_rows.append(
            {
                "plane_number": plane_number,
                "filename": path.name,
                "objects_before": int(np.count_nonzero(counts[1:])),
                "objects_after": int(len(np.unique(cleaned)) - 1),
                "global_unreliable_ids_present": int(
                    len(np.intersect1d(np.flatnonzero(counts[1:] > 0) + 1, global_remove_ids))
                ),
                "plane_fragment_ids": int(len(plane_reasons)),
                "pixels_removed": removed_pixels,
                "foreground_pixels_before": int(np.count_nonzero(labels)),
                "foreground_pixels_after": int(np.count_nonzero(cleaned)),
            }
        )

        if not dry_run:
            write_label_plane(output_dir / path.name, cleaned, settings.compression)
        print(
            f"Pass 2/2: {position + 1}/{len(files)} {path.name} | "
            f"removed {removed_pixels:,} pixels, "
            f"plane-fragment IDs={len(plane_reasons)}",
            flush=True,
        )

    summary = {
        "status": "dry_run" if dry_run else "completed",
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "output_dir": None if dry_run else str(output_dir),
        "n_planes": len(files),
        "shape_yx": list(shape),
        "settings": asdict(settings),
        "global_objects_before": len(stats),
        "global_objects_removed": len(global_reasons),
        "plane_fragment_records": len(fragment_rows),
        "total_pixels_removed": total_removed_pixels,
        "cluster_fragment_pixels_before_global_overlap_adjustment": (
            total_cluster_fragment_pixels
        ),
    }

    if not dry_run:
        write_csv(
            output_dir / "removed_global_objects.csv",
            global_rows,
            (
                "global_id",
                "n_z_planes",
                "sampled_z_thickness_um",
                "max_plane_area",
                "max_plane_area_um2",
                "total_voxels",
                "total_volume_um3",
                "reasons",
            ),
        )
        write_csv(
            output_dir / "removed_plane_fragments.csv",
            fragment_rows,
            (
                "plane_number",
                "filename",
                "global_id",
                "plane_area",
                "plane_area_um2",
                "reasons",
            ),
        )
        write_csv(
            output_dir / "cleaning_by_plane.csv",
            plane_rows,
            (
                "plane_number",
                "filename",
                "objects_before",
                "objects_after",
                "global_unreliable_ids_present",
                "plane_fragment_ids",
                "pixels_removed",
                "foreground_pixels_before",
                "foreground_pixels_after",
            ),
        )
        atomic_write_json(output_dir / "cleaning_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def output_is_completed(output_dir: Path) -> bool:
    summary_path = output_dir / "cleaning_summary.json"
    if not summary_path.is_file():
        return False
    try:
        with summary_path.open(encoding="utf-8") as stream:
            summary = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return False
    return summary.get("status") == "completed"


def discover_configured_label_dirs(
    label_dirname: str,
) -> tuple[list[tuple[str, Path]], list[dict[str, object]]]:
    discovered: list[tuple[str, Path]] = []
    root_records: list[dict[str, object]] = []

    for experiment_id, root in EXPERIMENT_ROOTS.items():
        experiment_dir = root / "Experiment"
        if not experiment_dir.is_dir():
            root_records.append(
                {
                    "experiment_id": experiment_id,
                    "experiment_dir": str(experiment_dir),
                    "status": "root_not_found",
                    "label_directories": 0,
                }
            )
            continue

        matches = sorted(
            (
                path.resolve()
                for path in experiment_dir.rglob(label_dirname)
                if path.is_dir() and path.name == label_dirname
            ),
            key=lambda path: str(path),
        )
        root_records.append(
            {
                "experiment_id": experiment_id,
                "experiment_dir": str(experiment_dir),
                "status": "found" if matches else "no_label_directories",
                "label_directories": len(matches),
            }
        )
        discovered.extend((experiment_id, path) for path in matches)

    return discovered, root_records


def run_one_stack(
    args: argparse.Namespace,
    input_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    settings = build_settings(args, input_dir)
    files = discover_label_planes(input_dir, args.glob)
    return clean_stack(files, input_dir, output_dir, settings, args.dry_run)


def run_all_experiments(args: argparse.Namespace) -> int:
    label_dirs, root_records = discover_configured_label_dirs(args.label_dirname)
    sample_records: list[dict[str, object]] = []

    for experiment_id, input_dir in label_dirs:
        output_dir = input_dir.with_name(input_dir.name + "_cleaned")
        record: dict[str, object] = {
            "experiment_id": experiment_id,
            "input_dir": str(input_dir),
            "output_dir": None if args.dry_run else str(output_dir),
        }

        if not args.dry_run and output_is_completed(output_dir):
            record["status"] = "skipped_completed"
            sample_records.append(record)
            print(f"SKIP completed: {input_dir}", flush=True)
            continue

        if not args.dry_run and output_dir.exists():
            output_is_nonempty = not output_dir.is_dir() or any(output_dir.iterdir())
            if output_is_nonempty:
                record["status"] = "skipped_incomplete_output"
                record["error"] = (
                    "Output exists but has no completed cleaning_summary.json"
                )
                sample_records.append(record)
                print(f"SKIP incomplete/nonempty output: {output_dir}", flush=True)
                continue

        print(f"RUN: {input_dir}", flush=True)
        try:
            summary = run_one_stack(args, input_dir, output_dir)
        except Exception as exc:
            record["status"] = "error"
            record["error_type"] = type(exc).__name__
            record["error"] = str(exc)
            sample_records.append(record)
            print(
                f"ERROR: {input_dir} | {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            continue

        record["status"] = summary["status"]
        record["global_objects_removed"] = summary["global_objects_removed"]
        record["total_pixels_removed"] = summary["total_pixels_removed"]
        sample_records.append(record)

    status_counts: defaultdict[str, int] = defaultdict(int)
    for record in sample_records:
        status_counts[str(record["status"])] += 1

    root_problem = any(record["status"] != "found" for record in root_records)
    sample_problem = any(
        record["status"] in {"error", "skipped_incomplete_output"}
        for record in sample_records
    )
    if args.dry_run:
        batch_status = (
            "dry_run_with_problems" if root_problem or sample_problem else "dry_run"
        )
    else:
        batch_status = (
            "completed_with_problems" if root_problem or sample_problem else "completed"
        )

    batch_summary = {
        "status": batch_status,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "label_dirname": args.label_dirname,
        "configured_roots": root_records,
        "label_directories_discovered": len(label_dirs),
        "status_counts": dict(sorted(status_counts.items())),
        "samples": sample_records,
    }
    print("BATCH SUMMARY", flush=True)
    print(json.dumps(batch_summary, indent=2), flush=True)

    return 1 if root_problem or sample_problem else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input-dir", type=Path)
    mode.add_argument(
        "--all-experiments",
        action="store_true",
        help="Clean all final Cellpose label directories under the five roots.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to a sibling folder named <input-name>_cleaned.",
    )
    parser.add_argument("--glob", default="Nucleus_Labels_*.tif")
    parser.add_argument(
        "--label-dirname",
        default=DEFAULT_LABEL_DIRNAME,
        help="Directory name searched by --all-experiments.",
    )
    parser.add_argument(
        "--experiment-id",
        choices=tuple(EXPERIMENT_VOXEL_SIZE_XYZ_NM),
        default=None,
        help="Defaults to automatic detection from --input-dir.",
    )
    parser.add_argument(
        "--voxel-size-nm",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Custom voxel size in nm; overrides the experiment lookup table.",
    )
    parser.add_argument(
        "--min-max-plane-area-um2",
        type=float,
        default=DEFAULT_MIN_MAX_PLANE_AREA_UM2,
    )
    parser.add_argument(
        "--min-z-thickness-um", type=float, default=DEFAULT_MIN_Z_THICKNESS_UM
    )
    parser.add_argument(
        "--small-object-max-area-um2",
        type=float,
        default=DEFAULT_SMALL_OBJECT_MAX_AREA_UM2,
    )
    parser.add_argument(
        "--cluster-min-area-um2", type=float, default=DEFAULT_CLUSTER_MIN_AREA_UM2
    )
    parser.add_argument(
        "--border-max-area-um2", type=float, default=DEFAULT_BORDER_MAX_AREA_UM2
    )
    parser.add_argument(
        "--min-max-plane-area",
        type=int,
        default=None,
        help="Optional pixel override for --min-max-plane-area-um2.",
    )
    parser.add_argument(
        "--min-z-planes",
        type=int,
        default=None,
        help="Optional plane-count override for --min-z-thickness-um.",
    )
    parser.add_argument(
        "--small-object-max-area",
        type=int,
        default=None,
        help="Optional pixel override for --small-object-max-area-um2.",
    )
    parser.add_argument("--cluster-min-labels", type=int, default=4)
    parser.add_argument(
        "--cluster-min-pixels",
        type=int,
        default=None,
        help="Optional pixel override for --cluster-min-area-um2.",
    )
    parser.add_argument("--remove-small-border-slivers", action="store_true")
    parser.add_argument(
        "--border-max-area",
        type=int,
        default=None,
        help="Optional pixel override for --border-max-area-um2.",
    )
    parser.add_argument("--border-min-aspect-ratio", type=float, default=3.0)
    parser.add_argument("--compression", choices=("none", "zlib"), default="zlib")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.all_experiments and args.output_dir is not None:
        raise SystemExit("--output-dir can only be used with --input-dir")
    if args.all_experiments and args.experiment_id is not None:
        raise SystemExit("--experiment-id can only be used with --input-dir")
    if args.all_experiments and args.voxel_size_nm is not None:
        raise SystemExit("--voxel-size-nm can only be used with --input-dir")
    if not args.label_dirname:
        raise SystemExit("--label-dirname must not be empty")

    integer_nonnegative = (
        (args.min_max_plane_area, "--min-max-plane-area"),
        (args.small_object_max_area, "--small-object-max-area"),
        (args.cluster_min_pixels, "--cluster-min-pixels"),
        (args.border_max_area, "--border-max-area"),
    )
    for value, name in integer_nonnegative:
        if value is not None and value < 0:
            raise SystemExit(f"{name} must be >=0")
    physical_nonnegative = (
        (args.min_max_plane_area_um2, "--min-max-plane-area-um2"),
        (args.min_z_thickness_um, "--min-z-thickness-um"),
        (args.small_object_max_area_um2, "--small-object-max-area-um2"),
        (args.cluster_min_area_um2, "--cluster-min-area-um2"),
        (args.border_max_area_um2, "--border-max-area-um2"),
    )
    for value, name in physical_nonnegative:
        if value < 0:
            raise SystemExit(f"{name} must be >=0")
    if args.voxel_size_nm is not None and any(
        value <= 0 for value in args.voxel_size_nm
    ):
        raise SystemExit("Every --voxel-size-nm value must be >0")
    if args.min_z_planes is not None and args.min_z_planes < 1:
        raise SystemExit("--min-z-planes must be >=1")
    if args.cluster_min_labels < 2:
        raise SystemExit("--cluster-min-labels must be >=2")
    if args.border_min_aspect_ratio < 1:
        raise SystemExit("--border-min-aspect-ratio must be >=1")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args)
    if args.all_experiments:
        return run_all_experiments(args)

    if args.input_dir is None:  # enforced by argparse; retained for type safety
        raise SystemExit("--input-dir is required outside batch mode")
    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else input_dir.with_name(input_dir.name + "_cleaned")
    )
    run_one_stack(args, input_dir, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
