#!/usr/bin/env python3
"""Stream Cellpose-SAM 2-D nuclear segmentation and conservative 3-D tracking.

The script is designed for the planarian smFISH directory layout used by
experiments 304, 306, 307, 314, and 332::

    <experiment root>/Experiment/<condition>/<animal>/405/<animal>_405.tif
    <experiment root>/Experiment/<condition>/<animal>/565/results/

Only samples below an ``Experiment`` directory are discovered; sibling
``Control`` trees are deliberately ignored.  For every sample, globally
tracked label planes and audit tables are written to::

    <animal>/565/results/<output-name>/

The input TIFF is memory-mapped and Cellpose-SAM processes one z-plane at a
time.  This avoids loading a multi-gigabyte stack, Cellpose intermediate
arrays, and the complete label volume into RAM simultaneously.

Adjacent 2-D masks are stitched with Cellpose's own
``cellpose.utils.stitch3D`` implementation.  Per-plane label TIFFs contain
global 3-D IDs, so the same nucleus has the same integer in consecutive
z-planes.  A disk-backed uint32 mask stack keeps the original 2-D inference
streaming and avoids holding the multi-gigabyte image and all network flows in
memory at once.

The default settings are the benchmark-selected production parameters for all
five experiments.  Recommended production run on the acquisition Mac::

    python cellpose_sam_nuclei_3d.py --self-test

    caffeinate -ims python cellpose_sam_nuclei_3d.py \
      --device mps --resume --verbose

After visually checking several planes and the tracking QC, broaden the
``--sample-regex`` while keeping the same parameters and output name.

Required packages: cellpose>=4.2, torch, numpy, tifffile.
"""

from __future__ import annotations

import argparse
import csv
import gc
import inspect
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

try:
    from tifffile import TiffFile, imread, imwrite, memmap
except ImportError:  # permit dependency-free tracking self-test
    TiffFile = None
    imread = None
    imwrite = None
    memmap = None


SCRIPT_VERSION = "1.3.0"
LOGGER = logging.getLogger("cellpose_sam_nuclei_3d")


DEFAULT_EXPERIMENTS = (
    (
        "304",
        "wnt1",
        "/Volumes/Backup Plus/Experiment_results/304_Analysis_results",
    ),
    (
        "306",
        "wnt1",
        "/Volumes/Backup Plus/Experiment_results/306_analysis_results_alter",
    ),
    (
        "307",
        "wnt1",
        "/Volumes/Backup Plus/Experiment_results/307_analysis_resutls/307_quantificatoin",
    ),
    (
        "314",
        "notum",
        "/Volumes/Backup Plus/Experiment_results/314_analysis_results",
    ),
    (
        "332",
        "notum",
        "/Volumes/Backup Plus/Experiment_results/332_notum_analysis",
    ),
)


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    gene: str
    root: Path


@dataclass(frozen=True)
class SampleSpec:
    experiment: str
    gene: str
    condition: str
    animal: str
    sample_dir: Path
    results_dir: Path
    nuclear_image: Path

    @property
    def sample_id(self) -> str:
        return f"{self.experiment}__{self.condition}__{self.animal}"


@dataclass(frozen=True)
class Parameters:
    model: str
    device: str
    batch_size: int
    diameter: float | None
    niter: int
    flow_threshold: float
    cellprob_threshold: float
    min_size: int
    tile_overlap: float
    normalization: str
    normalization_planes: int
    smooth_radius: float
    stitch_threshold: float
    compression: str


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def natural_key(path_or_name: str | Path) -> list[object]:
    name = Path(path_or_name).name
    return [
        int(piece) if piece.isdigit() else piece.lower()
        for piece in re.split(r"(\d+)", name)
    ]


def parse_experiment_argument(value: str) -> ExperimentSpec:
    parts = value.split(",", 2)
    if len(parts) != 3 or not all(part.strip() for part in parts):
        raise argparse.ArgumentTypeError(
            "--experiment must be formatted as NAME,GENE,ROOT"
        )
    name, gene, root = (part.strip() for part in parts)
    return ExperimentSpec(name=name, gene=gene, root=Path(root))


def locate_nuclear_image(sample_dir: Path, nuclear_channel: str) -> Path:
    channel_dir = sample_dir / nuclear_channel
    if not channel_dir.is_dir():
        raise FileNotFoundError(f"Missing nuclear channel folder: {channel_dir}")

    channel_token = nuclear_channel.lower()
    candidates = [
        path.resolve()
        for path in channel_dir.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in {".tif", ".tiff"}
        and channel_token in path.name.lower()
    ]
    unique = sorted(set(candidates), key=natural_key)
    if len(unique) != 1:
        raise FileNotFoundError(
            f"Expected exactly one TIFF whose filename contains {nuclear_channel!r} "
            f"in {channel_dir}; found {len(unique)}: "
            + ", ".join(str(path.name) for path in unique)
        )
    return unique[0]


def discover_samples(
    spec: ExperimentSpec,
    spot_channel: str,
    nuclear_channel: str,
) -> tuple[list[SampleSpec], list[dict[str, str]]]:
    """Discover only Experiment/<condition>/<animal>/<spot>/results paths."""

    if not spec.root.exists():
        LOGGER.warning("Experiment root does not exist; skipping: %s", spec.root)
        return [], []

    samples: list[SampleSpec] = []
    errors: list[dict[str, str]] = []
    for results_dir in spec.root.rglob("results"):
        if not results_dir.is_dir() or results_dir.parent.name != spot_channel:
            continue
        sample_dir = results_dir.parent.parent
        condition_dir = sample_dir.parent
        experiment_dir = condition_dir.parent
        if experiment_dir.name.lower() != "experiment":
            LOGGER.debug("Ignoring non-Experiment results path: %s", results_dir)
            continue
        try:
            nuclear_image = locate_nuclear_image(sample_dir, nuclear_channel)
        except Exception as exc:
            errors.append(
                {
                    "sample_id": f"{spec.name}__{condition_dir.name}__{sample_dir.name}",
                    "sample_path": str(sample_dir),
                    "error": str(exc),
                }
            )
            continue
        samples.append(
            SampleSpec(
                experiment=spec.name,
                gene=spec.gene,
                condition=condition_dir.name,
                animal=sample_dir.name,
                sample_dir=sample_dir,
                results_dir=results_dir,
                nuclear_image=nuclear_image,
            )
        )

    unique = {sample.results_dir.resolve(): sample for sample in samples}
    ordered = sorted(
        unique.values(),
        key=lambda sample: (
            natural_key(sample.condition),
            natural_key(sample.animal),
            str(sample.results_dir),
        ),
    )
    return ordered, errors


def inspect_tiff(path: Path) -> tuple[tuple[int, ...], str]:
    if TiffFile is None:
        raise SystemExit("Install tifffile: python -m pip install tifffile")
    with TiffFile(path) as tif:
        series = tif.series[0]
        return tuple(int(value) for value in series.shape), str(series.axes).upper()


def normalize_axes(array: np.ndarray, axes: str) -> np.ndarray:
    """Return a ZYX view and reject ambiguous multi-channel/time inputs."""

    if len(axes) != array.ndim:
        if array.ndim == 2:
            axes = "YX"
        elif array.ndim == 3:
            axes = "ZYX"
        else:
            raise ValueError(
                f"TIFF axes {axes!r} do not match array shape {array.shape}"
            )

    # Generic multi-page TIFF stacks are commonly reported by tifffile as
    # QYX or IYX rather than ZYX.  In this nuclei-only pipeline, a single
    # leading page axis is the acquisition z-axis.
    if array.ndim == 3 and axes.endswith("YX") and "Z" not in axes:
        LOGGER.warning("Interpreting generic TIFF page axis %s as Z", axes[0])
        axes = "ZYX"

    squeeze_axes = [
        index
        for index, (axis, size) in enumerate(zip(axes, array.shape, strict=True))
        if axis not in "ZYX" and size == 1
    ]
    if squeeze_axes:
        array = np.squeeze(array, axis=tuple(squeeze_axes))
        axes = "".join(axis for index, axis in enumerate(axes) if index not in squeeze_axes)

    unsupported = [
        (axis, int(size))
        for axis, size in zip(axes, array.shape, strict=True)
        if axis not in "ZYX"
    ]
    if unsupported:
        raise ValueError(
            "Nuclear TIFF must contain one Z stack and no non-singleton time/channel "
            f"axes; got axes={axes}, shape={array.shape}, unsupported={unsupported}"
        )

    if "Y" not in axes or "X" not in axes:
        if array.ndim == 2:
            axes = "YX"
        elif array.ndim == 3:
            axes = "ZYX"
        else:
            raise ValueError(f"Cannot identify Y/X axes: axes={axes}, shape={array.shape}")

    if "Z" not in axes:
        array = array[np.newaxis, ...]
        axes = "Z" + axes
    if array.ndim != 3:
        raise ValueError(f"Expected a 3-D ZYX view; got axes={axes}, shape={array.shape}")

    source = [axes.index(axis) for axis in "ZYX"]
    return np.moveaxis(array, source, [0, 1, 2])


@contextmanager
def open_zyx_volume(path: Path) -> Iterator[np.ndarray]:
    """Open a TIFF as a ZYX memory map; compressed input uses a disk temp map."""

    if memmap is None or imread is None:
        raise SystemExit("Install tifffile: python -m pip install tifffile")
    shape, axes = inspect_tiff(path)
    LOGGER.info("Input TIFF %s | axes=%s shape=%s", path, axes, shape)
    temporary_path: Path | None = None
    try:
        try:
            array = memmap(path, series=0, mode="r")
            LOGGER.info("Using direct TIFF memory map (input is not loaded into RAM)")
        except Exception as exc:
            LOGGER.warning(
                "Direct memory map unavailable (%s). Decompressing once to a temporary "
                "disk-backed array; free disk roughly equal to the TIFF size is required.",
                exc,
            )
            handle, name = tempfile.mkstemp(prefix="cellpose_input_", suffix=".memmap")
            os.close(handle)
            temporary_path = Path(name)
            array = imread(path, series=0, out=str(temporary_path))
        yield normalize_axes(np.asarray(array), axes)
    finally:
        try:
            del array
        except UnboundLocalError:
            pass
        gc.collect()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def estimate_global_lowhigh(
    volume: np.ndarray,
    n_planes: int,
    max_values_per_plane: int = 200_000,
) -> tuple[float, float]:
    z_count = int(volume.shape[0])
    plane_indices = np.unique(
        np.linspace(0, z_count - 1, min(z_count, n_planes), dtype=int)
    )
    samples: list[np.ndarray] = []
    for z_index in plane_indices:
        plane = np.asarray(volume[int(z_index)])
        stride = max(1, math.ceil(plane.size / max_values_per_plane))
        samples.append(np.ravel(plane)[::stride].astype(np.float32, copy=False))
    values = np.concatenate(samples)
    low, high = np.percentile(values, [1.0, 99.0])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError(
            f"Could not estimate useful 1/99 percentiles: low={low}, high={high}"
        )
    return float(low), float(high)


def choose_device(requested: str):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Cellpose requires PyTorch; install cellpose>=4.2") from exc

    if requested == "auto":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            requested = "mps"
        elif torch.cuda.is_available():
            requested = "cuda"
        else:
            requested = "cpu"

    if requested == "mps":
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS was requested but is unavailable. Use native arm64 Python/PyTorch "
                "on Apple Silicon, or run with --device cpu."
            )
        return torch.device("mps"), True, "mps"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return torch.device("cuda"), True, "cuda"
    return torch.device("cpu"), False, "cpu"


def create_cellpose_model(model_name: str, requested_device: str):
    device, use_gpu, device_name = choose_device(requested_device)
    try:
        from cellpose import models
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Install Cellpose-SAM first: python -m pip install --upgrade 'cellpose>=4.2'"
        ) from exc

    kwargs: dict[str, object] = {
        "gpu": use_gpu,
        "device": device,
        "pretrained_model": model_name,
    }
    # MPS has historically been more stable in float32 than bfloat16.  Current
    # Cellpose accepts this argument; the fallback keeps compatibility with
    # early 4.x releases while still recommending >=4.2 to the user.
    if device_name == "mps":
        kwargs["use_bfloat16"] = False
    try:
        model = models.CellposeModel(**kwargs)
    except TypeError:
        kwargs.pop("use_bfloat16", None)
        model = models.CellposeModel(**kwargs)
    LOGGER.info("Loaded Cellpose model %s on %s", model_name, device_name)
    return model, device_name


@lru_cache(maxsize=1)
def normalization_smooth_parameter() -> str | None:
    """Return the smoothing keyword accepted by this Cellpose installation."""

    try:
        from cellpose import transforms
    except ImportError as exc:  # pragma: no cover - exercised on the user's Mac
        raise SystemExit(
            "Install Cellpose-SAM first: python -m pip install --upgrade cellpose"
        ) from exc
    try:
        accepted = inspect.signature(transforms.normalize_img).parameters
    except (AttributeError, TypeError, ValueError):
        return None
    if "smooth_radius" in accepted:
        return "smooth_radius"
    if "smooth" in accepted:
        return "smooth"
    return None


def segment_plane(
    model,
    plane: np.ndarray,
    parameters: Parameters,
    normalization_values: tuple[float, float] | None,
) -> np.ndarray:
    normalize: bool | dict[str, object]
    if parameters.normalization == "global":
        if normalization_values is None:
            raise ValueError("Global normalization values were not calculated")
        normalize = {
            "normalize": True,
            "lowhigh": [normalization_values[0], normalization_values[1]],
        }
    elif parameters.normalization == "plane":
        # This matches the benchmark's norm-block=0 setting: each plane gets
        # its own 1/99-percentile normalization without local tile correction.
        normalize = {
            "normalize": True,
            "percentile": [1.0, 99.0],
            "tile_norm_blocksize": 0,
        }
    else:
        normalize = False

    if parameters.smooth_radius > 0:
        if not isinstance(normalize, dict):
            raise ValueError("--smooth-radius requires normalization to be enabled")
        smooth_parameter = normalization_smooth_parameter()
        if smooth_parameter is None:
            raise RuntimeError(
                "This Cellpose version does not expose smooth_radius/smooth in "
                "transforms.normalize_img; install the same version used by the "
                "working benchmark."
            )
        normalize[smooth_parameter] = parameters.smooth_radius

    # Cellpose 4.x releases expose slightly different model.eval keywords.
    # Filter optional compatibility settings by the installed signature while
    # requiring every benchmark-selected setting to be supported explicitly.
    candidates: dict[str, object] = {
        "batch_size": parameters.batch_size,
        "channels": None,
        "channel_axis": None,
        "z_axis": None,
        "normalize": normalize,
        "diameter": parameters.diameter,
        "niter": parameters.niter,
        "resample": True,
        "interp": True,
        "flow_threshold": parameters.flow_threshold,
        "cellprob_threshold": parameters.cellprob_threshold,
        "do_3D": False,
        "min_size": parameters.min_size,
        "augment": False,
        "tile_overlap": parameters.tile_overlap,
        "bsize": 256,
        "compute_masks": True,
    }
    eval_signature = inspect.signature(model.eval)
    accepts_extra_keywords = any(
        item.kind == inspect.Parameter.VAR_KEYWORD
        for item in eval_signature.parameters.values()
    )
    if accepts_extra_keywords:
        eval_kwargs = candidates
    else:
        eval_kwargs = {
            key: value
            for key, value in candidates.items()
            if key in eval_signature.parameters
        }

    required_keywords = {
        "batch_size",
        "normalize",
        "diameter",
        "niter",
        "flow_threshold",
        "cellprob_threshold",
        "min_size",
        "tile_overlap",
    }
    missing = sorted(required_keywords - set(eval_kwargs))
    if missing:
        raise RuntimeError(
            "This Cellpose model.eval cannot apply the requested benchmark "
            f"parameters because it lacks: {', '.join(missing)}"
        )
    skipped = sorted(set(candidates) - set(eval_kwargs))
    if skipped:
        LOGGER.debug(
            "Cellpose model.eval does not expose optional keywords; skipping: %s",
            ", ".join(skipped),
        )

    result = model.eval(np.asarray(plane), **eval_kwargs)
    masks = result[0] if isinstance(result, tuple) else result
    if isinstance(masks, list):
        if len(masks) != 1:
            raise ValueError(f"Expected one mask plane, Cellpose returned {len(masks)}")
        masks = masks[0]
    masks = np.asarray(masks)
    if masks.ndim == 3 and masks.shape[0] == 1:
        masks = masks[0]
    if masks.shape != plane.shape:
        raise ValueError(
            f"Cellpose mask shape {masks.shape} differs from input plane {plane.shape}"
        )
    if np.any(masks < 0):
        raise ValueError("Cellpose returned negative label values")
    return masks.astype(np.uint32, copy=False)


def atomic_write_json(data: object, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[dict[str, object]], fieldnames: Sequence[str]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


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


def clear_device_cache(device_name: str) -> None:
    try:
        import torch

        if device_name == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        elif device_name == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        LOGGER.debug("Could not clear accelerator cache", exc_info=True)


def _cellpose_stitch3d(mask_stack: np.ndarray, stitch_threshold: float) -> np.ndarray:
    """Call the public Cellpose stitching function on a uint32 disk array."""

    try:
        from cellpose import utils
    except ImportError as exc:  # pragma: no cover - exercised on the user's Mac
        raise SystemExit(
            "Install Cellpose-SAM first: python -m pip install --upgrade 'cellpose>=4.2'"
        ) from exc
    LOGGER.info(
        "Running official cellpose.utils.stitch3D with stitch_threshold=%.3f",
        stitch_threshold,
    )
    try:
        stitched = utils.stitch3D(mask_stack, stitch_threshold=stitch_threshold)
    except MemoryError as exc:
        raise MemoryError(
            "Cellpose stitch3D ran out of memory while constructing its adjacent-plane "
            "IoU table. Close other applications or test a higher stitch threshold; "
            "if this persists, the sparse tracker is required for this unusually dense image."
        ) from exc
    return stitched


def _plane_label_stats(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids, counts = np.unique(labels, return_counts=True)
    foreground = ids > 0
    return ids[foreground].astype(np.uint32, copy=False), counts[foreground]


def _write_cellpose_audit_tables(
    stitched: np.ndarray,
    output_dir: Path,
) -> dict[str, object]:
    """Write per-plane objects, adjacent stitched links, and 3-D summaries."""

    plane_path = output_dir / "plane_objects.csv"
    link_path = output_dir / "tracking_links.csv"
    plane_temporary = plane_path.with_name(plane_path.name + ".tmp")
    link_temporary = link_path.with_name(link_path.name + ".tmp")

    aggregates: dict[int, dict[str, int]] = {}
    total_plane_objects = 0
    stitch_links = 0
    previous_labels: np.ndarray | None = None

    with plane_temporary.open("w", newline="", encoding="utf-8") as plane_stream, \
            link_temporary.open("w", newline="", encoding="utf-8") as link_stream:
        plane_writer = csv.DictWriter(
            plane_stream,
            fieldnames=("z", "global_cell_id", "area_pixels"),
        )
        link_writer = csv.DictWriter(
            link_stream,
            fieldnames=(
                "z_previous",
                "z_current",
                "global_cell_id",
                "intersection_pixels",
                "iou",
            ),
        )
        plane_writer.writeheader()
        link_writer.writeheader()

        for z_index in range(int(stitched.shape[0])):
            labels = np.asarray(stitched[z_index])
            ids, counts = _plane_label_stats(labels)
            total_plane_objects += len(ids)
            for global_id_raw, area_raw in zip(ids, counts, strict=True):
                global_id = int(global_id_raw)
                area = int(area_raw)
                plane_writer.writerow(
                    {"z": z_index, "global_cell_id": global_id, "area_pixels": area}
                )
                record = aggregates.setdefault(
                    global_id,
                    {
                        "global_cell_id": global_id,
                        "first_z": z_index,
                        "last_z": z_index,
                        "n_z_planes": 0,
                        "voxel_count": 0,
                        "max_cross_section_area_pixels": 0,
                    },
                )
                record["first_z"] = min(record["first_z"], z_index)
                record["last_z"] = max(record["last_z"], z_index)
                record["n_z_planes"] += 1
                record["voxel_count"] += area
                record["max_cross_section_area_pixels"] = max(
                    record["max_cross_section_area_pixels"], area
                )

            if previous_labels is not None:
                same = (labels > 0) & (labels == previous_labels)
                if np.any(same):
                    shared_ids, intersections = np.unique(
                        labels[same], return_counts=True
                    )
                    current_area = dict(zip(ids.astype(int), counts.astype(int), strict=True))
                    previous_ids, previous_counts = _plane_label_stats(previous_labels)
                    previous_area = dict(
                        zip(
                            previous_ids.astype(int),
                            previous_counts.astype(int),
                            strict=True,
                        )
                    )
                    for global_id_raw, intersection_raw in zip(
                        shared_ids, intersections, strict=True
                    ):
                        global_id = int(global_id_raw)
                        intersection = int(intersection_raw)
                        union = (
                            current_area[global_id]
                            + previous_area[global_id]
                            - intersection
                        )
                        link_writer.writerow(
                            {
                                "z_previous": z_index - 1,
                                "z_current": z_index,
                                "global_cell_id": global_id,
                                "intersection_pixels": intersection,
                                "iou": intersection / union,
                            }
                        )
                        stitch_links += 1
            previous_labels = labels.copy()

    plane_temporary.replace(plane_path)
    link_temporary.replace(link_path)

    object_fields = (
        "global_cell_id",
        "first_z",
        "last_z",
        "n_z_planes",
        "voxel_count",
        "max_cross_section_area_pixels",
    )
    object_rows = [aggregates[key] for key in sorted(aggregates)]
    write_csv(output_dir / "objects_3d.csv", object_rows, object_fields)

    distribution: dict[int, int] = {}
    for row in object_rows:
        n_planes = int(row["n_z_planes"])
        distribution[n_planes] = distribution.get(n_planes, 0) + 1
    ordered_lengths = np.asarray(
        [int(row["n_z_planes"]) for row in object_rows], dtype=np.int64
    )
    return {
        "plane_objects": total_plane_objects,
        "global_3d_objects": len(object_rows),
        "single_plane_objects": distribution.get(1, 0),
        "stitch_links": stitch_links,
        "median_z_planes_per_object": float(np.median(ordered_lengths))
        if len(ordered_lengths)
        else 0.0,
        "p95_z_planes_per_object": float(np.percentile(ordered_lengths, 95))
        if len(ordered_lengths)
        else 0.0,
        "maximum_z_planes_per_object": int(ordered_lengths.max())
        if len(ordered_lengths)
        else 0,
        "n_z_planes_distribution": {
            str(key): distribution[key] for key in sorted(distribution)
        },
    }


def _existing_written_planes(output_dir: Path) -> int:
    files = sorted(output_dir.glob("Nucleus_Labels_*.tif"), key=natural_key)
    pattern = re.compile(r"Nucleus_Labels_(\d+)\.tif$")
    indices = [
        int(match.group(1))
        for path in files
        if (match := pattern.search(path.name)) is not None
    ]
    if indices and indices != list(range(len(indices))):
        raise ValueError(
            f"Output label planes are not contiguous from z=0: {indices[:20]}"
        )
    return len(indices)


def remove_output_tree(output_dir: Path) -> None:
    """Remove an old output tree while tolerating disappearing AppleDouble files.

    On macOS external volumes, deleting a data file can also remove its ``._``
    AppleDouble companion.  ``shutil.rmtree`` may already have listed that
    companion and then receive ``FileNotFoundError`` when it tries to unlink it.
    That race is harmless and can be ignored.  Every other deletion error is
    re-raised, and the final directory removal is verified so stale masks can
    never be reused by an overwrite run.
    """

    def ignore_disappeared_file(function, path, exc_info) -> None:
        error = exc_info[1]
        if isinstance(error, FileNotFoundError):
            LOGGER.debug(
                "Ignoring file that disappeared during output cleanup: %s", path
            )
            return
        raise error

    try:
        shutil.rmtree(output_dir, onerror=ignore_disappeared_file)
    except FileNotFoundError:
        # The output directory itself may disappear between exists() and rmtree().
        pass

    if output_dir.exists():
        raise OSError(f"Could not fully remove prior output directory: {output_dir}")


def process_sample(
    sample: SampleSpec,
    parameters: Parameters,
    model,
    device_name: str,
    output_name: str,
    overwrite: bool,
    resume: bool,
) -> dict[str, object]:
    output_dir = sample.results_dir / output_name
    complete_path = output_dir / "_COMPLETE.json"
    config_path = output_dir / "run_configuration.json"
    progress_path = output_dir / "_PROGRESS.json"
    local_store_path = output_dir / ".local_2d_masks.uint32.memmap"
    stitched_store_path = output_dir / ".cellpose_stitched.uint32.memmap"

    if complete_path.exists() and not overwrite:
        if not config_path.exists():
            raise FileNotFoundError(
                f"Completed output lacks {config_path}; use --overwrite"
            )
        with config_path.open(encoding="utf-8") as stream:
            completed_config = json.load(stream)
        if completed_config.get("parameters") != asdict(parameters):
            raise ValueError(
                f"Completed output for {sample.sample_id} used different parameters. "
                "Use --overwrite to regenerate it."
            )
        LOGGER.info("Skipping completed sample %s", sample.sample_id)
        with complete_path.open(encoding="utf-8") as stream:
            return json.load(stream)

    if output_dir.exists() and overwrite:
        LOGGER.warning("Overwriting prior output: %s", output_dir)
        remove_output_tree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    partial_exists = any(output_dir.iterdir())
    if partial_exists and not resume and not overwrite:
        raise FileExistsError(
            f"Partial output exists in {output_dir}. Use --resume or --overwrite."
        )

    with open_zyx_volume(sample.nuclear_image) as volume:
        z_count, height, width = (int(value) for value in volume.shape)
        store_shape = (z_count, height, width)
        configuration = {
            "script_version": SCRIPT_VERSION,
            "tracking_backend": "cellpose.utils.stitch3D",
            "sample_id": sample.sample_id,
            "experiment": sample.experiment,
            "gene": sample.gene,
            "condition": sample.condition,
            "animal": sample.animal,
            "sample_path": str(sample.sample_dir),
            "nuclear_image": str(sample.nuclear_image),
            "output_name": output_name,
            "output_dir": str(output_dir),
            "input_shape_zyx": [z_count, height, width],
            "input_dtype": str(volume.dtype),
            "parameters": asdict(parameters),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        if config_path.exists():
            with config_path.open(encoding="utf-8") as stream:
                old_config = json.load(stream)
            compare = ("sample_id", "nuclear_image", "input_shape_zyx", "parameters")
            changed = [key for key in compare if old_config.get(key) != configuration.get(key)]
            if changed:
                raise ValueError(
                    f"Resume configuration differs for {changed}; use --overwrite."
                )
        else:
            atomic_write_json(configuration, config_path)

        progress: dict[str, object] = {
            "completed_segmentation_planes": 0,
            "segmentation_complete": False,
            "stitch_complete": False,
            "written_label_planes": 0,
        }
        if progress_path.exists():
            with progress_path.open(encoding="utf-8") as stream:
                progress.update(json.load(stream))

        completed_segmentation = int(progress["completed_segmentation_planes"])
        if completed_segmentation and not local_store_path.exists():
            raise FileNotFoundError(
                f"Segmentation progress exists but {local_store_path} is missing; "
                "use --overwrite."
            )
        local_mode = "r+" if local_store_path.exists() else "w+"
        local_masks = np.memmap(
            local_store_path,
            dtype=np.uint32,
            mode=local_mode,
            shape=store_shape,
        )

        if parameters.normalization == "global":
            lowhigh = estimate_global_lowhigh(volume, parameters.normalization_planes)
            LOGGER.info("Global nuclear intensity 1/99 percentiles: %.3f / %.3f", *lowhigh)
        else:
            lowhigh = None

        for z_index in range(completed_segmentation, z_count):
            LOGGER.info(
                "%s | Cellpose-SAM 2-D z=%d/%d",
                sample.sample_id,
                z_index + 1,
                z_count,
            )
            plane = np.asarray(volume[z_index])
            masks = segment_plane(model, plane, parameters, lowhigh)
            local_masks[z_index] = masks
            local_masks.flush()
            progress["completed_segmentation_planes"] = z_index + 1
            progress["segmentation_complete"] = z_index + 1 == z_count
            atomic_write_json(progress, progress_path)
            del plane, masks
            gc.collect()
            clear_device_cache(device_name)

        if not bool(progress.get("stitch_complete")):
            stitched_masks = np.memmap(
                stitched_store_path,
                dtype=np.uint32,
                mode="w+",
                shape=store_shape,
            )
            for z_index in range(z_count):
                stitched_masks[z_index] = local_masks[z_index]
            stitched_masks.flush()
            result = _cellpose_stitch3d(
                stitched_masks,
                parameters.stitch_threshold,
            )
            if result is not stitched_masks:
                for z_index in range(z_count):
                    stitched_masks[z_index] = np.asarray(result[z_index], dtype=np.uint32)
            stitched_masks.flush()
            progress["stitch_complete"] = True
            progress["written_label_planes"] = 0
            atomic_write_json(progress, progress_path)
        else:
            if not stitched_store_path.exists():
                raise FileNotFoundError(
                    f"Stitching marked complete but {stitched_store_path} is missing; "
                    "use --overwrite."
                )
            stitched_masks = np.memmap(
                stitched_store_path,
                dtype=np.uint32,
                mode="r+",
                shape=store_shape,
            )

        written = _existing_written_planes(output_dir)
        progress["written_label_planes"] = written
        for z_index in range(written, z_count):
            write_label_plane(
                output_dir / f"Nucleus_Labels_{z_index:04d}.tif",
                stitched_masks[z_index],
                parameters.compression,
            )
            progress["written_label_planes"] = z_index + 1
            atomic_write_json(progress, progress_path)

        audit = _write_cellpose_audit_tables(stitched_masks, output_dir)
        summary: dict[str, object] = {
            **audit,
            "status": "completed",
            "script_version": SCRIPT_VERSION,
            "tracking_backend": "cellpose.utils.stitch3D",
            "stitch_threshold": parameters.stitch_threshold,
            "parameters": asdict(parameters),
            "sample_id": sample.sample_id,
            "experiment": sample.experiment,
            "gene": sample.gene,
            "condition": sample.condition,
            "animal": sample.animal,
            "sample_path": str(sample.sample_dir),
            "nuclear_image": str(sample.nuclear_image),
            "output_name": output_name,
            "output_dir": str(output_dir),
            "z_planes": z_count,
            "height": height,
            "width": width,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(summary, output_dir / "sample_qc.json")
        atomic_write_json(summary, complete_path)

        del local_masks, stitched_masks
        gc.collect()
        local_store_path.unlink(missing_ok=True)
        stitched_store_path.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)
        return summary


def write_batch_csv(rows: Sequence[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    preferred = [
        "status",
        "experiment",
        "gene",
        "condition",
        "animal",
        "sample_id",
        "z_planes",
        "height",
        "width",
        "plane_objects",
        "global_3d_objects",
        "single_plane_objects",
        "stitch_links",
        "nuclear_image",
        "output_name",
        "output_dir",
        "error",
    ]
    present = {key for row in rows for key in row}
    fields = [key for key in preferred if key in present]
    fields.extend(sorted(present - set(fields) - {"n_z_planes_distribution"}))
    serializable: list[dict[str, object]] = []
    for row in rows:
        serializable.append(
            {
                key: json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list))
                else value
                for key, value in row.items()
                if key in fields
            }
        )
    write_csv(path, serializable, fields)


def run_pipeline(
    experiments: Sequence[ExperimentSpec],
    parameters: Parameters,
    spot_channel: str,
    nuclear_channel: str,
    sample_regex: str | None,
    output_name: str,
    overwrite: bool,
    resume: bool,
    fail_fast: bool,
    dry_run: bool,
    summary_csv: Path,
) -> int:
    samples: list[SampleSpec] = []
    discovery_errors: list[dict[str, str]] = []
    for experiment in experiments:
        found, errors = discover_samples(experiment, spot_channel, nuclear_channel)
        LOGGER.info("Discovered %d samples for experiment %s", len(found), experiment.name)
        samples.extend(found)
        discovery_errors.extend(errors)

    if sample_regex:
        pattern = re.compile(sample_regex)
        samples = [sample for sample in samples if pattern.search(sample.sample_id)]
        discovery_errors = [
            error
            for error in discovery_errors
            if pattern.search(error.get("sample_id", ""))
        ]
        LOGGER.info("%d samples remain after --sample-regex", len(samples))

    for sample in samples:
        LOGGER.info("Sample: %s | %s", sample.sample_id, sample.nuclear_image)
    for error in discovery_errors:
        LOGGER.warning("Discovery error %s: %s", error["sample_id"], error["error"])

    if dry_run:
        return 0 if samples else 2
    if not samples:
        LOGGER.error("No valid samples found. Check roots, channel names, and regex.")
        return 2

    model, device_name = create_cellpose_model(parameters.model, parameters.device)
    rows: list[dict[str, object]] = [
        {**error, "status": "discovery_failed"} for error in discovery_errors
    ]
    for sample in samples:
        try:
            rows.append(
                process_sample(
                    sample,
                    parameters,
                    model,
                    device_name,
                    output_name=output_name,
                    overwrite=overwrite,
                    resume=resume,
                )
            )
        except Exception as exc:
            LOGGER.exception("Failed sample %s", sample.sample_id)
            rows.append(
                {
                    "status": "failed",
                    "experiment": sample.experiment,
                    "gene": sample.gene,
                    "condition": sample.condition,
                    "animal": sample.animal,
                    "sample_id": sample.sample_id,
                    "sample_path": str(sample.sample_dir),
                    "nuclear_image": str(sample.nuclear_image),
                    "output_name": output_name,
                    "output_dir": str(sample.results_dir / output_name),
                    "error": str(exc),
                }
            )
            if fail_fast:
                write_batch_csv(rows, summary_csv)
                raise
        write_batch_csv(rows, summary_csv)

    failed = sum(row.get("status") not in {"completed"} for row in rows)
    LOGGER.info("Batch complete: %d succeeded, %d failed", len(rows) - failed, failed)
    return 1 if failed else 0


def run_self_test() -> int:
    stitch_tested = False
    try:
        from cellpose import utils

        masks = np.zeros((3, 64, 64), dtype=np.uint32)
        masks[0, 10:26, 10:26] = 1
        masks[0, 38:52, 38:52] = 2
        masks[1, 11:27, 11:27] = 1
        masks[1, 37:51, 37:51] = 2
        masks[1, 4:12, 48:58] = 3
        masks[2, 12:28, 12:28] = 1
        masks[2, 36:50, 36:50] = 2
        stitched = utils.stitch3D(masks.copy(), stitch_threshold=0.25)
        assert int(stitched[0, 16, 16]) == int(stitched[1, 16, 16])
        assert int(stitched[1, 16, 16]) == int(stitched[2, 16, 16])
        assert int(stitched[0, 44, 44]) == int(stitched[2, 44, 44])
        assert int(stitched[1, 6, 52]) not in {
            0,
            int(stitched[1, 16, 16]),
            int(stitched[1, 44, 44]),
        }
        stitch_tested = True
    except ImportError:
        pass

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "experiment_root"
        valid_results = root / "Experiment" / "6hr_Amputation" / "Image1" / "565" / "results"
        valid_nuclear = root / "Experiment" / "6hr_Amputation" / "Image1" / "405"
        control_results = root / "Control" / "6hr_Amputation" / "Image1" / "565" / "results"
        control_nuclear = root / "Control" / "6hr_Amputation" / "Image1" / "405"
        valid_results.mkdir(parents=True)
        valid_nuclear.mkdir(parents=True)
        control_results.mkdir(parents=True)
        control_nuclear.mkdir(parents=True)
        # Discovery checks paths only; TIFF contents are irrelevant here and
        # keeping this test independent of tifffile makes it portable.
        (valid_nuclear / "Image1_405.tif").touch()
        (valid_nuclear / "._Image1_405.tif").touch()
        (control_nuclear / "Image1_405.tif").touch()
        samples, errors = discover_samples(
            ExperimentSpec("test", "gene", root), "565", "405"
        )
        assert not errors
        assert len(samples) == 1
        assert "Experiment" in samples[0].nuclear_image.parts

        # Reproduce the macOS/external-volume race from overwrite cleanup: the
        # AppleDouble sidecar vanishes immediately before rmtree unlinks it.
        race_dir = Path(temporary) / "overwrite_race"
        race_dir.mkdir()
        sidecar_name = "._.local_2d_masks.uint32.memmap"
        (race_dir / sidecar_name).touch()
        original_unlink = os.unlink
        race_triggered = False

        def unlink_with_appledouble_race(path, *args, **kwargs):
            nonlocal race_triggered
            if not race_triggered and Path(os.fspath(path)).name == sidecar_name:
                race_triggered = True
                original_unlink(path, *args, **kwargs)
                raise FileNotFoundError(os.fspath(path))
            return original_unlink(path, *args, **kwargs)

        try:
            os.unlink = unlink_with_appledouble_race
            remove_output_tree(race_dir)
        finally:
            os.unlink = original_unlink
        assert race_triggered
        assert not race_dir.exists()

    if stitch_tested:
        print("Official Cellpose stitch3D and Experiment-only discovery self-test passed.")
    else:
        print(
            "Experiment-only discovery self-test passed. Cellpose is not installed in "
            "this environment, so the official stitch3D test was skipped."
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Cellpose-SAM on each nuclear z-plane and use the official "
            "cellpose.utils.stitch3D function to assign global 3-D cell IDs."
        )
    )
    parser.add_argument(
        "--experiment",
        action="append",
        type=parse_experiment_argument,
        help=(
            "Repeatable NAME,GENE,ROOT override; defaults to experiments "
            "304/306/307/314/332. Supplying any --experiment replaces the "
            "complete default list."
        ),
    )
    parser.add_argument("--spot-channel", default="565")
    parser.add_argument("--nuclear-channel", default="405")
    parser.add_argument("--sample-regex", default=None)
    parser.add_argument("--model", default="cpsam_v2")
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--diameter",
        type=float,
        default=90.0,
        help="Expected nuclear diameter in pixels (production default: 90).",
    )
    parser.add_argument(
        "--niter",
        type=int,
        default=500,
        help="Mask-dynamics iterations (benchmark-selected default: 500).",
    )
    parser.add_argument(
        "--flow-threshold",
        type=float,
        default=0.8,
        help="Cellpose flow-error QC threshold (production default: 0.8).",
    )
    parser.add_argument(
        "--cellprob-threshold",
        type=float,
        default=-1.75,
        help="Cell-probability threshold (production default: -1.75).",
    )
    parser.add_argument("--min-size", type=int, default=15)
    parser.add_argument(
        "--tile-overlap",
        type=float,
        default=0.5,
        help="Cellpose tile overlap fraction (production default: 0.5).",
    )
    parser.add_argument(
        "--normalization",
        choices=("global", "plane", "none"),
        default="plane",
        help=(
            "global uses one sampled 1/99 range for the whole stack; plane "
            "normalizes every z-plane independently and matches benchmark "
            "norm-block=0; none disables normalization."
        ),
    )
    parser.add_argument("--normalization-planes", type=int, default=16)
    parser.add_argument(
        "--smooth-radius",
        type=float,
        default=4.0,
        help="Cellpose normalization low-pass smoothing radius in pixels.",
    )
    parser.add_argument(
        "--stitch-threshold",
        type=float,
        default=0.25,
        help="Official Cellpose adjacent-plane IoU threshold (default: 0.25).",
    )
    parser.add_argument("--compression", choices=("none", "zlib"), default="zlib")
    parser.add_argument(
        "--output-name",
        default="cellpose_label_plane_s4_cp-1p75_f0p8_d90_n500_st0p25",
        help=(
            "Subfolder created under each 565/results directory. Use a new name "
            "for each parameter test so earlier labels are preserved."
        ),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Batch QC CSV; defaults to <output-name>_batch_qc.csv.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true")
    mode.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >=1")
    if args.diameter is not None and args.diameter <= 0:
        raise SystemExit("--diameter must be >0")
    if args.niter < 1:
        raise SystemExit("--niter must be >=1")
    if args.min_size < 0:
        raise SystemExit("--min-size must be >=0")
    if args.normalization_planes < 1:
        raise SystemExit("--normalization-planes must be >=1")
    if args.smooth_radius < 0:
        raise SystemExit("--smooth-radius must be >=0")
    if args.smooth_radius > 0 and args.normalization == "none":
        raise SystemExit("--smooth-radius requires --normalization global or plane")
    if (
        not args.output_name
        or args.output_name in {".", ".."}
        or Path(args.output_name).name != args.output_name
    ):
        raise SystemExit("--output-name must be one non-empty folder name")
    for value, name in (
        (args.flow_threshold, "--flow-threshold"),
        (args.tile_overlap, "--tile-overlap"),
        (args.stitch_threshold, "--stitch-threshold"),
    ):
        if value < 0 or (name != "--flow-threshold" and value > 1):
            raise SystemExit(f"{name} is outside its valid range")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args)
    configure_logging(args.verbose)

    if args.self_test:
        return run_self_test()

    # Must be set before importing torch/cellpose. Unsupported MPS operations
    # then fall back to CPU instead of terminating a long batch.
    if args.device in {"auto", "mps"}:
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    experiments = args.experiment or [
        ExperimentSpec(name=name, gene=gene, root=Path(root))
        for name, gene, root in DEFAULT_EXPERIMENTS
    ]
    parameters = Parameters(
        model=args.model,
        device=args.device,
        batch_size=args.batch_size,
        diameter=args.diameter,
        niter=args.niter,
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob_threshold,
        min_size=args.min_size,
        tile_overlap=args.tile_overlap,
        normalization=args.normalization,
        normalization_planes=args.normalization_planes,
        smooth_radius=args.smooth_radius,
        stitch_threshold=args.stitch_threshold,
        compression=args.compression,
    )
    return run_pipeline(
        experiments=experiments,
        parameters=parameters,
        spot_channel=args.spot_channel,
        nuclear_channel=args.nuclear_channel,
        sample_regex=args.sample_regex,
        output_name=args.output_name,
        overwrite=args.overwrite,
        resume=args.resume,
        fail_fast=args.fail_fast,
        dry_run=args.dry_run,
        summary_csv=args.summary_csv or Path(f"{args.output_name}_batch_qc.csv"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
