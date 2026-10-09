#!/usr/bin/env python3
"""Small, repeatable Cellpose-SAM parameter benchmark for one large 2-D ROI.

The script reads only one plane from a potentially large TIFF, lets the user
select a representative ROI, and compares Cellpose normalization smoothing,
object scale, and mask-dynamics parameters.  The expensive network inference is
run once per ``smooth_radius``, ``diameter``, and normalization mode; the saved
flows are then reused across ``niter``, ``flow_threshold``, and
``cellprob_threshold`` combinations.
Outputs are Fiji-compatible label TIFFs, individual contour overlays, a contact
sheet, and a CSV summary.

Designed for Cellpose-SAM / Cellpose 4 in the user's existing ``cellpose``
environment.  Fiji Z positions are 1-based; ROI coordinates are 0-based pixels.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT = Path(
    "/Volumes/Backup Plus/Experiment_results/Cellpose_benchmark/"
    "Image1/C1-Composite.tif"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Cellpose-SAM smoothing, diameter, dynamics iterations, "
            "flow QC, cell-probability thresholds, and normalization on one "
            "representative ROI."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Exact output directory. If omitted, create a new timestamped "
            "directory beside the input TIFF on every run."
        ),
    )
    parser.add_argument(
        "--fiji-z",
        type=int,
        default=None,
        help="Fiji-style 1-based Z plane. If omitted, use the middle Z plane.",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="0-based channel index if the TIFF contains a C axis.",
    )
    parser.add_argument(
        "--time",
        type=int,
        default=0,
        help="0-based time index if the TIFF contains a T axis.",
    )
    roi_group = parser.add_mutually_exclusive_group()
    roi_group.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
        help="Exact 0-based ROI in source-image pixels.",
    )
    roi_group.add_argument(
        "--interactive-roi",
        action="store_true",
        help="Show the raw plane; click two opposite corners of the ROI.",
    )
    roi_group.add_argument(
        "--whole-plane",
        action="store_true",
        help="Benchmark the entire selected plane (may be slow).",
    )
    parser.add_argument(
        "--roi-size",
        type=int,
        default=2048,
        help="Centered square ROI size when no ROI option is supplied.",
    )
    parser.add_argument(
        "--cellprob-thresholds",
        nargs="+",
        type=float,
        default=[-2.0],
        help=(
            "Cell-probability threshold sweep. More-negative values are more "
            "permissive. The default is fixed at -2 because lower values "
            "plateaued on the benchmark ROI."
        ),
    )
    parser.add_argument(
        "--norm-blocks",
        nargs="+",
        type=int,
        default=[0],
        help=(
            "Normalization block sizes. 0 is global normalization; positive "
            "values enable Cellpose local/tile normalization."
        ),
    )
    parser.add_argument(
        "--diameters",
        nargs="+",
        type=float,
        default=[90.0],
        help=(
            "Estimated ROI diameters in pixels. Cellpose rescales each value "
            "toward its training diameter. Use 0 for diameter=None/no "
            "diameter-based rescaling."
        ),
    )
    parser.add_argument(
        "--niters",
        nargs="+",
        type=int,
        default=[500],
        help="Dynamics iteration counts used when reconstructing masks.",
    )
    parser.add_argument(
        "--flow-thresholds",
        nargs="+",
        type=float,
        default=[0.6],
        help=(
            "Flow-error QC thresholds. Larger positive values are more "
            "permissive; 0 disables flow QC entirely."
        ),
    )
    parser.add_argument(
        "--flow-threshold",
        dest="legacy_flow_threshold",
        type=float,
        default=None,
        help=(
            "Deprecated single-value alias. When supplied, it overrides "
            "--flow-thresholds."
        ),
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable Cellpose tiled flip augmentation (slower).",
    )
    parser.add_argument(
        "--sharpen",
        type=float,
        default=0.0,
        help=(
            "Cellpose normalization high-pass radius. Keep at 0 for the "
            "current internally textured nuclei unless deliberately testing "
            "sharpening. The script maps this to the parameter name supported "
            "by the installed Cellpose version."
        ),
    )
    parser.add_argument(
        "--smooth-radii",
        nargs="+",
        type=float,
        default=[0.0, 2.0, 4.0, 8.0],
        help=(
            "Cellpose normalization low-pass smoothing radii in pixels. "
            "Each value requires a separate neural-network inference; 0 "
            "keeps the unsmoothed baseline."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--bsize",
        type=int,
        default=256,
        help="Cellpose-SAM uses a fixed 256-pixel network tile.",
    )
    parser.add_argument("--tile-overlap", type=float, default=0.1)
    parser.add_argument(
        "--percentiles",
        nargs=2,
        type=float,
        default=[1.0, 99.0],
        metavar=("LOW", "HIGH"),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=15,
        help="Cellpose minimum mask size in pixels.",
    )
    return parser.parse_args()


def require_runtime() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import tifffile
        import torch
        from cellpose import models, transforms
    except ImportError as exc:
        raise SystemExit(
            "Missing a required package. Activate the same conda environment "
            "used by the working Cellpose-SAM pipeline, then rerun.\n"
            f"Original import error: {exc}"
        ) from exc
    return np, tifffile, plt, torch, models, transforms


def selected_indices(
    axes: str,
    shape: tuple[int, ...],
    fiji_z: int | None,
    channel: int,
    time_index: int,
) -> tuple[list[int], dict[str, int]]:
    """Return indices for axes preceding YX plus a readable index mapping."""
    leading_axes = axes[:-2]
    leading_shape = shape[:-2]
    out: list[int] = []
    selected: dict[str, int] = {}

    for axis, size in zip(leading_axes, leading_shape):
        if axis == "Z":
            index = size // 2 if fiji_z is None else fiji_z - 1
            if not 0 <= index < size:
                raise ValueError(f"--fiji-z must be between 1 and {size}")
            selected[axis] = index
        elif axis == "C":
            index = channel
            if not 0 <= index < size:
                raise ValueError(f"--channel must be between 0 and {size - 1}")
            selected[axis] = index
        elif axis == "T":
            index = time_index
            if not 0 <= index < size:
                raise ValueError(f"--time must be between 0 and {size - 1}")
            selected[axis] = index
        else:
            index = 0
            selected[axis] = index
        out.append(index)
    return out, selected


def read_one_plane(
    path: Path,
    fiji_z: int | None,
    channel: int,
    time_index: int,
    np: Any,
    tifffile: Any,
) -> tuple[Any, dict[str, Any]]:
    """Read one YX plane without loading a multi-gigabyte stack."""
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        axes = series.axes.upper()
        shape = tuple(int(v) for v in series.shape)

        if axes.endswith("YXS"):
            if len(shape) != 3:
                raise ValueError(f"Unsupported sampled TIFF axes: {axes} {shape}")
            plane = series.pages[0].asarray()
            if not 0 <= channel < plane.shape[-1]:
                raise ValueError(
                    f"--channel must be between 0 and {plane.shape[-1] - 1}"
                )
            plane = plane[..., channel]
            selected = {"S": channel}
        elif axes.endswith("YX"):
            leading_indices, selected = selected_indices(
                axes, shape, fiji_z, channel, time_index
            )
            if leading_indices:
                page_index = int(
                    np.ravel_multi_index(leading_indices, shape[:-2])
                )
            else:
                page_index = 0
            if page_index >= len(series.pages):
                raise ValueError(
                    "The TIFF page layout is not a simple YX-plane series. "
                    "Export the desired 405-nm plane from Fiji and rerun this "
                    "script on that 2-D TIFF."
                )
            plane = series.pages[page_index].asarray()
        else:
            raise ValueError(
                f"Unsupported TIFF axes {axes} with shape {shape}. Export one "
                "405-nm YX plane from Fiji and rerun."
            )

    plane = np.asarray(plane)
    plane = np.squeeze(plane)
    if plane.ndim != 2:
        raise ValueError(f"Expected a 2-D plane, got shape {plane.shape}")
    metadata = {
        "source_axes": axes,
        "source_shape": list(shape),
        "selected_zero_based_indices": selected,
        "plane_shape": list(plane.shape),
        "plane_dtype": str(plane.dtype),
    }
    return plane, metadata


def display_normalize(image: Any, np: Any) -> Any:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.float32)
    low, high = np.percentile(finite, (1.0, 99.5))
    if not math.isfinite(float(low)) or not math.isfinite(float(high)) or high <= low:
        low = float(finite.min())
        high = float(finite.max())
    if high <= low:
        return np.zeros(image.shape, dtype=np.float32)
    return np.clip((image.astype(np.float32) - low) / (high - low), 0.0, 1.0)


def interactive_roi(image: Any, np: Any, plt: Any) -> tuple[int, int, int, int]:
    preview = display_normalize(image, np)
    max_dim = max(image.shape)
    stride = max(1, int(math.ceil(max_dim / 1800)))
    shown = preview[::stride, ::stride]

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(shown, cmap="gray", interpolation="nearest")
    ax.set_title(
        "Click two opposite corners of the benchmark ROI (close window to cancel)"
    )
    points = plt.ginput(2, timeout=-1, show_clicks=True)
    plt.close(fig)
    if len(points) != 2:
        raise SystemExit("ROI selection cancelled.")

    x_values = [int(round(p[0] * stride)) for p in points]
    y_values = [int(round(p[1] * stride)) for p in points]
    x0, x1 = sorted(x_values)
    y0, y1 = sorted(y_values)
    x0 = max(0, min(x0, image.shape[1] - 1))
    y0 = max(0, min(y0, image.shape[0] - 1))
    x1 = max(x0 + 1, min(x1, image.shape[1]))
    y1 = max(y0 + 1, min(y1, image.shape[0]))
    return x0, y0, x1 - x0, y1 - y0


def resolve_roi(args: argparse.Namespace, image: Any) -> tuple[int, int, int, int]:
    height, width = image.shape
    if args.roi is not None:
        x, y, roi_width, roi_height = args.roi
    elif args.interactive_roi:
        raise RuntimeError("Interactive ROI must be resolved by the caller.")
    elif args.whole_plane:
        x, y, roi_width, roi_height = 0, 0, width, height
    else:
        roi_width = min(args.roi_size, width)
        roi_height = min(args.roi_size, height)
        x = (width - roi_width) // 2
        y = (height - roi_height) // 2

    if roi_width <= 0 or roi_height <= 0:
        raise ValueError("ROI width and height must be positive")
    if x < 0 or y < 0 or x + roi_width > width or y + roi_height > height:
        raise ValueError(
            f"ROI {(x, y, roi_width, roi_height)} is outside image shape "
            f"{(height, width)}"
        )
    return x, y, roi_width, roi_height


def choose_device(name: str, torch: Any) -> Any:
    if name == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            name = "mps"
        elif torch.cuda.is_available():
            name = "cuda"
        else:
            name = "cpu"
    if name == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise SystemExit("MPS was requested but is not available in this environment.")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available in this environment.")
    return torch.device(name)


def build_model(models: Any, device: Any) -> Any:
    signature = inspect.signature(models.CellposeModel.__init__)
    kwargs: dict[str, Any] = {}
    if "gpu" in signature.parameters:
        kwargs["gpu"] = device.type != "cpu"
    if "device" in signature.parameters:
        kwargs["device"] = device
    try:
        return models.CellposeModel(**kwargs)
    except Exception as exc:
        raise SystemExit(
            "Could not initialize Cellpose-SAM. This script expects the same "
            "Cellpose 4 environment as the existing pipeline.\n"
            f"Initialization error: {exc}"
        ) from exc


def normalization_sharpen_parameter(transforms: Any) -> str | None:
    """Return the sharpening keyword accepted by this Cellpose version."""
    try:
        parameters = inspect.signature(transforms.normalize_img).parameters
    except (AttributeError, TypeError, ValueError):
        return None
    if "sharpen_radius" in parameters:
        return "sharpen_radius"
    if "sharpen" in parameters:
        return "sharpen"
    return None


def normalization_smooth_parameter(transforms: Any) -> str | None:
    """Return the smoothing keyword accepted by this Cellpose version."""
    try:
        parameters = inspect.signature(transforms.normalize_img).parameters
    except (AttributeError, TypeError, ValueError):
        return None
    if "smooth_radius" in parameters:
        return "smooth_radius"
    if "smooth" in parameters:
        return "smooth"
    return None


def eval_kwargs(
    model: Any,
    args: argparse.Namespace,
    norm_block: int,
    diameter: float,
    smooth_radius: float,
    sharpen_parameter: str | None,
    smooth_parameter: str | None,
) -> dict[str, Any]:
    signature = inspect.signature(model.eval)
    allowed = signature.parameters
    normalize: dict[str, Any] = {
        "normalize": True,
        "percentile": [float(args.percentiles[0]), float(args.percentiles[1])],
        "tile_norm_blocksize": int(norm_block),
    }
    # Cellpose versions use different names for this optional setting.  More
    # importantly, do not pass any sharpening keyword for the default value of
    # zero: versions without the feature must still run normally.
    if float(args.sharpen) > 0 and sharpen_parameter is not None:
        normalize[sharpen_parameter] = float(args.sharpen)
    if float(smooth_radius) > 0 and smooth_parameter is not None:
        normalize[smooth_parameter] = float(smooth_radius)
    candidates: dict[str, Any] = {
        "normalize": normalize,
        "diameter": None if diameter == 0 else float(diameter),
        "resample": True,
        "interp": True,
        "augment": bool(args.augment),
        "batch_size": int(args.batch_size),
        "bsize": int(args.bsize),
        "tile_overlap": float(args.tile_overlap),
        "min_size": int(args.min_size),
        "do_3D": False,
        "channel_axis": None,
        "z_axis": None,
    }
    return {key: value for key, value in candidates.items() if key in allowed}


def unpack_masks(result: Any, np: Any) -> Any:
    masks = result[0] if isinstance(result, tuple) else result
    if isinstance(masks, list):
        if len(masks) != 1:
            raise ValueError(f"Expected one mask image, received {len(masks)}")
        masks = masks[0]
    masks = np.asarray(masks)
    masks = np.squeeze(masks)
    if masks.ndim != 2:
        raise ValueError(f"Expected 2-D masks, got shape {masks.shape}")
    return masks.astype(np.uint32, copy=False)


def masks_from_saved_flows(
    model: Any,
    flows: Any,
    image_shape: tuple[int, int],
    cellprob_threshold: float,
    flow_threshold: float,
    niter: int,
    args: argparse.Namespace,
    np: Any,
) -> Any:
    """Recompute masks without rerunning the expensive neural network."""
    d_p = np.asarray(flows[1])
    cell_probability = np.asarray(flows[2])
    if d_p.ndim == 3:
        d_p = d_p[:, np.newaxis, ...]
    if cell_probability.ndim == 2:
        cell_probability = cell_probability[np.newaxis, ...]
    if d_p.ndim != 4 or cell_probability.ndim != 3:
        raise ValueError(
            "Unexpected saved flow shapes: "
            f"dP={d_p.shape}, cellprob={cell_probability.shape}"
        )
    model_input_shape = (1, int(image_shape[0]), int(image_shape[1]), 3)
    masks = model._compute_masks(
        model_input_shape,
        d_p,
        cell_probability,
        flow_threshold=float(flow_threshold),
        cellprob_threshold=float(cellprob_threshold),
        min_size=int(args.min_size),
        max_size_fraction=0.4,
        niter=int(niter),
        do_3D=False,
        stitch_threshold=0.0,
    )
    return np.asarray(masks).squeeze().astype(np.uint32, copy=False)


def label_boundaries(labels: Any, np: Any) -> Any:
    boundary = np.zeros(labels.shape, dtype=bool)
    different_y = labels[:-1, :] != labels[1:, :]
    different_x = labels[:, :-1] != labels[:, 1:]
    boundary[:-1, :] |= different_y
    boundary[1:, :] |= different_y
    boundary[:, :-1] |= different_x
    boundary[:, 1:] |= different_x
    return boundary & (labels > 0)


def contour_overlay(image: Any, labels: Any, np: Any) -> Any:
    gray = display_normalize(image, np)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    boundary = label_boundaries(labels, np)
    rgb[boundary, 0] = 1.0
    rgb[boundary, 1] = 0.15
    rgb[boundary, 2] = 0.05
    return rgb


def threshold_tag(value: float) -> str:
    if value == 0:
        return "0"
    magnitude = f"{abs(value):g}".replace(".", "p")
    return f"m{magnitude}" if value < 0 else magnitude


def diameter_tag(value: float) -> str:
    return "auto" if value == 0 else threshold_tag(value)


def unique_preserve_order(values: Iterable[Any]) -> list[Any]:
    out: list[Any] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def new_timestamped_output_dir(input_path: Path) -> Path:
    """Return a new per-run output path without overwriting an earlier run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = input_path.parent / (
        f"{input_path.stem}_parameter_benchmark_{timestamp}"
    )
    candidate = base
    suffix = 2
    while candidate.exists():
        candidate = Path(f"{base}_{suffix:02d}")
        suffix += 1
    return candidate


def main() -> int:
    args = parse_args()
    np, tifffile, plt, torch, models, transforms = require_runtime()

    input_path = args.input.expanduser()
    if not input_path.exists():
        raise SystemExit(f"Input TIFF does not exist: {input_path}")

    output_dir = (
        args.output.expanduser()
        if args.output is not None
        else new_timestamped_output_dir(input_path)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading one plane from: {input_path}")
    plane, source_metadata = read_one_plane(
        input_path, args.fiji_z, args.channel, args.time, np, tifffile
    )
    print(
        f"TIFF axes={source_metadata['source_axes']} "
        f"shape={tuple(source_metadata['source_shape'])}; "
        f"selected={source_metadata['selected_zero_based_indices']}"
    )

    if args.interactive_roi:
        roi_spec = interactive_roi(plane, np, plt)
    else:
        roi_spec = resolve_roi(args, plane)
    x, y, width, height = roi_spec
    roi = np.ascontiguousarray(plane[y : y + height, x : x + width])
    print(f"Benchmark ROI: x={x}, y={y}, width={width}, height={height}")

    tifffile.imwrite(output_dir / "benchmark_roi_raw.tif", roi)
    plt.imsave(
        output_dir / "benchmark_roi_preview.png",
        display_normalize(roi, np),
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    device = choose_device(args.device, torch)
    print(f"Using device: {device}")
    model = build_model(models, device)

    thresholds = unique_preserve_order(args.cellprob_thresholds)
    norm_blocks = unique_preserve_order(args.norm_blocks)
    smooth_radii = unique_preserve_order(args.smooth_radii)
    diameters = unique_preserve_order(args.diameters)
    niters = unique_preserve_order(args.niters)
    flow_thresholds = unique_preserve_order(args.flow_thresholds)
    if args.legacy_flow_threshold is not None:
        flow_thresholds = [float(args.legacy_flow_threshold)]
    if any(block < 0 for block in norm_blocks):
        raise SystemExit("--norm-blocks values must be >= 0")
    if any(radius < 0 for radius in smooth_radii):
        raise SystemExit("--smooth-radii values must be >= 0")
    if any(diameter < 0 for diameter in diameters):
        raise SystemExit("--diameters values must be >= 0 (0 means automatic)")
    if any(niter <= 0 for niter in niters):
        raise SystemExit("--niters values must be positive integers")
    if any(flow_threshold < 0 for flow_threshold in flow_thresholds):
        raise SystemExit("--flow-thresholds values must be >= 0")
    if args.sharpen < 0:
        raise SystemExit("--sharpen must be >= 0")
    sharpen_parameter = normalization_sharpen_parameter(transforms)
    smooth_parameter = normalization_smooth_parameter(transforms)
    effective_sharpen = float(args.sharpen) if sharpen_parameter else 0.0
    if args.sharpen > 0 and sharpen_parameter is None:
        print(
            "WARNING: this Cellpose version does not expose a normalization "
            "sharpening parameter; --sharpen will be ignored."
        )
    if any(radius > 0 for radius in smooth_radii) and smooth_parameter is None:
        raise SystemExit(
            "This Cellpose version does not expose normalize_img smoothing; "
            "cannot run the requested --smooth-radii sweep."
        )

    settings = {
        "input": str(input_path),
        "output": str(output_dir),
        "source": source_metadata,
        "roi_xywh": [x, y, width, height],
        "device": str(device),
        "cellprob_thresholds": thresholds,
        "norm_blocks": norm_blocks,
        "smooth_radii": smooth_radii,
        "smooth_parameter_name": smooth_parameter,
        "diameters": diameters,
        "niters": niters,
        "flow_thresholds": flow_thresholds,
        "augment": args.augment,
        "sharpen": effective_sharpen,
        "sharpen_requested": args.sharpen,
        "sharpen_parameter_name": sharpen_parameter,
        "resample": True,
        "interp": True,
        "batch_size": args.batch_size,
        "bsize": args.bsize,
        "tile_overlap": args.tile_overlap,
        "percentiles": args.percentiles,
        "min_size": args.min_size,
    }
    (output_dir / "benchmark_settings.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )

    rows: list[dict[str, Any]] = []
    panels: list[tuple[str, Any]] = []
    eval_signature = inspect.signature(model.eval)
    if "cellprob_threshold" not in eval_signature.parameters:
        raise SystemExit(
            "This Cellpose model.eval does not expose cellprob_threshold; "
            "check that the working Cellpose-SAM environment is active."
        )

    total_runs = (
        len(norm_blocks)
        * len(smooth_radii)
        * len(diameters)
        * len(niters)
        * len(flow_thresholds)
        * len(thresholds)
    )
    run_number = 0
    for norm_block in norm_blocks:
        norm_mode = "global" if norm_block == 0 else f"local{norm_block}"
        inference_modes = [
            (smooth_radius, diameter)
            for smooth_radius in smooth_radii
            for diameter in diameters
        ]
        for smooth_radius, diameter in inference_modes:
            smooth_mode = f"smooth{threshold_tag(float(smooth_radius))}"
            base_kwargs = eval_kwargs(
                model,
                args,
                norm_block,
                float(diameter),
                float(smooth_radius),
                sharpen_parameter,
                smooth_parameter,
            )
            mode = (
                f"{norm_mode}_{smooth_mode}_"
                f"diam{diameter_tag(float(diameter))}"
            )
            can_reuse_flows = (
                hasattr(model, "_compute_masks")
                and "compute_masks" in eval_signature.parameters
            )
            saved_flows = None
            network_seconds = 0.0
            if can_reuse_flows:
                print(
                    f"Running the neural network once for {mode}; niter, flow "
                    "QC, and cellprob combinations will reuse its flows."
                )
                network_started = time.perf_counter()
                inference_result = model.eval(
                    roi,
                    cellprob_threshold=0.0,
                    compute_masks=False,
                    **base_kwargs,
                )
                network_seconds = time.perf_counter() - network_started
                if (
                    not isinstance(inference_result, tuple)
                    or len(inference_result) < 2
                ):
                    raise ValueError("Unexpected Cellpose result while saving flows")
                saved_flows = inference_result[1]
                cellprob_map = np.asarray(saved_flows[2]).squeeze().astype(
                    np.float32, copy=False
                )
                tifffile.imwrite(
                    output_dir / f"cellprob_{mode}.tif",
                    cellprob_map,
                    photometric="minisblack",
                )
                plt.imsave(
                    output_dir / f"cellprob_{mode}.png",
                    cellprob_map,
                    cmap="magma",
                    vmin=-4,
                    vmax=4,
                )
                del cellprob_map

            for niter in niters:
                for flow_threshold in flow_thresholds:
                    for threshold in thresholds:
                        run_number += 1
                        name = (
                            f"{mode}_niter{int(niter)}_"
                            f"flow{threshold_tag(float(flow_threshold))}_"
                            f"cellprob{threshold_tag(float(threshold))}"
                        )
                        print(
                            f"[{run_number}/{total_runs}] {name}: "
                            f"smooth={smooth_radius:g}, diameter={diameter:g}, "
                            f"niter={niter}, "
                            f"flow={flow_threshold:g}, cellprob={threshold:g}"
                        )
                        started = time.perf_counter()
                        if saved_flows is not None:
                            masks = masks_from_saved_flows(
                                model,
                                saved_flows,
                                roi.shape,
                                float(threshold),
                                float(flow_threshold),
                                int(niter),
                                args,
                                np,
                            )
                            result = None
                        else:
                            eval_run_kwargs = dict(base_kwargs)
                            eval_run_kwargs.update(
                                {
                                    "flow_threshold": float(flow_threshold),
                                    "niter": int(niter),
                                }
                            )
                            result = model.eval(
                                roi,
                                cellprob_threshold=float(threshold),
                                **eval_run_kwargs,
                            )
                            masks = unpack_masks(result, np)
                        elapsed = time.perf_counter() - started
                        n_masks = int(masks.max())
                        foreground_fraction = float(
                            np.count_nonzero(masks) / masks.size
                        )
                        mask_sizes = np.bincount(masks.ravel())[1:]
                        nonzero_sizes = mask_sizes[mask_sizes > 0]
                        median_area = (
                            float(np.median(nonzero_sizes))
                            if nonzero_sizes.size
                            else 0.0
                        )
                        area_p10 = (
                            float(np.percentile(nonzero_sizes, 10))
                            if nonzero_sizes.size
                            else 0.0
                        )
                        area_p90 = (
                            float(np.percentile(nonzero_sizes, 90))
                            if nonzero_sizes.size
                            else 0.0
                        )
                        n_masks_lt1000 = int(
                            np.count_nonzero(nonzero_sizes < 1000)
                        )

                        label_path = output_dir / f"labels_{name}.tif"
                        overlay_path = output_dir / f"overlay_{name}.png"
                        tifffile.imwrite(
                            label_path, masks, photometric="minisblack"
                        )
                        overlay = contour_overlay(roi, masks, np)
                        plt.imsave(overlay_path, overlay)

                        max_display_dim = 1200
                        display_stride = max(
                            1,
                            int(
                                math.ceil(
                                    max(overlay.shape[:2]) / max_display_dim
                                )
                            ),
                        )
                        panels.append(
                            (
                                f"{norm_mode}; smooth={smooth_radius:g}; "
                                f"diam={diameter:g}\n"
                                f"niter={niter}; "
                                f"flow={flow_threshold:g}; "
                                f"cellprob={threshold:g}; n={n_masks}; "
                                f"area={median_area:.0f}px",
                                overlay[::display_stride, ::display_stride],
                            )
                        )
                        rows.append(
                            {
                                "name": name,
                                "norm_block": norm_block,
                                "smooth_radius": smooth_radius,
                                "smooth_parameter_name": smooth_parameter,
                                "diameter": diameter,
                                "niter": niter,
                                "flow_threshold": flow_threshold,
                                "cellprob_threshold": threshold,
                                "augment": args.augment,
                                "sharpen": effective_sharpen,
                                "sharpen_requested": args.sharpen,
                                "sharpen_parameter_name": sharpen_parameter,
                                "resample": True,
                                "interp": True,
                                "n_masks": n_masks,
                                "foreground_fraction": foreground_fraction,
                                "median_mask_area_px": median_area,
                                "mask_area_p10_px": area_p10,
                                "mask_area_p90_px": area_p90,
                                "n_masks_lt1000px": n_masks_lt1000,
                                "network_seconds_for_inference_mode": (
                                    network_seconds
                                ),
                                "mask_reconstruction_seconds": elapsed,
                                "labels_file": label_path.name,
                                "overlay_file": overlay_path.name,
                            }
                        )
                        print(
                            f"    masks={n_masks}, "
                            f"foreground={foreground_fraction:.3f}, "
                            f"median_area={median_area:.1f}px, "
                            f"time={elapsed:.1f}s"
                        )
                        del masks, result, overlay
            del saved_flows

    with (output_dir / "benchmark_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    preferred_columns = len(flow_thresholds) * len(thresholds)
    if preferred_columns == 1 and len(smooth_radii) > 1:
        preferred_columns = len(smooth_radii)
    columns = min(4, preferred_columns, len(panels))
    rows_count = int(math.ceil(len(panels) / columns))
    fig, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(5.2 * columns, 5.2 * rows_count),
        squeeze=False,
    )
    for ax, panel in zip(axes.flat, panels):
        title, image = panel
        ax.imshow(image, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    for ax in axes.flat[len(panels) :]:
        ax.axis("off")
    fig.suptitle(
        "Cellpose-SAM smoothing benchmark — red lines are boundaries",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_dir / "threshold_comparison.png", dpi=180)
    plt.close(fig)

    print("\nFinished.")
    print(f"Open this first: {output_dir / 'threshold_comparison.png'}")
    print(f"Fiji label TIFFs and the CSV summary are in: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nCancelled by user.", file=sys.stderr)
        raise SystemExit(130)
