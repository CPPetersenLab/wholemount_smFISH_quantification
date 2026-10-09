#!/usr/bin/env python3
"""Batch-run the Big-FISH workflow from 1_Get_spots.ipynb.

The default configuration scans:
    /Volumes/Backup Plus/Experiment_results/304_Analysis_results/Experiment/**/565/*.tif(f)

Each image is isolated from the others.  If one image fails, its path, the
processing stage, and the complete traceback are logged, and the next image is
processed.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tifffile import imread, imwrite

import bigfish.detection
import bigfish.plot
import bigfish.stack


DEFAULT_EXPERIMENT_DIR = Path(
    "/Volumes/Backup Plus/Experiment_results/304_Analysis_results/Experiment"
)

# Requested fixed threshold.  This is passed directly to
# bigfish.detection.detect_spots(..., threshold=...).
experiment_average_threshold = 12.0

# Parameters copied from 1_Get_spots.ipynb.
MINIMUM_DISTANCE = (2, 2, 2)
LOG_KERNEL_SIZE = (1, 1.5, 1.5)
VOXEL_SIZE = (361, 75, 75)
SPOT_RADIUS = (600, 300, 300)
DECOMPOSITION = (0.7, 1, 5)
MIN_SPOTS_FOR_CLUSTERS = 4
CLUSTER_RADIUS = 250
PLOT_SPOT_SIZE = 4


class ImageProcessingError(RuntimeError):
    """An exception annotated with the image and processing stage."""

    def __init__(self, image_path: Path, stage: str, original: Exception):
        super().__init__(f"{image_path} failed during {stage}: {original}")
        self.image_path = image_path
        self.stage = stage
        self.original = original


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 1_Get_spots.ipynb Big-FISH workflow on every TIFF in "
            "Experiment/**/565. Errors are reported and skipped per image."
        )
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR,
        help=f"Experiment root (default: {DEFAULT_EXPERIMENT_DIR})",
    )
    parser.add_argument(
        "--experiment-average-threshold",
        type=float,
        default=experiment_average_threshold,
        help=f"Fixed Big-FISH detection threshold (default: {experiment_average_threshold:g})",
    )
    parser.add_argument(
        "--channel-dir",
        default="565",
        help="Name of the image channel directory to scan (default: 565)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip images already completed with the same threshold",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def discover_images(experiment_dir: Path, channel_dir: str) -> list[Path]:
    """Find TIFF files directly inside channel directories, never in results."""
    images = [
        path
        for path in experiment_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".tif", ".tiff"}
        and path.parent.name == channel_dir
        # macOS writes AppleDouble metadata files such as ._image.tif on
        # external drives.  They keep the TIFF suffix but are not TIFF images.
        and not path.name.startswith("._")
    ]
    return sorted(images, key=lambda path: str(path).lower())


def make_logger(experiment_dir: Path, threshold: float) -> tuple[logging.Logger, Path]:
    threshold_label = f"{threshold:g}".replace(".", "p")
    log_path = experiment_dir / f"bigfish_batch_threshold{threshold_label}.log"
    logger = logging.getLogger("bigfish_batch")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger, log_path


def output_directory(image_path: Path, files_per_parent: Counter[Path]) -> Path:
    """Use the notebook layout unless a 565 folder contains multiple inputs."""
    base = image_path.parent / "results"
    if files_per_parent[image_path.parent] > 1:
        return base / image_path.stem
    return base


def completion_marker(output_dir: Path, image_path: Path) -> Path:
    return output_dir / f".{image_path.stem}_bigfish_complete.json"


def is_completed(output_dir: Path, image_path: Path, threshold: float) -> bool:
    marker = completion_marker(output_dir, image_path)
    if not marker.exists():
        return False
    try:
        metadata = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False
    return (
        metadata.get("status") == "success"
        and metadata.get("image") == str(image_path)
        and float(metadata.get("experiment_average_threshold")) == float(threshold)
    )


def save_histogram(values: list[float], average: float, output_path: Path, title: str) -> None:
    plt.figure(figsize=(7, 5))
    sns.histplot(values, kde=len(values) > 1)
    plt.axvline(
        average,
        color="r",
        linestyle="--",
        linewidth=1,
        label=f"Average Signal Intensity: {average:.2f}",
    )
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_post_decomposition_histogram(
    pre_values: list[float], post_values: list[float], output_path: Path
) -> None:
    plt.figure(figsize=(7, 5))
    sns.histplot(post_values, kde=len(post_values) > 1)
    plt.axvline(
        float(np.mean(pre_values)),
        color="r",
        linestyle="--",
        linewidth=1,
        label=f"Average PreProcessing Intensity: {np.mean(pre_values):.2f}",
    )
    plt.axvline(
        float(np.mean(post_values)),
        color="b",
        linestyle="--",
        linewidth=1,
        label=f"Average PostDecomposition Intensity: {np.mean(post_values):.2f}",
    )
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.title("Histogram of smFISH Spot Intensities")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def find_spots_around(
    coordinate: np.ndarray, array: np.ndarray, max_iterations: int = 20
) -> np.ndarray:
    """Collect downhill-connected nonzero voxels around one detected maximum."""
    start = tuple(int(value) for value in coordinate[:3])
    frontier = [start]
    collected = [start]
    seen = {start}
    shape = array.shape

    for _ in range(max_iterations):
        new_frontier: list[tuple[int, int, int]] = []
        for z, y, x in frontier:
            current_value = array[z, y, x]
            for nz, ny, nx in (
                (z, y, x - 1),
                (z, y, x + 1),
                (z, y - 1, x),
                (z, y + 1, x),
                (z - 1, y, x),
                (z + 1, y, x),
            ):
                neighbor = (nz, ny, nx)
                if (
                    0 <= nz < shape[0]
                    and 0 <= ny < shape[1]
                    and 0 <= nx < shape[2]
                    and neighbor not in seen
                    and array[neighbor] > 0
                    and current_value >= array[neighbor]
                ):
                    seen.add(neighbor)
                    collected.append(neighbor)
                    new_frontier.append(neighbor)
        if not new_frontier:
            break
        frontier = new_frontier

    return np.asarray(collected, dtype=np.int64)


def spot_intensities(
    spots: np.ndarray, raw_image: np.ndarray, log_image: np.ndarray
) -> list[float]:
    intensities: list[float] = []
    for spot in spots:
        coordinates = find_spots_around(spot, log_image, max_iterations=20)
        values = raw_image[
            coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
        ]
        intensities.append(float(np.sum(values, dtype=np.float64)))
    return intensities


def write_spot_info(output_dir: Path, spots: np.ndarray, threshold: float) -> None:
    (output_dir / "spot_info.txt").write_text(
        "detected spots\n"
        f"shape: {spots.shape}\n"
        f"dtype: {spots.dtype}\n"
        f"threshold: {threshold}\n",
        encoding="utf-8",
    )


def write_decomposition_info(
    output_dir: Path, before: np.ndarray, after: np.ndarray
) -> None:
    (output_dir / "spot_decomposition_info.txt").write_text(
        "detected spots before decomposition\n"
        f"shape: {before.shape}\n"
        f"dtype: {before.dtype}\n"
        "detected spots after decomposition\n"
        f"shape: {after.shape}\n"
        f"dtype: {after.dtype}\n",
        encoding="utf-8",
    )


def write_cluster_info(
    output_dir: Path, spots_post_clustering: np.ndarray, clusters: np.ndarray
) -> None:
    (output_dir / "Cluster_info.txt").write_text(
        "detected spots after clustering\n"
        f"shape: {spots_post_clustering.shape}\n"
        f"dtype: {spots_post_clustering.dtype}\n"
        "detected clusters\n"
        f"shape: {clusters.shape}\n"
        f"dtype: {clusters.dtype}\n",
        encoding="utf-8",
    )


def build_spot_plot(image_shape: tuple[int, ...], spots: np.ndarray) -> np.ndarray:
    """Make the notebook-style ring plot around each detected spot."""
    spot_plot = np.zeros(image_shape, dtype=np.uint8)
    max_y = image_shape[1] - 1
    max_x = image_shape[2] - 1
    radius = PLOT_SPOT_SIZE

    offsets = [
        (dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if abs(dy) + abs(dx) == radius
    ]
    for spot in spots:
        z, y, x = (int(value) for value in spot[:3])
        for dy, dx in offsets:
            py, px = y + dy, x + dx
            if 0 <= py <= max_y and 0 <= px <= max_x:
                spot_plot[z, py, px] = 255
    return spot_plot


def process_image(
    image_path: Path,
    output_dir: Path,
    threshold: float,
    logger: logging.Logger,
) -> dict[str, object]:
    stage = "initialization"
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_image: np.ndarray | None = None
    log_image: np.ndarray | None = None
    spot_plot: np.ndarray | None = None

    try:
        stage = "reading TIFF"
        raw_image = imread(image_path)
        if raw_image.ndim != 3:
            raise ValueError(f"expected a 3D ZYX TIFF, got shape {raw_image.shape}")

        stage = "LoG filtering"
        log_image = bigfish.stack.log_filter(raw_image, LOG_KERNEL_SIZE)
        imwrite(
            output_dir / "smFISHChannelArray_LoG.tif",
            log_image,
            photometric="minisblack",
        )

        stage = "spot detection"
        spots, used_threshold = bigfish.detection.detect_spots(
            images=raw_image,
            threshold=threshold,
            return_threshold=True,
            voxel_size=VOXEL_SIZE,
            spot_radius=SPOT_RADIUS,
            log_kernel_size=LOG_KERNEL_SIZE,
            minimum_distance=MINIMUM_DISTANCE,
        )
        spots = np.asarray(spots)
        np.save(output_dir / "spots.npy", spots)
        write_spot_info(output_dir, spots, float(used_threshold))
        logger.info("DETECTED | %s | spots=%d | threshold=%s", image_path, len(spots), used_threshold)

        stage = "elbow plot"
        bigfish.plot.plot_elbow(
            images=raw_image,
            voxel_size=VOXEL_SIZE,
            spot_radius=SPOT_RADIUS,
            path_output=str(output_dir / "Elbow.png"),
            show=False,
        )
        plt.close("all")

        if len(spots) == 0:
            logger.warning("ZERO SPOTS | %s", image_path)
            post_decomposition = spots.copy()
            pre_intensities: list[float] = []
            post_intensities: list[float] = []
        else:
            stage = "dense-region decomposition"
            post_decomposition, _dense_regions, _reference_spot = (
                bigfish.detection.decompose_dense(
                    image=raw_image,
                    spots=spots,
                    voxel_size=VOXEL_SIZE,
                    spot_radius=SPOT_RADIUS,
                    alpha=DECOMPOSITION[0],
                    beta=DECOMPOSITION[1],
                    gamma=DECOMPOSITION[2],
                )
            )
            post_decomposition = np.asarray(post_decomposition)

            stage = "pre-decomposition intensity measurements"
            pre_intensities = spot_intensities(spots, raw_image, log_image)
            save_histogram(
                pre_intensities,
                float(np.mean(pre_intensities)),
                output_dir / "histogram_of_smFISH_spot_intensities.png",
                "Histogram of smFISH Spot Intensities",
            )

            stage = "post-decomposition intensity measurements"
            post_intensities = spot_intensities(
                post_decomposition, raw_image, log_image
            )
            if post_intensities:
                save_post_decomposition_histogram(
                    pre_intensities,
                    post_intensities,
                    output_dir
                    / "histogram_of_smFISH_spot_intensities_postdecomposition.png",
                )

        stage = "saving decomposed spots"
        np.save(output_dir / "spots_post_decomposition.npy", post_decomposition)
        write_decomposition_info(output_dir, spots, post_decomposition)

        stage = "cluster detection"
        if len(post_decomposition) == 0:
            # Big-FISH cluster arrays add one cluster-id column to 3D spots and
            # produce z/y/x/count/index cluster rows.
            spots_post_clustering = np.empty((0, 4), dtype=np.int64)
            clusters = np.empty((0, 5), dtype=np.float64)
        else:
            spots_post_clustering, clusters = bigfish.detection.detect_clusters(
                spots=post_decomposition,
                voxel_size=VOXEL_SIZE,
                radius=CLUSTER_RADIUS,
                nb_min_spots=MIN_SPOTS_FOR_CLUSTERS,
            )
        np.save(output_dir / "spots_post_clustering.npy", spots_post_clustering)
        np.save(output_dir / "clusters.npy", clusters)
        write_cluster_info(output_dir, spots_post_clustering, clusters)

        stage = "spot plot"
        spot_plot = build_spot_plot(raw_image.shape, post_decomposition)
        imwrite(output_dir / "spotPlot.tif", spot_plot, photometric="minisblack")

        elapsed = time.monotonic() - started
        metadata = {
            "status": "success",
            "image": str(image_path),
            "output_directory": str(output_dir),
            "experiment_average_threshold": float(threshold),
            "threshold_returned_by_bigfish": float(used_threshold),
            "spots_before_decomposition": int(len(spots)),
            "spots_after_decomposition": int(len(post_decomposition)),
            "clusters": int(len(clusters)),
            "elapsed_seconds": round(elapsed, 3),
            "completed_at_utc": utc_now(),
        }
        completion_marker(output_dir, image_path).write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        return metadata
    except Exception as exc:
        raise ImageProcessingError(image_path, stage, exc) from exc
    finally:
        plt.close("all")
        del raw_image, log_image, spot_plot
        gc.collect()


def write_summary(
    experiment_dir: Path,
    threshold: float,
    records: list[dict[str, object]],
) -> tuple[Path, Path]:
    threshold_label = f"{threshold:g}".replace(".", "p")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = experiment_dir / f"bigfish_threshold{threshold_label}_summary_{timestamp}.json"
    csv_path = experiment_dir / f"bigfish_threshold{threshold_label}_summary_{timestamp}.csv"

    counts = Counter(str(record["status"]) for record in records)
    payload = {
        "experiment_directory": str(experiment_dir),
        "experiment_average_threshold": threshold,
        "completed_at_utc": utc_now(),
        "counts": dict(counts),
        "images": records,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fieldnames = [
        "status",
        "image",
        "output_directory",
        "stage",
        "error_type",
        "error",
        "spots_before_decomposition",
        "spots_after_decomposition",
        "clusters",
        "elapsed_seconds",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    return json_path, csv_path


def main() -> int:
    args = parse_args()
    experiment_dir = args.experiment_dir.expanduser().resolve()
    threshold = float(args.experiment_average_threshold)

    if not experiment_dir.is_dir():
        print(f"ERROR: Experiment directory does not exist: {experiment_dir}", file=sys.stderr)
        return 2

    logger, log_path = make_logger(experiment_dir, threshold)
    images = discover_images(experiment_dir, args.channel_dir)
    if not images:
        logger.error(
            "No .tif/.tiff images found directly under **/%s in %s",
            args.channel_dir,
            experiment_dir,
        )
        return 2

    grouped: defaultdict[Path, list[Path]] = defaultdict(list)
    for image_path in images:
        grouped[image_path.parent].append(image_path)
    files_per_parent = Counter({parent: len(paths) for parent, paths in grouped.items()})

    logger.info("START | root=%s", experiment_dir)
    logger.info("SETTINGS | experiment_average_threshold=%s | images=%d", threshold, len(images))
    for parent, parent_images in grouped.items():
        if len(parent_images) > 1:
            logger.warning(
                "MULTIPLE INPUTS | %s contains %d TIFFs; outputs will be separated by image stem",
                parent,
                len(parent_images),
            )

    records: list[dict[str, object]] = []
    for index, image_path in enumerate(images, start=1):
        output_dir = output_directory(image_path, files_per_parent)
        if args.resume and is_completed(output_dir, image_path, threshold):
            logger.info("SKIP COMPLETED [%d/%d] | %s", index, len(images), image_path)
            records.append(
                {
                    "status": "skipped_completed",
                    "image": str(image_path),
                    "output_directory": str(output_dir),
                }
            )
            continue

        logger.info("PROCESS [%d/%d] | %s", index, len(images), image_path)
        try:
            record = process_image(image_path, output_dir, threshold, logger)
            records.append(record)
            logger.info(
                "SUCCESS [%d/%d] | %s | %.1f s",
                index,
                len(images),
                image_path,
                record["elapsed_seconds"],
            )
        except ImageProcessingError as exc:
            error_traceback = traceback.format_exc()
            record = {
                "status": "failed",
                "image": str(image_path),
                "output_directory": str(output_dir),
                "stage": exc.stage,
                "error_type": type(exc.original).__name__,
                "error": str(exc.original),
                "traceback": error_traceback,
            }
            records.append(record)
            logger.error(
                "FAILED [%d/%d] | image=%s | stage=%s | %s: %s\n%s",
                index,
                len(images),
                image_path,
                exc.stage,
                type(exc.original).__name__,
                exc.original,
                error_traceback,
            )
            logger.info("CONTINUE | moving to the next image")

    json_summary, csv_summary = write_summary(experiment_dir, threshold, records)
    counts = Counter(str(record["status"]) for record in records)
    logger.info(
        "FINISHED | success=%d | failed=%d | skipped_completed=%d",
        counts["success"],
        counts["failed"],
        counts["skipped_completed"],
    )
    logger.info("SUMMARY JSON | %s", json_summary)
    logger.info("SUMMARY CSV | %s", csv_summary)
    logger.info("FULL LOG | %s", log_path)

    # Individual failures are deliberately non-fatal: they are reported in the
    # summaries and log, while the batch itself finishes normally.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
