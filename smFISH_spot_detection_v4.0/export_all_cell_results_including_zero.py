#!/usr/bin/env python3
"""Collect the five all-cell (including zero) tables into one folder.

The script reads the per-experiment wide tables produced by
``reassign_smfish_3d.py`` and creates exactly one prefixed CSV per experiment::

    304_output_all_cells_including_zero.csv
    306_output_all_cells_including_zero.csv
    307_output_all_cells_including_zero.csv
    314_output_all_cells_including_zero.csv
    332_output_all_cells_including_zero.csv

Each column remains one biological replicate, and all zero-expression cells
are preserved unchanged. Existing destination files are atomically refreshed.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import tempfile
from pathlib import Path
from typing import Sequence


EXPERIMENTS = ("304", "306", "307", "314", "332")

DEFAULT_INPUT_ROOT = Path(
    "/Volumes/Backup Plus/Experiment_results/smFISH_cellpose_assignment"
)
DEFAULT_OUTPUT_DIR = Path(
    "/Volumes/Backup Plus/Experiment_results/all_cell_results_including_zero"
)

# The first name is produced by the current assignment pipeline. The second
# keeps this exporter compatible with the earlier wide-table naming.
SOURCE_FILENAMES = (
    "all_cells_including_zero_wide.csv",
    "all_cells_wide_including_zero.csv",
)


def locate_source(input_root: Path, experiment: str) -> Path:
    results_dir = input_root / experiment / "results"
    for filename in SOURCE_FILENAMES:
        candidate = results_dir / filename
        if candidate.is_file():
            return candidate
    expected = " or ".join(str(results_dir / name) for name in SOURCE_FILENAMES)
    raise FileNotFoundError(f"Experiment {experiment}: expected {expected}")


def inspect_csv(path: Path) -> tuple[int, int, int]:
    """Return biological-replicate columns, data rows, and numeric zero cells."""

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"CSV is empty: {path}") from exc
        if not header or not any(cell.strip() for cell in header):
            raise ValueError(f"CSV has no biological-replicate columns: {path}")
        row_count = 0
        zero_count = 0
        for row in reader:
            row_count += 1
            for value in row:
                stripped = value.strip()
                if not stripped:
                    continue
                try:
                    if float(stripped) == 0:
                        zero_count += 1
                except ValueError:
                    pass
    return len(header), row_count, zero_count


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def export_experiments(input_root: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    completed = 0

    for experiment in EXPERIMENTS:
        try:
            source = locate_source(input_root, experiment)
            n_columns, n_rows, n_zeros = inspect_csv(source)
            destination = output_dir / (
                f"{experiment}_output_all_cells_including_zero.csv"
            )
            atomic_copy(source, destination)
            print(
                f"WROTE {destination} | biological replicates={n_columns}, "
                f"rows={n_rows}, numeric zeros={n_zeros}",
                flush=True,
            )
            completed += 1
        except Exception as exc:
            message = f"ERROR experiment {experiment}: {type(exc).__name__}: {exc}"
            print(message, flush=True)
            errors.append(message)

    print(
        f"FINISHED: {completed}/{len(EXPERIMENTS)} experiments exported to "
        f"{output_dir}",
        flush=True,
    )
    return 1 if errors else 0


def run_self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="all_cell_export_test_") as temporary:
        base = Path(temporary)
        input_root = base / "input"
        output_dir = base / "output"
        expected_bytes: dict[str, bytes] = {}

        for experiment in EXPERIMENTS:
            source = (
                input_root
                / experiment
                / "results"
                / "all_cells_including_zero_wide.csv"
            )
            source.parent.mkdir(parents=True)
            content = "condition__Image1,condition__Image2\n0,2\n1,0\n,3\n"
            source.write_text(content, encoding="utf-8")
            expected_bytes[experiment] = source.read_bytes()

        status = export_experiments(input_root, output_dir)
        if status != 0:
            raise AssertionError(f"Self-test exporter returned {status}")
        for experiment in EXPERIMENTS:
            destination = output_dir / (
                f"{experiment}_output_all_cells_including_zero.csv"
            )
            if destination.read_bytes() != expected_bytes[experiment]:
                raise AssertionError(f"Copied bytes differ for {experiment}")
        if len(list(output_dir.iterdir())) != len(EXPERIMENTS):
            raise AssertionError("Output folder must contain exactly five CSV files")
        print("SELF-TEST PASSED", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return run_self_test()
    return export_experiments(
        args.input_root.expanduser(),
        args.output_dir.expanduser(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
