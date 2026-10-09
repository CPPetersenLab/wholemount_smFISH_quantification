#!/usr/bin/env python3
"""Export all reassigned cells, including zero-spot cells, to wide CSV tables.

This script is a post-processing step for either ``reassign_smfish_3d.py`` or
``reassign_smfish_projection_for_now.py``.  It does NOT rerun Cellpose, spot
detection, or reassignment.

Why scan per-sample folders?
----------------------------
The reassignment pipelines write one authoritative ``cell_counts.csv`` and one
``sample_qc.json`` inside every completed sample directory.  When samples are
processed in several ``--sample-regex`` batches, a top-level aggregate CSV can
represent only the most recent batch.  This exporter therefore scans every
completed sample folder and rebuilds the aggregate tables from those files.

Outputs (under ``<input-root>/all_cells_including_zero_export`` by default):

* ``all_cells_long.csv``: all cell metadata and transcript counts, including 0.
* ``all_cells_wide_including_zero.csv``: every sample as one column.
* ``<experiment>_output_all_cells_including_zero.csv``: 306_output-style tables.
* ``animal_summary.csv``: summary metrics calculated from all cells.
* ``activating cell percentage.csv``: strict >0, >1, >2, >5, >10, and >100
  transcript percentages, using all segmented cells as the denominator.
* ``sample_qc.csv``: all original per-sample QC fields, retained and combined.
* ``all_cells_export_qc.csv``: independent cell/spot accounting checks.
* ``all_cells_export_manifest.json``: source and output summary.

Typical use::

    python export_all_cells_including_zero.py \
      --input-root '/Volumes/Backup Plus/Experiment_results/smfish_reassignment_cellpose_v1'

The original reassignment outputs and QC files are never deleted or modified.
The export can also be run while the first round is incomplete; it includes all
samples whose per-sample ``COMPLETE.json`` marker already exists.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


SCRIPT_VERSION = "1.1.0"
DEFAULT_INPUT_ROOT = Path(
    "/Volumes/Backup Plus/Experiment_results/smfish_reassignment_cellpose_v1"
)
CONDITION_PRIORITY = {
    "incision": 0,
    "amputation": 1,
    "agarose": 2,
}


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=True)
    os.replace(temporary, path)


def natural_key(value: object) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", str(value))
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def parse_condition(condition: str, fallback_time: object) -> tuple[str, float, str]:
    """Return treatment, numeric time, and display time such as ``6h``."""
    match = re.match(
        r"^\s*(\d+(?:\.\d+)?)\s*(?:hr|h)\s*_(.+?)\s*$",
        condition,
        flags=re.IGNORECASE,
    )
    if match:
        time_value = float(match.group(1))
        treatment = match.group(2)
    else:
        treatment = condition
        try:
            time_value = float(fallback_time)
        except (TypeError, ValueError):
            time_value = math.inf

    if math.isfinite(time_value):
        if time_value.is_integer():
            time_label = f"{int(time_value)}h"
        else:
            time_label = f"{time_value:g}h"
    else:
        time_label = "NAh"
    return treatment, time_value, time_label


def animal_label(animal: str) -> tuple[str, int]:
    match = re.search(r"(\d+)$", animal)
    if match:
        return match.group(1), int(match.group(1))
    return animal, 10**9


def sample_metadata(cell_frame: pd.DataFrame, source_path: Path) -> dict[str, Any]:
    if cell_frame.empty:
        raise ValueError(f"Empty cell_counts.csv: {source_path}")

    required = {
        "experiment",
        "condition",
        "animal",
        "sample_id",
        "transcript_count",
    }
    missing = sorted(required.difference(cell_frame.columns))
    if missing:
        raise ValueError(f"{source_path} is missing columns: {', '.join(missing)}")

    for column in ("experiment", "condition", "animal", "sample_id"):
        unique = cell_frame[column].dropna().astype(str).unique()
        if len(unique) != 1:
            raise ValueError(
                f"Expected one {column} in {source_path}, found {unique.tolist()}"
            )

    counts = pd.to_numeric(cell_frame["transcript_count"], errors="raise")
    if counts.isna().any():
        raise ValueError(f"Missing transcript_count value in {source_path}")
    if (counts < 0).any():
        raise ValueError(f"Negative transcript_count value in {source_path}")
    if not np.allclose(counts.to_numpy(), np.rint(counts.to_numpy())):
        raise ValueError(f"Non-integer transcript_count value in {source_path}")
    cell_frame["transcript_count"] = np.rint(counts).astype(np.int64)

    first = cell_frame.iloc[0]
    condition = str(first["condition"])
    treatment, time_value, time_label = parse_condition(
        condition, first.get("time_hours", math.nan)
    )
    animal = str(first["animal"])
    animal_display, animal_number = animal_label(animal)
    experiment = str(first["experiment"])
    if experiment.endswith(".0") and experiment[:-2].isdigit():
        experiment = experiment[:-2]

    return {
        "experiment": experiment,
        "condition": condition,
        "treatment": treatment,
        "time_value": time_value,
        "time_label": time_label,
        "animal": animal,
        "animal_display": animal_display,
        "animal_number": animal_number,
        "sample_id": str(first["sample_id"]),
        "output_header": f"{treatment}_{time_label}_{animal_display}",
        "source_path": source_path,
    }


def sample_sort_key(metadata: dict[str, Any]) -> tuple[Any, ...]:
    treatment = str(metadata["treatment"])
    return (
        natural_key(metadata["experiment"]),
        metadata["time_value"],
        CONDITION_PRIORITY.get(treatment.lower(), 99),
        treatment.lower(),
        metadata["animal_number"],
        natural_key(metadata["animal"]),
    )


def summarize_animal(cell_frame: pd.DataFrame) -> dict[str, Any]:
    first = cell_frame.iloc[0]
    counts = cell_frame["transcript_count"].to_numpy(dtype=np.int64)
    positive = counts[counts > 0]
    total_cells = len(counts)
    return {
        "experiment": first.get("experiment"),
        "gene": first.get("gene"),
        "condition": first.get("condition"),
        "time_hours": first.get("time_hours"),
        "injury": first.get("injury"),
        "animal": first.get("animal"),
        "sample_id": first.get("sample_id"),
        "total_cells": int(total_cells),
        "zero_spot_cells": int(np.sum(counts == 0)),
        "responding_cells": int(len(positive)),
        "responding_fraction": float(len(positive) / total_cells),
        "total_assigned_transcripts": int(np.sum(counts)),
        "mean_count_all_cells": float(np.mean(counts)),
        "mean_log1p_all_cells": float(np.log1p(counts).mean()),
        "mean_count_responding_cells": (
            float(np.mean(positive)) if len(positive) else math.nan
        ),
        "median_count_responding_cells": (
            float(np.median(positive)) if len(positive) else math.nan
        ),
        "mean_log1p_responding_cells": (
            float(np.log1p(positive).mean()) if len(positive) else math.nan
        ),
        "fraction_responding_cells_ge_5": (
            float(np.mean(positive >= 5)) if len(positive) else math.nan
        ),
        "maximum_count": int(np.max(positive)) if len(positive) else 0,
    }


def activating_cell_percentage(cell_frame: pd.DataFrame) -> dict[str, Any]:
    """Return strict transcript-count threshold percentages for one sample."""
    first = cell_frame.iloc[0]
    counts = cell_frame["transcript_count"].to_numpy(dtype=np.int64)
    row: dict[str, Any] = {
        "experiment": first.get("experiment"),
        "gene": first.get("gene"),
        "condition": first.get("condition"),
        "time_hours": first.get("time_hours"),
        "injury": first.get("injury"),
        "animal": first.get("animal"),
        "sample_id": first.get("sample_id"),
        "total_cells": int(len(counts)),
    }
    for threshold in (0, 1, 2, 5, 10, 100):
        row[f"percent_cells_gt_{threshold}"] = float(
            100.0 * np.mean(counts > threshold)
        )
    return row


def read_sample_qc(sample_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    qc_path = sample_dir / "sample_qc.json"
    if qc_path.is_file():
        with qc_path.open(encoding="utf-8") as stream:
            qc = json.load(stream)
        if not isinstance(qc, dict):
            raise ValueError(f"QC JSON is not an object: {qc_path}")
    else:
        qc = {}

    # These identifiers guarantee that a missing/older QC field never makes a
    # sample disappear from the combined QC table.
    qc.setdefault("experiment", metadata["experiment"])
    qc.setdefault("condition", metadata["condition"])
    qc.setdefault("animal", metadata["animal"])
    qc.setdefault("sample_id", metadata["sample_id"])
    qc["sample_qc_json_present"] = qc_path.is_file()
    qc["sample_qc_json_path"] = str(qc_path)
    return qc


def qc_expected_cell_count(qc: dict[str, Any]) -> int | None:
    for field in ("total_3d_cells", "total_linked_cells", "total_cells"):
        value = qc.get(field)
        if value is not None and not pd.isna(value):
            return int(value)
    return None


def build_export_qc(
    cell_frame: pd.DataFrame,
    metadata: dict[str, Any],
    qc: dict[str, Any],
    complete_marker_present: bool,
) -> dict[str, Any]:
    counts = cell_frame["transcript_count"].to_numpy(dtype=np.int64)
    total_cells = int(len(counts))
    zero_cells = int(np.sum(counts == 0))
    positive_cells = int(np.sum(counts > 0))
    transcript_sum = int(np.sum(counts))
    expected_cells = qc_expected_cell_count(qc)
    assigned_spots = qc.get("assigned_spots")
    assigned_spots_int = (
        int(assigned_spots)
        if assigned_spots is not None and not pd.isna(assigned_spots)
        else None
    )

    cell_count_matches = (
        total_cells == expected_cells if expected_cells is not None else None
    )
    transcript_sum_matches = (
        transcript_sum == assigned_spots_int if assigned_spots_int is not None else None
    )
    spot_accounting_matches = qc.get("spot_accounting_matches")
    checks = [
        zero_cells + positive_cells == total_cells,
        cell_count_matches,
        transcript_sum_matches,
        spot_accounting_matches,
    ]
    export_qc_pass = all(check is not False for check in checks)

    return {
        "experiment": metadata["experiment"],
        "condition": metadata["condition"],
        "animal": metadata["animal"],
        "sample_id": metadata["sample_id"],
        "output_header": metadata["output_header"],
        "complete_marker_present": complete_marker_present,
        "sample_qc_json_present": bool(qc.get("sample_qc_json_present")),
        "total_cells_exported": total_cells,
        "zero_spot_cells_exported": zero_cells,
        "positive_cells_exported": positive_cells,
        "zero_plus_positive_matches_total": zero_cells + positive_cells == total_cells,
        "total_transcripts_from_cells": transcript_sum,
        "qc_expected_cells": expected_cells,
        "cell_count_matches_qc": cell_count_matches,
        "qc_assigned_spots": assigned_spots_int,
        "transcript_sum_matches_assigned_spots": transcript_sum_matches,
        "spot_accounting_matches": spot_accounting_matches,
        "cellpose_qc_matches_label_files": qc.get(
            "cellpose_qc_matches_label_files"
        ),
        "export_qc_pass": export_qc_pass,
        "source_cell_counts_csv": str(metadata["source_path"]),
        "source_sample_qc_json": qc.get("sample_qc_json_path"),
    }


def discover_samples(
    input_root: Path,
    include_incomplete: bool,
    sample_pattern: re.Pattern[str] | None,
) -> list[tuple[pd.DataFrame, dict[str, Any], dict[str, Any], bool]]:
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    discovered: list[tuple[pd.DataFrame, dict[str, Any], dict[str, Any], bool]] = []
    seen_sample_ids: dict[str, Path] = {}
    for cell_path in sorted(input_root.rglob("cell_counts.csv"), key=natural_key):
        sample_dir = cell_path.parent
        complete_present = (sample_dir / "COMPLETE.json").is_file()
        if not complete_present and not include_incomplete:
            continue

        frame = pd.read_csv(cell_path)
        metadata = sample_metadata(frame, cell_path)
        sample_id = metadata["sample_id"]
        if sample_pattern and not sample_pattern.search(sample_id):
            continue
        if sample_id in seen_sample_ids:
            raise ValueError(
                f"Duplicate sample_id {sample_id}: {seen_sample_ids[sample_id]} and {cell_path}"
            )
        seen_sample_ids[sample_id] = cell_path

        # Cell ID order is made deterministic without changing any count.
        if "global_cell_id" in frame.columns:
            frame = frame.sort_values("global_cell_id", kind="stable").reset_index(drop=True)
        else:
            frame = frame.reset_index(drop=True)
        qc = read_sample_qc(sample_dir, metadata)
        discovered.append((frame, metadata, qc, complete_present))

    discovered.sort(key=lambda item: sample_sort_key(item[1]))
    if not discovered:
        qualifier = " (including incomplete samples)" if include_incomplete else ""
        raise RuntimeError(
            f"No completed cell_counts.csv files found under {input_root}{qualifier}"
        )
    return discovered


def wide_table(
    samples: Sequence[tuple[pd.DataFrame, dict[str, Any], dict[str, Any], bool]],
    header_field: str,
) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    for frame, metadata, _, _ in samples:
        header = str(metadata[header_field])
        if header in columns:
            raise ValueError(f"Duplicate output column name: {header}")
        columns[header] = pd.Series(
            frame["transcript_count"].to_numpy(dtype=np.int64), dtype="Int64"
        )
    return pd.DataFrame(columns)


def export_all_cells(
    input_root: Path,
    export_dir: Path,
    include_incomplete: bool = False,
    sample_regex: str | None = None,
) -> dict[str, Any]:
    pattern = re.compile(sample_regex) if sample_regex else None
    samples = discover_samples(input_root, include_incomplete, pattern)
    export_dir.mkdir(parents=True, exist_ok=True)

    all_cells = pd.concat([item[0] for item in samples], ignore_index=True)
    atomic_write_csv(all_cells, export_dir / "all_cells_long.csv")

    combined_wide = wide_table(samples, "sample_id")
    atomic_write_csv(
        combined_wide, export_dir / "all_cells_wide_including_zero.csv"
    )

    experiments: dict[str, list[tuple[pd.DataFrame, dict[str, Any], dict[str, Any], bool]]] = {}
    for item in samples:
        experiments.setdefault(str(item[1]["experiment"]), []).append(item)

    experiment_outputs: dict[str, str] = {}
    for experiment in sorted(experiments, key=natural_key):
        experiment_wide = wide_table(experiments[experiment], "output_header")
        output_path = (
            export_dir / f"{experiment}_output_all_cells_including_zero.csv"
        )
        atomic_write_csv(experiment_wide, output_path)
        experiment_outputs[experiment] = str(output_path)

    animal_rows = [summarize_animal(item[0]) for item in samples]
    activation_rows = [activating_cell_percentage(item[0]) for item in samples]
    qc_rows = [item[2] for item in samples]
    export_qc_rows = [
        build_export_qc(item[0], item[1], item[2], item[3]) for item in samples
    ]
    animal_summary = pd.DataFrame(animal_rows)
    sample_qc = pd.DataFrame(qc_rows)
    export_qc = pd.DataFrame(export_qc_rows)
    atomic_write_csv(animal_summary, export_dir / "animal_summary.csv")
    atomic_write_csv(
        pd.DataFrame(activation_rows),
        export_dir / "activating cell percentage.csv",
    )
    atomic_write_csv(sample_qc, export_dir / "sample_qc.csv")
    atomic_write_csv(export_qc, export_dir / "all_cells_export_qc.csv")

    failed_qc = export_qc.loc[~export_qc["export_qc_pass"].astype(bool), "sample_id"]
    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(input_root),
        "export_dir": str(export_dir),
        "include_incomplete": include_incomplete,
        "sample_regex": sample_regex,
        "samples_exported": len(samples),
        "experiments_exported": sorted(experiments, key=natural_key),
        "total_cells_exported": int(len(all_cells)),
        "zero_spot_cells_exported": int(
            (all_cells["transcript_count"] == 0).sum()
        ),
        "positive_cells_exported": int(
            (all_cells["transcript_count"] > 0).sum()
        ),
        "all_samples_pass_export_qc": len(failed_qc) == 0,
        "failed_export_qc_sample_ids": failed_qc.astype(str).tolist(),
        "experiment_output_files": experiment_outputs,
        "original_qc_files_modified": False,
    }
    atomic_write_json(manifest, export_dir / "all_cells_export_manifest.json")
    return manifest


def run_self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="all_cells_zero_test_") as temporary:
        root = Path(temporary) / "reassignment"
        test_samples = [
            (
                "306",
                "0hr_Incision",
                "Image1",
                [0, 2, 0, 1],
                "wnt1",
                "incision",
                0.0,
            ),
            (
                "307",
                "24h_Agarose",
                "Image1",
                [0, 0, 5],
                "wnt1",
                "agarose",
                24.0,
            ),
        ]
        for experiment, condition, animal, counts, gene, injury, time_hours in test_samples:
            sample_dir = root / experiment / condition / animal
            sample_dir.mkdir(parents=True, exist_ok=True)
            sample_id = f"{experiment}__{condition}__{animal}"
            frame = pd.DataFrame(
                {
                    "experiment": experiment,
                    "gene": gene,
                    "condition": condition,
                    "time_hours": time_hours,
                    "injury": injury,
                    "animal": animal,
                    "sample_id": sample_id,
                    "global_cell_id": np.arange(1, len(counts) + 1),
                    "transcript_count": counts,
                    "responding": np.asarray(counts) > 0,
                }
            )
            frame.to_csv(sample_dir / "cell_counts.csv", index=False)
            with (sample_dir / "sample_qc.json").open("w", encoding="utf-8") as stream:
                json.dump(
                    {
                        "experiment": experiment,
                        "condition": condition,
                        "animal": animal,
                        "sample_id": sample_id,
                        "assigned_spots": int(sum(counts)),
                        "total_3d_cells": len(counts),
                        "spot_accounting_matches": True,
                        "cellpose_qc_matches_label_files": True,
                    },
                    stream,
                )
            (sample_dir / "COMPLETE.json").write_text("{}\n", encoding="utf-8")

        export_dir = root / "all_cells_including_zero_export"
        manifest = export_all_cells(root, export_dir)
        output_306 = pd.read_csv(
            export_dir / "306_output_all_cells_including_zero.csv"
        )
        assert output_306.columns.tolist() == ["Incision_0h_1"]
        assert output_306["Incision_0h_1"].tolist() == [0, 2, 0, 1]
        output_307 = pd.read_csv(
            export_dir / "307_output_all_cells_including_zero.csv"
        )
        assert output_307.columns.tolist() == ["Agarose_24h_1"]
        assert output_307["Agarose_24h_1"].tolist() == [0, 0, 5]
        qc = pd.read_csv(export_dir / "all_cells_export_qc.csv")
        assert qc["export_qc_pass"].all()
        activation = pd.read_csv(export_dir / "activating cell percentage.csv")
        activation_306 = activation.loc[
            activation["sample_id"] == "306__0hr_Incision__Image1"
        ].iloc[0]
        expected_306 = np.asarray([0, 2, 0, 1], dtype=np.int64)
        for threshold in (0, 1, 2, 5, 10, 100):
            column = f"percent_cells_gt_{threshold}"
            expected_percent = float(100.0 * np.mean(expected_306 > threshold))
            assert np.isclose(float(activation_306[column]), expected_percent)
        assert manifest["samples_exported"] == 2
        assert manifest["total_cells_exported"] == 7
        assert manifest["zero_spot_cells_exported"] == 4
        assert manifest["all_samples_pass_export_qc"] is True

    print("Self-test passed: zero cells, column names, per-sample QC, and totals agree.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=(
            "Reassignment output root containing experiment/condition/animal/"
            "cell_counts.csv directories."
        ),
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        help=(
            "Destination directory. Default: INPUT_ROOT/"
            "all_cells_including_zero_export"
        ),
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Also include sample folders without COMPLETE.json (not recommended).",
    )
    parser.add_argument(
        "--sample-regex",
        help="Optional regular expression applied to sample_id after discovery.",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return run_self_test()

    input_root = args.input_root.expanduser().resolve()
    export_dir = (
        args.export_dir.expanduser().resolve()
        if args.export_dir
        else input_root / "all_cells_including_zero_export"
    )
    manifest = export_all_cells(
        input_root=input_root,
        export_dir=export_dir,
        include_incomplete=args.include_incomplete,
        sample_regex=args.sample_regex,
    )
    print(
        f"Export complete: {manifest['samples_exported']} samples, "
        f"{manifest['total_cells_exported']} total cells, "
        f"{manifest['zero_spot_cells_exported']} zero-spot cells."
    )
    print(f"All samples pass export QC: {manifest['all_samples_pass_export_qc']}")
    print(f"Output: {manifest['export_dir']}")
    return 0 if manifest["all_samples_pass_export_qc"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
