#!/usr/bin/env python3
"""Compare linked and unlinked legacy-label smFISH quantification.

Primary analysis
----------------
For every animal/image, calculate the mean transcript count among cells with
transcript_count > 0.  These animal-level means, not individual cells, are the
replicates in two-sided Welch t-tests comparing amputation with incision.

The default analysis includes:
  * wnt1: experiments 304 and 306
  * notum: experiments 314 and 332
  * 6 h and 12 h
  * reciprocal-linking and no-linking output roots

Expected input in each mode root:
  all_cells_long.csv

This file is produced by reassign_smfish_old_labels_modes.py.

Outputs include sample-level summaries, t-test results, a mode-consistency
table, paired linked/unlinked sample means, and publication-ready PNG/PDF plots.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_LINKING_ROOT = Path(
    "/Volumes/Backup Plus/Experiment_results/smfish_old_labels_reciprocal_final"
)
DEFAULT_NO_LINKING_ROOT = Path(
    "/Volumes/Backup Plus/Experiment_results/smfish_old_labels_no_link_final"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/Volumes/Backup Plus/Experiment_results/"
    "smfish_linking_sensitivity_nonzero_sample_mean"
)

EXPERIMENT_TO_GENE = {
    "304": "wnt1",
    "306": "wnt1",
    "314": "notum",
    "332": "notum",
}
GENE_EXPERIMENTS = {
    "wnt1": ("304", "306"),
    "notum": ("314", "332"),
}
TIMES = (6.0, 12.0)
INJURIES = ("incision", "amputation")
MODES = ("linking", "no_linking")

INJURY_COLORS = {
    "incision": "#B7B7B7",
    "amputation": "#6C5CE7",
}
EXPERIMENT_MARKERS = {
    "304": "o",
    "306": "s",
    "314": "o",
    "332": "s",
}

REQUIRED_COLUMNS = {
    "experiment",
    "gene",
    "condition",
    "time_hours",
    "injury",
    "animal",
    "sample_id",
    "transcript_count",
}


@dataclass(frozen=True)
class AnalysisConfig:
    linking_root: Path
    no_linking_root: Path
    output_root: Path
    expected_samples_per_experiment: int
    allow_incomplete: bool


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use non-zero-cell animal means to compare amputation vs incision "
            "for linked and unlinked smFISH assignments."
        )
    )
    parser.add_argument(
        "--linking-root",
        type=Path,
        default=DEFAULT_LINKING_ROOT,
        help="Reciprocal-linking output root containing all_cells_long.csv.",
    )
    parser.add_argument(
        "--no-linking-root",
        type=Path,
        default=DEFAULT_NO_LINKING_ROOT,
        help="No-linking output root containing all_cells_long.csv.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="New directory for tables and figures.",
    )
    parser.add_argument(
        "--expected-samples-per-experiment",
        type=int,
        default=3,
        help=(
            "Expected independent Image/animal samples per experiment, time, "
            "and injury group (default: 3)."
        ),
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Continue with missing groups or unexpected sample counts.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run an internal synthetic-data test instead of reading real data.",
    )
    return parser.parse_args(argv)


def normalize_experiment(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def normalize_injury(value: object) -> str:
    text = str(value).strip().lower()
    if "amput" in text:
        return "amputation"
    if "incis" in text:
        return "incision"
    return text


def read_mode(mode: str, root: Path) -> pd.DataFrame:
    path = root / "all_cells_long.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"{mode}: expected input not found: {path}\n"
            "Run reassign_smfish_old_labels_modes.py for this mode first."
        )

    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing)}"
        )

    frame = frame.copy()
    frame["mode"] = mode
    frame["experiment"] = frame["experiment"].map(normalize_experiment)
    frame["gene"] = frame["gene"].astype(str).str.strip().str.lower()
    frame["injury"] = frame["injury"].map(normalize_injury)
    frame["time_hours"] = pd.to_numeric(frame["time_hours"], errors="coerce")
    frame["transcript_count"] = pd.to_numeric(
        frame["transcript_count"], errors="coerce"
    )

    if frame["transcript_count"].isna().any():
        bad = int(frame["transcript_count"].isna().sum())
        raise ValueError(f"{path} contains {bad} non-numeric transcript counts.")
    if (frame["transcript_count"] < 0).any():
        raise ValueError(f"{path} contains negative transcript counts.")

    expected_backend = {
        "linking": "old_projection_labels_reciprocal_linking",
        "no_linking": "old_projection_labels_no_link",
    }[mode]
    config_path = root / "run_configuration.json"
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as handle:
            run_config = json.load(handle)
        actual_backend = run_config.get("assignment_backend")
        if actual_backend and actual_backend != expected_backend:
            raise ValueError(
                f"{mode}: {config_path} reports assignment_backend="
                f"{actual_backend!r}; expected {expected_backend!r}."
            )

    return frame


def select_analysis_cells(frame: pd.DataFrame) -> pd.DataFrame:
    expected_experiments = set(EXPERIMENT_TO_GENE)
    selected = frame.loc[
        frame["experiment"].isin(expected_experiments)
        & frame["time_hours"].isin(TIMES)
        & frame["injury"].isin(INJURIES)
    ].copy()

    if selected.empty:
        raise ValueError("No requested 304/306/314/332 6 h or 12 h cells found.")

    expected_gene = selected["experiment"].map(EXPERIMENT_TO_GENE)
    mismatched = selected.loc[expected_gene.ne(selected["gene"])]
    if not mismatched.empty:
        examples = mismatched[["experiment", "gene", "sample_id"]].drop_duplicates()
        raise ValueError(
            "Experiment/gene mismatch detected:\n" + examples.head(10).to_string(index=False)
        )
    return selected


def summarize_samples(cells: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "mode",
        "experiment",
        "gene",
        "condition",
        "time_hours",
        "injury",
        "animal",
        "sample_id",
    ]

    rows: list[dict[str, object]] = []
    for values, group in cells.groupby(keys, sort=True, dropna=False):
        row = dict(zip(keys, values))
        counts = group["transcript_count"].to_numpy(dtype=float)
        positive = counts[counts > 0]
        if positive.size == 0:
            mean_positive = math.nan
            median_positive = math.nan
            sd_positive = math.nan
            sem_positive = math.nan
        else:
            mean_positive = float(np.mean(positive))
            median_positive = float(np.median(positive))
            sd_positive = (
                float(np.std(positive, ddof=1)) if positive.size > 1 else math.nan
            )
            sem_positive = (
                sd_positive / math.sqrt(positive.size)
                if positive.size > 1
                else math.nan
            )

        row.update(
            {
                "total_segmented_cells": int(counts.size),
                "nonzero_cells": int(positive.size),
                "fraction_nonzero_cells": (
                    float(positive.size / counts.size) if counts.size else math.nan
                ),
                "mean_transcripts_nonzero_cells": mean_positive,
                "median_transcripts_nonzero_cells": median_positive,
                "sd_transcripts_nonzero_cells": sd_positive,
                "sem_transcripts_nonzero_cells": sem_positive,
                "total_assigned_transcripts": int(np.sum(counts)),
            }
        )
        rows.append(row)

    result = pd.DataFrame(rows)
    if result["mean_transcripts_nonzero_cells"].isna().any():
        bad = result.loc[
            result["mean_transcripts_nonzero_cells"].isna(),
            ["mode", "sample_id"],
        ]
        raise ValueError(
            "At least one requested sample has no non-zero cells:\n"
            + bad.to_string(index=False)
        )
    return result.sort_values(
        ["mode", "gene", "time_hours", "injury", "experiment", "animal"]
    ).reset_index(drop=True)


def validate_sample_design(
    sample_means: pd.DataFrame,
    expected_per_experiment: int,
    allow_incomplete: bool,
) -> pd.DataFrame:
    counts = (
        sample_means.groupby(
            ["mode", "gene", "time_hours", "injury", "experiment"],
            sort=True,
        )["sample_id"]
        .nunique()
        .rename("observed_samples")
        .reset_index()
    )

    expected_rows = []
    for mode in MODES:
        for gene, experiments in GENE_EXPERIMENTS.items():
            for time in TIMES:
                for injury in INJURIES:
                    for experiment in experiments:
                        expected_rows.append(
                            {
                                "mode": mode,
                                "gene": gene,
                                "time_hours": time,
                                "injury": injury,
                                "experiment": experiment,
                                "expected_samples": expected_per_experiment,
                            }
                        )
    design = pd.DataFrame(expected_rows).merge(
        counts,
        on=["mode", "gene", "time_hours", "injury", "experiment"],
        how="left",
    )
    design["observed_samples"] = design["observed_samples"].fillna(0).astype(int)
    design["sample_count_matches"] = (
        design["observed_samples"] == design["expected_samples"]
    )

    problems = design.loc[~design["sample_count_matches"]]
    if not problems.empty:
        message = (
            "Unexpected or missing sample counts:\n"
            + problems.to_string(index=False)
        )
        if allow_incomplete:
            print("WARNING:", message, file=sys.stderr)
        else:
            raise ValueError(message + "\nUse --allow-incomplete only if intentional.")
    return design


def welch_df(a: np.ndarray, b: np.ndarray) -> float:
    va = float(np.var(a, ddof=1))
    vb = float(np.var(b, ddof=1))
    na = a.size
    nb = b.size
    numerator = (va / na + vb / nb) ** 2
    denominator = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    return numerator / denominator if denominator > 0 else math.nan


def hedges_g(amputation: np.ndarray, incision: np.ndarray) -> float:
    n_amp = amputation.size
    n_inc = incision.size
    df = n_amp + n_inc - 2
    if df <= 0:
        return math.nan
    pooled_var = (
        (n_amp - 1) * np.var(amputation, ddof=1)
        + (n_inc - 1) * np.var(incision, ddof=1)
    ) / df
    if pooled_var <= 0:
        return math.nan
    d = (np.mean(amputation) - np.mean(incision)) / math.sqrt(pooled_var)
    correction = 1.0 - 3.0 / (4.0 * df - 1.0) if df > 1 else 1.0
    return float(correction * d)


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    order = np.argsort(p)
    adjusted_sorted = np.empty_like(p)
    running = 0.0
    m = p.size
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * p[index])
        running = max(running, value)
        adjusted_sorted[rank] = running
    adjusted = np.empty_like(p)
    adjusted[order] = adjusted_sorted
    return adjusted


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    order = np.argsort(p)
    sorted_p = p[order]
    m = p.size
    adjusted_sorted = sorted_p * m / np.arange(1, m + 1)
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    adjusted = np.empty_like(p)
    adjusted[order] = adjusted_sorted
    return adjusted


def calculate_ttests(sample_means: pd.DataFrame) -> pd.DataFrame:
    metric = "mean_transcripts_nonzero_cells"
    rows: list[dict[str, object]] = []
    for (mode, gene, time), group in sample_means.groupby(
        ["mode", "gene", "time_hours"], sort=True
    ):
        amp = group.loc[group["injury"] == "amputation", metric].to_numpy(float)
        inc = group.loc[group["injury"] == "incision", metric].to_numpy(float)
        if amp.size < 2 or inc.size < 2:
            raise ValueError(
                f"Need at least two sample means per group: {mode}, {gene}, {time} h."
            )

        test = stats.ttest_ind(amp, inc, equal_var=False, alternative="two-sided")
        difference = float(np.mean(amp) - np.mean(inc))
        se_difference = float(
            math.sqrt(np.var(amp, ddof=1) / amp.size + np.var(inc, ddof=1) / inc.size)
        )
        degrees_freedom = welch_df(amp, inc)
        critical = float(stats.t.ppf(0.975, degrees_freedom))
        ci_low = difference - critical * se_difference
        ci_high = difference + critical * se_difference
        mean_inc = float(np.mean(inc))
        ratio = float(np.mean(amp) / mean_inc) if mean_inc != 0 else math.nan

        rows.append(
            {
                "mode": mode,
                "gene": gene,
                "time_hours": float(time),
                "comparison": "amputation_vs_incision",
                "statistical_unit": "sample_mean_of_nonzero_cells",
                "test": "two_sided_welch_t_test",
                "n_amputation_samples": int(amp.size),
                "n_incision_samples": int(inc.size),
                "mean_amputation": float(np.mean(amp)),
                "sem_amputation": float(stats.sem(amp)),
                "mean_incision": mean_inc,
                "sem_incision": float(stats.sem(inc)),
                "mean_difference_amputation_minus_incision": difference,
                "difference_ci95_low": float(ci_low),
                "difference_ci95_high": float(ci_high),
                "fold_difference_amputation_over_incision": ratio,
                "hedges_g": hedges_g(amp, inc),
                "t_statistic": float(test.statistic),
                "degrees_freedom": float(degrees_freedom),
                "p_value": float(test.pvalue),
            }
        )

    result = pd.DataFrame(rows).sort_values(
        ["mode", "gene", "time_hours"]
    ).reset_index(drop=True)
    result["p_holm_within_mode"] = np.nan
    result["q_bh_within_mode"] = np.nan
    for mode, indices in result.groupby("mode").groups.items():
        idx = list(indices)
        values = result.loc[idx, "p_value"].to_numpy(float)
        result.loc[idx, "p_holm_within_mode"] = holm_adjust(values)
        result.loc[idx, "q_bh_within_mode"] = benjamini_hochberg(values)
    result["raw_p_lt_0_05"] = result["p_value"] < 0.05
    result["holm_p_lt_0_05"] = result["p_holm_within_mode"] < 0.05
    result["effect_direction"] = np.where(
        result["mean_difference_amputation_minus_incision"] > 0,
        "amputation_higher",
        np.where(
            result["mean_difference_amputation_minus_incision"] < 0,
            "incision_higher",
            "equal",
        ),
    )
    return result


def calculate_group_summary(sample_means: pd.DataFrame) -> pd.DataFrame:
    metric = "mean_transcripts_nonzero_cells"
    return (
        sample_means.groupby(["mode", "gene", "time_hours", "injury"], sort=True)
        .agg(
            n_samples=("sample_id", "nunique"),
            mean_of_sample_means=(metric, "mean"),
            sd_of_sample_means=(metric, "std"),
            sem_of_sample_means=(metric, "sem"),
        )
        .reset_index()
    )


def calculate_paired_mode_table(sample_means: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "experiment",
        "gene",
        "condition",
        "time_hours",
        "injury",
        "animal",
        "sample_id",
    ]
    metric = "mean_transcripts_nonzero_cells"
    wide = sample_means.pivot(index=keys, columns="mode", values=metric).reset_index()
    wide.columns.name = None
    if set(MODES).difference(wide.columns):
        raise ValueError("Could not pair all linking and no-linking sample means.")
    if wide[list(MODES)].isna().any().any():
        missing = wide.loc[wide[list(MODES)].isna().any(axis=1)]
        raise ValueError(
            "Samples are not matched between modes:\n" + missing.head(20).to_string(index=False)
        )
    wide["linking_minus_no_linking"] = wide["linking"] - wide["no_linking"]
    wide["linking_over_no_linking"] = wide["linking"] / wide["no_linking"]
    return wide.sort_values(
        ["gene", "time_hours", "injury", "experiment", "animal"]
    ).reset_index(drop=True)


def calculate_mode_consistency(ttests: pd.DataFrame) -> pd.DataFrame:
    value_columns = [
        "mean_amputation",
        "mean_incision",
        "mean_difference_amputation_minus_incision",
        "difference_ci95_low",
        "difference_ci95_high",
        "fold_difference_amputation_over_incision",
        "hedges_g",
        "p_value",
        "p_holm_within_mode",
        "q_bh_within_mode",
        "effect_direction",
        "raw_p_lt_0_05",
        "holm_p_lt_0_05",
    ]
    wide = ttests.pivot(
        index=["gene", "time_hours"], columns="mode", values=value_columns
    )
    wide.columns = [f"{metric}__{mode}" for metric, mode in wide.columns]
    wide = wide.reset_index()
    wide["same_effect_direction"] = (
        wide["effect_direction__linking"]
        == wide["effect_direction__no_linking"]
    )
    wide["same_raw_significance"] = (
        wide["raw_p_lt_0_05__linking"]
        == wide["raw_p_lt_0_05__no_linking"]
    )
    wide["same_holm_significance"] = (
        wide["holm_p_lt_0_05__linking"]
        == wide["holm_p_lt_0_05__no_linking"]
    )
    wide["same_direction_and_raw_conclusion"] = (
        wide["same_effect_direction"] & wide["same_raw_significance"]
    )
    return wide.sort_values(["gene", "time_hours"]).reset_index(drop=True)


def format_p_value(p: float) -> str:
    if p < 0.0001:
        return "p < 0.0001"
    if p < 0.001:
        return f"p = {p:.4f}"
    return f"p = {p:.3f}"


def add_p_bracket(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y: float,
    height: float,
    p_value: float,
) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], color="black", lw=0.9)
    ax.text(
        (x1 + x2) / 2,
        y + height * 1.25,
        format_p_value(p_value),
        ha="center",
        va="bottom",
        fontsize=8,
    )


def plot_mode_bars(
    sample_means: pd.DataFrame,
    ttests: pd.DataFrame,
    mode: str,
    output_root: Path,
) -> None:
    metric = "mean_transcripts_nonzero_cells"
    genes = ("wnt1", "notum")
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.8))
    rng = np.random.default_rng(20260827)

    for ax, gene in zip(axes, genes):
        gene_data = sample_means.loc[
            (sample_means["mode"] == mode) & (sample_means["gene"] == gene)
        ]
        positions = {
            (6.0, "incision"): 0.0,
            (6.0, "amputation"): 1.0,
            (12.0, "incision"): 2.5,
            (12.0, "amputation"): 3.5,
        }

        maxima: list[float] = []
        for time in TIMES:
            for injury in INJURIES:
                subset = gene_data.loc[
                    (gene_data["time_hours"] == time)
                    & (gene_data["injury"] == injury)
                ]
                values = subset[metric].to_numpy(float)
                x = positions[(time, injury)]
                mean = float(np.mean(values))
                sem = float(stats.sem(values))
                ax.bar(
                    x,
                    mean,
                    width=0.72,
                    color=INJURY_COLORS[injury],
                    edgecolor="black",
                    linewidth=0.8,
                    zorder=1,
                )
                ax.errorbar(
                    x,
                    mean,
                    yerr=sem,
                    color="black",
                    capsize=3,
                    lw=1.0,
                    zorder=3,
                )
                for experiment, exp_data in subset.groupby("experiment", sort=True):
                    jitter = rng.uniform(-0.12, 0.12, size=len(exp_data))
                    ax.scatter(
                        np.full(len(exp_data), x) + jitter,
                        exp_data[metric],
                        marker=EXPERIMENT_MARKERS.get(experiment, "o"),
                        s=34,
                        facecolor=INJURY_COLORS[injury],
                        edgecolor="black",
                        linewidth=0.7,
                        zorder=4,
                    )
                maxima.append(float(np.max(values)))

        base_max = max(maxima) if maxima else 1.0
        bracket_step = max(0.12 * base_max, 0.18)
        for bracket_index, time in enumerate(TIMES):
            pair_data = gene_data.loc[gene_data["time_hours"] == time]
            local_max = float(pair_data[metric].max())
            p_value = float(
                ttests.loc[
                    (ttests["mode"] == mode)
                    & (ttests["gene"] == gene)
                    & (ttests["time_hours"] == time),
                    "p_value",
                ].iloc[0]
            )
            add_p_bracket(
                ax,
                positions[(time, "incision")],
                positions[(time, "amputation")],
                local_max + bracket_step * 0.35,
                bracket_step * 0.20,
                p_value,
            )

        ax.set_title(gene, fontstyle="italic", fontsize=12)
        ax.set_xticks([0.0, 1.0, 2.5, 3.5])
        ax.set_xticklabels(
            ["6 h\nIncision", "6 h\nAmputation", "12 h\nIncision", "12 h\nAmputation"],
            fontsize=8.5,
        )
        ax.set_xlim(-0.65, 4.15)
        ax.set_ylim(0, base_max + bracket_step * 1.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", width=0.8)
        ax.grid(axis="y", color="#EAEAEA", linewidth=0.6, zorder=0)

    axes[0].set_ylabel(
        "Mean transcripts per responding cell\n(non-zero cells; sample mean ± SEM)",
        fontsize=9.5,
    )
    mode_title = "Reciprocal projection linking" if mode == "linking" else "No projection linking"
    fig.suptitle(mode_title, fontsize=12.5)

    experiment_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=6,
            label="304 (wnt1) / 314 (notum)",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=6,
            label="306 (wnt1) / 332 (notum)",
        ),
    ]
    fig.legend(
        handles=experiment_handles,
        frameon=False,
        fontsize=7.5,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
    )
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.20, top=0.82, wspace=0.16)

    stem = f"barplot_nonzero_sample_means_{mode}"
    fig.savefig(output_root / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_root / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_effect_comparison(ttests: pd.DataFrame, output_root: Path) -> None:
    rows = []
    for gene in ("wnt1", "notum"):
        for time in TIMES:
            rows.append((gene, time, f"{gene} {int(time)} h"))

    fig, ax = plt.subplots(figsize=(6.7, 4.2), constrained_layout=True)
    y_base = np.arange(len(rows), dtype=float)[::-1]
    offsets = {"linking": 0.12, "no_linking": -0.12}
    colors = {"linking": "#6C5CE7", "no_linking": "#606060"}
    markers = {"linking": "o", "no_linking": "s"}

    for mode in MODES:
        x_values = []
        lower_errors = []
        upper_errors = []
        y_values = []
        for base, (gene, time, _) in zip(y_base, rows):
            row = ttests.loc[
                (ttests["mode"] == mode)
                & (ttests["gene"] == gene)
                & (ttests["time_hours"] == time)
            ].iloc[0]
            effect = float(row["mean_difference_amputation_minus_incision"])
            low = float(row["difference_ci95_low"])
            high = float(row["difference_ci95_high"])
            x_values.append(effect)
            lower_errors.append(effect - low)
            upper_errors.append(high - effect)
            y_values.append(base + offsets[mode])
        ax.errorbar(
            x_values,
            y_values,
            xerr=[lower_errors, upper_errors],
            fmt=markers[mode],
            color=colors[mode],
            ecolor=colors[mode],
            elinewidth=1.2,
            capsize=3,
            markersize=5.5,
            label="Linking" if mode == "linking" else "No linking",
        )

    ax.axvline(0, color="black", linewidth=0.9, linestyle="--")
    ax.set_yticks(y_base)
    ax.set_yticklabels([label for _, _, label in rows])
    ax.set_xlabel("Mean difference: amputation − incision\n(sample mean of non-zero cells; 95% CI)")
    ax.set_title("Sensitivity of biological effect to projection linking")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#EAEAEA", linewidth=0.6)
    ax.legend(frameon=False, loc="best")
    fig.savefig(
        output_root / "effect_size_linking_vs_no_linking.png",
        dpi=600,
        bbox_inches="tight",
    )
    fig.savefig(
        output_root / "effect_size_linking_vs_no_linking.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def write_metadata(config: AnalysisConfig, output_root: Path) -> None:
    metadata = {
        "analysis": "smFISH linking sensitivity using non-zero-cell sample means",
        "input_linking_root": str(config.linking_root),
        "input_no_linking_root": str(config.no_linking_root),
        "experiments": EXPERIMENT_TO_GENE,
        "time_hours": list(TIMES),
        "injuries": list(INJURIES),
        "cell_filter": "transcript_count > 0",
        "sample_statistic": "arithmetic mean transcript count among non-zero cells",
        "statistical_unit": "Image/animal sample mean",
        "test": "two-sided Welch independent-samples t-test",
        "comparison": "amputation vs incision, separately for each gene, time, and mode",
        "bar": "mean of sample means",
        "error_bar": "SEM across independent sample means",
        "multiple_testing": {
            "family": "four gene-by-time comparisons within each mode",
            "reported": ["raw p", "Holm-adjusted p", "Benjamini-Hochberg q"],
        },
        "expected_samples_per_experiment_group": config.expected_samples_per_experiment,
        "allow_incomplete": config.allow_incomplete,
        "interpretation": (
            "This metric estimates expression intensity among responding cells. "
            "It does not estimate the fraction of cells responding."
        ),
    }
    with (output_root / "analysis_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")


def run_analysis(config: AnalysisConfig) -> dict[str, pd.DataFrame]:
    config.output_root.mkdir(parents=True, exist_ok=True)
    linking = read_mode("linking", config.linking_root)
    no_linking = read_mode("no_linking", config.no_linking_root)
    cells = select_analysis_cells(pd.concat([linking, no_linking], ignore_index=True))
    sample_means = summarize_samples(cells)
    design = validate_sample_design(
        sample_means,
        expected_per_experiment=config.expected_samples_per_experiment,
        allow_incomplete=config.allow_incomplete,
    )
    group_summary = calculate_group_summary(sample_means)
    ttests = calculate_ttests(sample_means)
    paired_modes = calculate_paired_mode_table(sample_means)
    consistency = calculate_mode_consistency(ttests)

    sample_means.to_csv(config.output_root / "sample_means_nonzero_cells.csv", index=False)
    design.to_csv(config.output_root / "sample_design_qc.csv", index=False)
    group_summary.to_csv(config.output_root / "group_summary.csv", index=False)
    ttests.to_csv(config.output_root / "welch_ttest_results.csv", index=False)
    paired_modes.to_csv(
        config.output_root / "paired_linking_no_linking_sample_means.csv", index=False
    )
    consistency.to_csv(
        config.output_root / "linking_no_linking_conclusion_consistency.csv",
        index=False,
    )
    write_metadata(config, config.output_root)

    for mode in MODES:
        plot_mode_bars(sample_means, ttests, mode, config.output_root)
    plot_effect_comparison(ttests, config.output_root)

    return {
        "sample_means": sample_means,
        "design": design,
        "group_summary": group_summary,
        "ttests": ttests,
        "paired_modes": paired_modes,
        "consistency": consistency,
    }


def synthetic_cells(mode: str) -> pd.DataFrame:
    rng = np.random.default_rng(304 if mode == "linking" else 306)
    rows = []
    for experiment, gene in EXPERIMENT_TO_GENE.items():
        for time in TIMES:
            for injury in INJURIES:
                base = 2.0 + (0.6 if time == 12 else 0.0)
                effect = 1.1 if injury == "amputation" else 0.0
                mode_shift = 0.18 if mode == "linking" else 0.0
                for image in range(1, 4):
                    sample_id = f"{experiment}__{int(time)}h_{injury.title()}__Image{image}"
                    condition = f"{int(time)}h_{injury.title()}"
                    counts = np.concatenate(
                        [
                            np.zeros(30, dtype=int),
                            np.maximum(
                                1,
                                rng.poisson(base + effect + mode_shift, size=25),
                            ),
                        ]
                    )
                    for count in counts:
                        rows.append(
                            {
                                "mode": mode,
                                "experiment": experiment,
                                "gene": gene,
                                "condition": condition,
                                "time_hours": time,
                                "injury": injury,
                                "animal": f"Image{image}",
                                "sample_id": sample_id,
                                "transcript_count": int(count),
                            }
                        )
    return pd.DataFrame(rows)


def run_self_test() -> None:
    cells = pd.concat(
        [synthetic_cells("linking"), synthetic_cells("no_linking")],
        ignore_index=True,
    )
    sample_means = summarize_samples(cells)
    design = validate_sample_design(sample_means, 3, allow_incomplete=False)
    ttests = calculate_ttests(sample_means)
    paired = calculate_paired_mode_table(sample_means)
    consistency = calculate_mode_consistency(ttests)
    assert len(sample_means) == 96
    assert design["sample_count_matches"].all()
    assert len(ttests) == 8
    assert len(paired) == 48
    assert len(consistency) == 4
    assert set(ttests["n_amputation_samples"]) == {6}
    assert set(ttests["n_incision_samples"]) == {6}
    print("SELF-TEST PASSED")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0

    config = AnalysisConfig(
        linking_root=args.linking_root.expanduser().resolve(),
        no_linking_root=args.no_linking_root.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
        expected_samples_per_experiment=args.expected_samples_per_experiment,
        allow_incomplete=args.allow_incomplete,
    )
    results = run_analysis(config)
    print(f"Analysis completed: {config.output_root}")
    print("\nWelch t-tests (sample means; non-zero cells):")
    print(
        results["ttests"][
            [
                "mode",
                "gene",
                "time_hours",
                "n_amputation_samples",
                "n_incision_samples",
                "mean_difference_amputation_minus_incision",
                "p_value",
                "p_holm_within_mode",
            ]
        ].to_string(index=False)
    )
    print("\nLinking/no-linking consistency:")
    print(
        results["consistency"][
            [
                "gene",
                "time_hours",
                "same_effect_direction",
                "same_raw_significance",
                "same_holm_significance",
            ]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
