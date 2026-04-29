#!/usr/bin/env python3
"""Plot experiment 06: continuous runtime options ablation."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Sequence

from plot_common import (
    DEFAULT_RESULTS_DIR,
    add_common_args,
    configure_matplotlib,
    finite_pairs,
    parse_formats,
    read_rows,
    save_figure,
    to_float,
    plt,
)


EXP = "06_continuous_runtime_options_ablation"

VARIANT_ORDER = [
    "optimized",
    "runtime_admit",
    "cpu_cache_admit",
    "no_payload_precompute",
    "cpu_decode_append",
    "eager_release",
    "debug_tensors",
    "no_kernel_events",
    "dense_gpu_optimized",
]

OVERHEAD_COMPONENTS = [
    ("avg_h2d_ms", "H2D", "#4c72b0"),
    ("avg_d2h_ms", "D2H", "#dd8452"),
    ("avg_launch_ms", "launch", "#55a868"),
    ("avg_sync_ms", "sync", "#c44e52"),
    ("avg_prepare_sparse_layout_ms", "sparse layout", "#8172b2"),
]

WALL_COMPONENTS = [
    ("decode_payload_prep_ms", "payload prep", "#4c72b0"),
    ("prompt_preadmit_ms", "prompt preadmit", "#55a868"),
    ("runtime_admission_ms", "runtime admission", "#c44e52"),
    ("run_batch_wall_ms", "run batch", "#8172b2"),
    ("append_sync_ms", "append sync", "#ccb974"),
    ("release_sync_ms", "release sync", "#64b5cd"),
    ("serving_loop_other_ms", "other loop", "#937860"),
]


def condition_order(rows: Sequence[Dict[str, str]]) -> List[str]:
    seen: List[str] = []
    for row in rows:
        condition = row.get("condition", "")
        if condition and condition not in seen:
            seen.append(condition)
    return seen


def variant_rank(row: Dict[str, str]) -> int:
    variant = row.get("variant", "")
    try:
        return VARIANT_ORDER.index(variant)
    except ValueError:
        return len(VARIANT_ORDER)


def finite_value(row: Dict[str, str], key: str) -> float:
    value = to_float(row, key, 0.0)
    if not math.isfinite(value) or value < 0.0:
        return 0.0
    return value


def short_variant_label(variant: str) -> str:
    labels = {
        "optimized": "opt",
        "runtime_admit": "runtime admit",
        "cpu_cache_admit": "CPU cache",
        "no_payload_precompute": "no precompute",
        "cpu_decode_append": "CPU append",
        "eager_release": "eager release",
        "debug_tensors": "debug tensors",
        "no_kernel_events": "no events",
        "dense_gpu_optimized": "dense GPU",
    }
    return labels.get(variant, variant.replace("_", " "))


def plot_stacked_components(
    rows: Sequence[Dict[str, str]],
    condition: str,
    components: Sequence[tuple[str, str, str]],
    ylabel: str,
    title: str,
    stem: str,
    out_dir: Path,
    formats: Sequence[str],
) -> None:
    condition_rows = sorted(
        [row for row in rows if row.get("condition") == condition],
        key=variant_rank,
    )
    labels = [short_variant_label(row.get("variant", "")) for row in condition_rows]
    x = list(range(len(condition_rows)))
    bottoms = [0.0 for _ in condition_rows]

    fig, ax = plt.subplots(figsize=(max(8.0, 0.78 * len(condition_rows)), 4.8))
    for key, label, color in components:
        values = [finite_value(row, key) for row in condition_rows]
        ax.bar(x, values, bottom=bottoms, label=label, color=color)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    offset = max(bottoms) * 0.015 if bottoms and max(bottoms) > 0.0 else 0.01
    for index, total in enumerate(bottoms):
        if total > 0.0:
            ax.text(index, total + offset, f"{total:.3g}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(ncols=3, loc="best")
    fig.tight_layout()
    save_figure(fig, out_dir, stem, formats)


def plot_total_overhead_by_condition(
    rows: Sequence[Dict[str, str]],
    conditions: Sequence[str],
    out_dir: Path,
    formats: Sequence[str],
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for variant in VARIANT_ORDER:
        variant_rows = [row for row in rows if row.get("variant") == variant]
        values = []
        for condition in conditions:
            match = next((row for row in variant_rows if row.get("condition") == condition), None)
            values.append(finite_value(match, "total_runtime_overhead_ms") if match else math.nan)
        x_clean, y_clean = finite_pairs(list(range(len(conditions))), values)
        if x_clean:
            ax.plot(x_clean, y_clean, marker="o", label=short_variant_label(variant))

    ax.set_xticks(list(range(len(conditions))))
    ax.set_xticklabels([condition.replace("_", " ") for condition in conditions], rotation=15, ha="right")
    ax.set_ylabel("avg runtime overhead per step (ms)")
    ax.set_title("Runtime Overhead Across Workload Conditions")
    ax.legend(ncols=2, fontsize=8)
    fig.tight_layout()
    save_figure(fig, out_dir, "runtime_overhead_by_condition", formats)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot continuous runtime options ablation.")
    add_common_args(
        parser,
        DEFAULT_RESULTS_DIR / EXP / "continuous_runtime_options_ablation.csv",
        DEFAULT_RESULTS_DIR / EXP / "figures",
    )
    args = parser.parse_args()
    configure_matplotlib()

    rows = read_rows(Path(args.csv))
    formats = parse_formats(args.formats)
    out_dir = Path(args.out_dir)
    conditions = condition_order(rows)

    for condition in conditions:
        safe_condition = condition.replace("/", "_")
        plot_stacked_components(
            rows,
            condition,
            OVERHEAD_COMPONENTS,
            "avg runtime overhead per step (ms)",
            f"Runtime Overhead Components: {condition.replace('_', ' ')}",
            f"{safe_condition}_runtime_overhead_stack",
            out_dir,
            formats,
        )
        plot_stacked_components(
            rows,
            condition,
            WALL_COMPONENTS,
            "total serving-loop time (ms)",
            f"Serving Loop Components: {condition.replace('_', ' ')}",
            f"{safe_condition}_serving_loop_stack",
            out_dir,
            formats,
        )

    plot_total_overhead_by_condition(rows, conditions, out_dir, formats)

    print(f"wrote figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
