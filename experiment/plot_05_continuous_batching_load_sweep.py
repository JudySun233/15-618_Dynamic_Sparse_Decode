#!/usr/bin/env python3
"""Plot experiment 05: continuous batching load sweep."""

import argparse
from pathlib import Path
from typing import Dict, List

from plot_common import (
    DEFAULT_RESULTS_DIR,
    add_common_args,
    configure_matplotlib,
    finite_pairs,
    parse_formats,
    read_rows,
    save_figure,
    to_float,
    to_int,
    plt,
)


EXP = "05_continuous_batching_load_sweep"


def plot_kernel_vs_total(
    rows: List[Dict[str, str]],
    x_key: str,
    x_label: str,
    title: str,
    stem: str,
    out_dir: Path,
    formats: List[str],
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    context_lengths = sorted({context_length(row) for row in rows}) if x_key == "max_active_requests" else []
    if len(context_lengths) > 1:
        markers = ["o", "s", "^", "D", "P", "X", "v"]
        for index, ctx in enumerate(context_lengths):
            ctx_rows = sorted(
                [row for row in rows if context_length(row) == ctx],
                key=lambda row: to_int(row, x_key),
            )
            x = [to_int(row, x_key) for row in ctx_rows]
            for key, label, linestyle in [
                ("avg_step_ms", "total", "-"),
                ("avg_kernel_ms", "kernel", "--"),
            ]:
                values = [to_float(row, key) for row in ctx_rows]
                x_clean, y_clean = finite_pairs(x, values)
                ax.plot(
                    x_clean,
                    y_clean,
                    marker=markers[index % len(markers)],
                    linestyle=linestyle,
                    label=f"ctx {ctx} {label}",
                )
        ax.set_xticks(sorted({to_int(row, x_key) for row in rows}))
    else:
        x = [to_int(row, x_key) for row in rows]
        avg_total = [to_float(row, "avg_step_ms") for row in rows]
        avg_kernel = [to_float(row, "avg_kernel_ms") for row in rows]
        for values, label, marker in [
            (avg_total, "avg total", "o"),
            (avg_kernel, "avg sparse kernel", "s"),
        ]:
            x_clean, y_clean = finite_pairs(x, values)
            ax.plot(x_clean, y_clean, marker=marker, label=label)
        ax.set_xticks(x)
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.set_xlabel(x_label)
    ax.set_ylabel("avg per decode step (ms)")
    ax.set_title(title)
    ax.legend(loc="best")
    save_figure(fig, out_dir, stem, formats)


def context_length(row: Dict[str, str]) -> int:
    min_prompt = to_int(row, "min_prompt_tokens")
    max_prompt = to_int(row, "max_prompt_tokens")
    return max_prompt if max_prompt == min_prompt else max_prompt


def plot_metric_vs_concurrency_by_context(
    rows: List[Dict[str, str]],
    y_key: str,
    y_label: str,
    title: str,
    stem: str,
    out_dir: Path,
    formats: List[str],
) -> None:
    context_lengths = sorted({context_length(row) for row in rows})

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    markers = ["o", "s", "^", "D", "P", "X", "v"]
    for index, ctx in enumerate(context_lengths):
        ctx_rows = sorted(
            [row for row in rows if context_length(row) == ctx],
            key=lambda row: to_int(row, "max_active_requests"),
        )
        x = [to_int(row, "max_active_requests") for row in ctx_rows]
        values = [to_float(row, y_key) for row in ctx_rows]
        x_clean, y_clean = finite_pairs(x, values)
        ax.plot(
            x_clean,
            y_clean,
            marker=markers[index % len(markers)],
            label=f"ctx {ctx}",
        )

    all_x = sorted({to_int(row, "max_active_requests") for row in rows})
    if y_key == "continuous_vs_serial_speedup":
        ax.axhline(1.0, linestyle=":", color="0.35", linewidth=1.0)
    ax.set_xticks(all_x)
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.set_xlabel("max active requests")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(loc="best", title="context length")
    save_figure(fig, out_dir, stem, formats)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot continuous batching load sweep results.")
    add_common_args(
        parser,
        DEFAULT_RESULTS_DIR / EXP / "continuous_batching_load_sweep.csv",
        DEFAULT_RESULTS_DIR / EXP / "figures",
    )
    args = parser.parse_args()
    configure_matplotlib()

    rows = read_rows(Path(args.csv))
    formats = parse_formats(args.formats)
    out_dir = Path(args.out_dir)

    max_active_rows = sorted(
        [row for row in rows if row.get("variant", "").startswith("max_active_")],
        key=lambda row: to_int(row, "max_active_requests"),
    )
    arrival_rows = sorted(
        [row for row in rows if row.get("variant", "").startswith("arrival_window_")],
        key=lambda row: to_int(row, "arrival_window"),
    )

    plot_metric_vs_concurrency_by_context(
        max_active_rows,
        "continuous_vs_serial_speedup",
        "speedup vs serial",
        "Speedup vs Concurrency by Context Length",
        "load_max_active_speedup",
        out_dir,
        formats,
    )

    plot_metric_vs_concurrency_by_context(
        max_active_rows,
        "avg_active_batch_size",
        "avg active batch size",
        "Average Active Batch vs Concurrency by Context Length",
        "load_max_active_avg_batch",
        out_dir,
        formats,
    )

    plot_kernel_vs_total(
        max_active_rows,
        "max_active_requests",
        "max active requests",
        "Kernel Time vs Total Time by Concurrency",
        "load_max_active_kernel_vs_total",
        out_dir,
        formats,
    )

    x = [to_int(row, "arrival_window") for row in arrival_rows]
    speedup = [to_float(row, "continuous_vs_serial_speedup") for row in arrival_rows]
    avg_batch = [to_float(row, "avg_active_batch_size") for row in arrival_rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x_clean, y_clean = finite_pairs(x, speedup)
    ax.plot(x_clean, y_clean, marker="^", color="#c44e52", label="speedup vs serial")
    ax.axhline(1.0, linestyle=":", color="0.35", linewidth=1.0)
    ax.set_xticks(x)
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.set_xlabel("arrival window")
    ax.set_ylabel("speedup vs serial")
    ax.set_title("Speedup vs Arrival Window")
    ax.legend(loc="best")
    save_figure(fig, out_dir, "load_arrival_window_speedup", formats)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x_clean, y_clean = finite_pairs(x, avg_batch)
    ax.plot(x_clean, y_clean, marker="s", color="#55a868", label="avg active batch")
    ax.set_xticks(x)
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.set_xlabel("arrival window")
    ax.set_ylabel("avg active batch size")
    ax.set_title("Average Active Batch vs Arrival Window")
    ax.legend(loc="best")
    save_figure(fig, out_dir, "load_arrival_window_avg_batch", formats)

    plot_kernel_vs_total(
        arrival_rows,
        "arrival_window",
        "arrival window",
        "Kernel Time vs Total Time by Arrival Window",
        "load_arrival_window_kernel_vs_total",
        out_dir,
        formats,
    )

    print(f"wrote figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
