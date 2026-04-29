#!/usr/bin/env python3
"""Plot experiment 03: top-k sparsity tradeoff."""

from __future__ import annotations

import argparse
from pathlib import Path

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


EXP = "03_decode_topk_sparsity_sweep"


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot decode top-k sparsity sweep results.")
    add_common_args(
        parser,
        DEFAULT_RESULTS_DIR / EXP / "decode_topk_sparsity_sweep.csv",
        DEFAULT_RESULTS_DIR / EXP / "figures",
    )
    args = parser.parse_args()
    configure_matplotlib()

    rows = sorted(read_rows(Path(args.csv)), key=lambda row: to_int(row, "top_k_pages"))
    formats = parse_formats(args.formats)
    out_dir = Path(args.out_dir)

    x = [to_int(row, "top_k_pages") for row in rows]
    sparse_gpu = [to_float(row, "sparse_gpu_total_ms") for row in rows]
    selected = [to_float(row, "total_selected_tokens") for row in rows]
    target_bytes = [to_float(row, "sparse_target_over_dense_bytes") for row in rows]
    diff = [to_float(row, "avg_max_abs_diff_sparse_gpu_vs_dense_cpu") for row in rows]

    fig, ax1 = plt.subplots(figsize=(7.2, 4.4))
    x_clean, y_clean = finite_pairs(x, sparse_gpu)
    ax1.plot(x_clean, y_clean, marker="o", label="sparse GPU total")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("top_k_pages")
    ax1.set_ylabel("latency (ms)")
    ax2 = ax1.twinx()
    x_clean, y_clean = finite_pairs(x, selected)
    ax2.plot(x_clean, y_clean, marker="s", color="#55a868", label="selected tokens")
    ax2.set_ylabel("selected tokens")
    ax1.set_title("Runtime Cost of Selecting More Pages")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    save_figure(fig, out_dir, "topk_runtime_selected_tokens", formats)

    fig, ax1 = plt.subplots(figsize=(7.2, 4.4))
    x_clean, y_clean = finite_pairs(x, target_bytes)
    ax1.plot(x_clean, y_clean, marker="o", color="#4c72b0", label="target sparse / dense bytes")
    ax1.axhline(1.0, linestyle=":", color="0.35", linewidth=1.0)
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("top_k_pages")
    ax1.set_ylabel("memory traffic ratio")
    ax2 = ax1.twinx()
    x_clean, y_clean = finite_pairs(x, diff)
    ax2.plot(x_clean, y_clean, marker="s", color="#c44e52", label="max abs diff vs dense CPU")
    ax2.set_ylabel("max abs diff")
    ax1.set_title("Sparsity vs Memory and Accuracy")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    save_figure(fig, out_dir, "topk_memory_accuracy_tradeoff", formats)

    print(f"wrote figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
