#!/usr/bin/env python3
"""Plot experiment 02: decode batch-size scaling."""

from __future__ import annotations

import argparse
from pathlib import Path

from plot_common import (
    DEFAULT_RESULTS_DIR,
    add_common_args,
    add_speedup_axis,
    configure_matplotlib,
    finite_pairs,
    parse_formats,
    read_rows,
    save_figure,
    to_float,
    to_int,
    plt,
)


EXP = "02_decode_batch_scaling_sweep"


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot decode batch scaling results.")
    add_common_args(
        parser,
        DEFAULT_RESULTS_DIR / EXP / "decode_batch_scaling_sweep.csv",
        DEFAULT_RESULTS_DIR / EXP / "figures",
    )
    args = parser.parse_args()
    configure_matplotlib()

    rows = sorted(read_rows(Path(args.csv)), key=lambda row: to_int(row, "batch_size"))
    formats = parse_formats(args.formats)
    out_dir = Path(args.out_dir)

    x = [to_int(row, "batch_size") for row in rows]
    dense_gpu = [to_float(row, "dense_gpu_total_ms") for row in rows]
    sparse_gpu = [to_float(row, "sparse_gpu_total_ms") for row in rows]
    overhead = [to_float(row, "sparse_gpu_total_overhead_ms") for row in rows]
    kernel = [to_float(row, "sparse_gpu_kernel_ms") for row in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for values, label, marker in [
        (dense_gpu, "dense GPU total", "o"),
        (sparse_gpu, "sparse GPU total", "s"),
    ]:
        x_clean, y_clean = finite_pairs(x, values)
        ax.plot(x_clean, y_clean, marker=marker, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("batch size")
    ax.set_ylabel("latency (ms)")
    ax.set_title("Decode Latency vs Batch Size")
    add_speedup_axis(ax, x, dense_gpu, sparse_gpu)
    save_figure(fig, out_dir, "batch_scaling_latency", formats)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x_clean, y_kernel = finite_pairs(x, kernel)
    ax.plot(x_clean, y_kernel, marker="o", label="sparse kernels")
    x_clean, y_overhead = finite_pairs(x, overhead)
    ax.plot(x_clean, y_overhead, marker="s", label="sparse overhead")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("batch size")
    ax.set_ylabel("time (ms)")
    ax.set_title("Sparse Runtime Breakdown vs Batch Size")
    ax.legend(loc="best")
    save_figure(fig, out_dir, "batch_scaling_sparse_breakdown", formats)

    print(f"wrote figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
