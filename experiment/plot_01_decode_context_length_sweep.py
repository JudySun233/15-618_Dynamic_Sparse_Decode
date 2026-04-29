#!/usr/bin/env python3
"""Plot experiment 01: context-length sweep."""

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


EXP = "01_decode_context_length_sweep"


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot decode context-length sweep results.")
    add_common_args(
        parser,
        DEFAULT_RESULTS_DIR / EXP / "decode_context_length_sweep.csv",
        DEFAULT_RESULTS_DIR / EXP / "figures",
    )
    args = parser.parse_args()
    configure_matplotlib()

    rows = sorted(read_rows(Path(args.csv)), key=lambda row: to_int(row, "max_ctx"))
    formats = parse_formats(args.formats)
    out_dir = Path(args.out_dir)

    x = [to_int(row, "max_ctx") for row in rows]
    dense_gpu = [to_float(row, "dense_gpu_total_ms") for row in rows]
    dense_kernel = [to_float(row, "dense_gpu_kernel_ms") for row in rows]
    sparse_gpu = [to_float(row, "sparse_gpu_total_ms") for row in rows]
    sparse_kernel = [to_float(row, "sparse_gpu_kernel_ms") for row in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for values, label, marker in [
        (dense_gpu, "dense GPU total", "o"),
        (dense_kernel, "dense GPU kernels only", "^"),
        (sparse_gpu, "sparse GPU total", "s"),
        (sparse_kernel, "sparse GPU kernels only", "D"),
    ]:
        x_clean, y_clean = finite_pairs(x, values)
        ax.plot(x_clean, y_clean, marker=marker, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("latency (ms)")
    ax.set_title("Decode Latency vs Context Length")
    ax.legend(loc="best")
    save_figure(fig, out_dir, "context_length_latency", formats)

    print(f"wrote figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
