#!/usr/bin/env python3
"""Plot experiment 04: page-size sweep."""

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


EXP = "04_decode_page_size_sweep"


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot decode page-size sweep results.")
    add_common_args(
        parser,
        DEFAULT_RESULTS_DIR / EXP / "decode_page_size_sweep.csv",
        DEFAULT_RESULTS_DIR / EXP / "figures",
    )
    args = parser.parse_args()
    configure_matplotlib()

    rows = sorted(read_rows(Path(args.csv)), key=lambda row: to_int(row, "page_size"))
    formats = parse_formats(args.formats)
    out_dir = Path(args.out_dir)

    x = [to_int(row, "page_size") for row in rows]
    dense_gpu = [to_float(row, "dense_gpu_total_ms") for row in rows]
    sparse_gpu = [to_float(row, "sparse_gpu_total_ms") for row in rows]
    score = [to_float(row, "sparse_gpu_score_ms") for row in rows]
    topk = [to_float(row, "sparse_gpu_topk_ms") for row in rows]
    attention = [to_float(row, "sparse_gpu_attention_ms") for row in rows]
    candidate_pages = [to_float(row, "total_candidate_pages") for row in rows]
    selected_tokens = [to_float(row, "total_selected_tokens") for row in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    width = 0.22
    offsets = [-width, 0.0, width]
    for values, label, offset in [
        (score, "score", offsets[0]),
        (topk, "top-k", offsets[1]),
        (attention, "attention", offsets[2]),
    ]:
        ax.bar([value + offset * value for value in x], values, width=[width * value for value in x], label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(value) for value in x])
    ax.set_xlabel("page size (tokens)")
    ax.set_ylabel("time (ms)")
    ax.set_title("Sparse GPU Stage Costs vs Page Size")
    ax.legend(loc="best")
    save_figure(fig, out_dir, "page_size_stage_costs", formats)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for values, label, marker in [
        (dense_gpu, "dense GPU total", "o"),
        (sparse_gpu, "sparse GPU total", "s"),
    ]:
        x_clean, y_clean = finite_pairs(x, values)
        ax.plot(x_clean, y_clean, marker=marker, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(value) for value in x])
    ax.set_xlabel("page size (tokens)")
    ax.set_ylabel("end-to-end latency (ms)")
    ax.set_title("End-to-End Latency vs Page Size")
    ax.legend(loc="best")
    save_figure(fig, out_dir, "page_size_end_to_end_latency", formats)

    fig, ax1 = plt.subplots(figsize=(7.2, 4.4))
    x_clean, y_clean = finite_pairs(x, candidate_pages)
    ax1.plot(x_clean, y_clean, marker="o", label="candidate pages")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("page size (tokens)")
    ax1.set_ylabel("candidate pages")
    ax2 = ax1.twinx()
    x_clean, y_clean = finite_pairs(x, selected_tokens)
    ax2.plot(x_clean, y_clean, marker="s", color="#55a868", label="selected tokens")
    ax2.set_ylabel("selected tokens")
    ax1.set_title("Page Granularity Changes the Sparse Work")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    save_figure(fig, out_dir, "page_size_work_shape", formats)

    print(f"wrote figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
