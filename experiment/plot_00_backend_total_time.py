#!/usr/bin/env python3
"""Plot experiment 00: dense/sparse CPU/GPU end-to-end total time."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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


EXP = "00_backend_total_time"

PATH_ORDER = ["dense_cpu", "sparse_cpu", "dense_gpu", "sparse_gpu"]
CONTEXT_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "*"]


def path_label(path: str) -> str:
    labels = {
        "dense_cpu": "dense CPU",
        "sparse_cpu": "sparse CPU",
        "dense_gpu": "dense GPU",
        "sparse_gpu": "sparse GPU",
    }
    return labels.get(path, path.replace("_", " "))


def row_sort_key(row: Dict[str, str]) -> Tuple[int, int, str]:
    path = row.get("path", "")
    try:
        path_rank = PATH_ORDER.index(path)
    except ValueError:
        path_rank = len(PATH_ORDER)
    return (to_int(row, "max_ctx"), path_rank, path)


def contexts(rows: Sequence[Dict[str, str]]) -> List[int]:
    return sorted({to_int(row, "max_ctx") for row in rows})


def value_for(rows: Sequence[Dict[str, str]], ctx: int, path: str, key: str) -> float:
    for row in rows:
        if to_int(row, "max_ctx") == ctx and row.get("path") == path:
            value = to_float(row, key, math.nan)
            return value if math.isfinite(value) and value > 0.0 else math.nan
    return math.nan


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot dense/sparse CPU/GPU total time.")
    add_common_args(
        parser,
        DEFAULT_RESULTS_DIR / EXP / "backend_total_time.csv",
        DEFAULT_RESULTS_DIR / EXP / "figures",
    )
    args = parser.parse_args()
    configure_matplotlib()

    rows = sorted(read_rows(Path(args.csv)), key=row_sort_key)
    formats = parse_formats(args.formats)
    out_dir = Path(args.out_dir)
    ctxs = contexts(rows)

    fig, ax = plt.subplots(figsize=(max(6.5, 1.35 * len(PATH_ORDER)), 4.6))
    x_positions = list(range(len(PATH_ORDER)))

    for index, ctx in enumerate(ctxs):
        values = [value_for(rows, ctx, path, "total_ms") for path in PATH_ORDER]
        x_clean, y_clean = finite_pairs(x_positions, values)
        if not x_clean:
            continue
        ax.plot(
            x_clean,
            y_clean,
            marker=CONTEXT_MARKERS[index % len(CONTEXT_MARKERS)],
            linewidth=2.0,
            label=f"{ctx}",
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([path_label(path) for path in PATH_ORDER])
    ax.set_xlabel("experiment condition setting")
    ax.set_ylabel("total time (ms, log scale)")
    ax.set_yscale("log")
    ax.set_title("End-to-End Decode Total Time")
    ax.legend(title="context length", ncol=2)
    fig.tight_layout()
    save_figure(fig, out_dir, "backend_total_time", formats)

    print(f"wrote figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
