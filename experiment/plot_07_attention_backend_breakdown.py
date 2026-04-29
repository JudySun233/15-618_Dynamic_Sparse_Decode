#!/usr/bin/env python3
"""Plot experiment 07: dense/sparse CPU/GPU attention breakdown."""

import argparse
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from plot_common import (
    DEFAULT_RESULTS_DIR,
    add_common_args,
    configure_matplotlib,
    parse_formats,
    read_rows,
    save_figure,
    to_float,
    to_int,
    plt,
)


EXP = "07_attention_backend_breakdown"
PATH_ORDER = ["sparse_gpu"]
COMPONENTS = [
    ("score_ms", "score", "#4c72b0"),
    ("topk_ms", "top-k", "#55a868"),
    ("gather_ms", "gather", "#c44e52"),
    ("attention_ms", "attention", "#8172b2"),
    ("gpu_kernel_other_ms", "GPU kernel other", "#ccb974"),
]


def row_sort_key(row: Dict[str, str]) -> Tuple[int, int, str]:
    path = row.get("path", "")
    try:
        path_rank = PATH_ORDER.index(path)
    except ValueError:
        path_rank = len(PATH_ORDER)
    return (to_int(row, "max_ctx"), path_rank, path)


def finite_component(row: Dict[str, str], key: str) -> float:
    value = to_float(row, key, 0.0)
    if not math.isfinite(value) or value < 0.0:
        return 0.0
    return value


def active_components(rows: Sequence[Dict[str, str]]) -> List[Tuple[str, str, str]]:
    return [
        component
        for component in COMPONENTS
        if any(finite_component(row, component[0]) > 1e-9 for row in rows)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot dense/sparse CPU/GPU attention breakdown.")
    add_common_args(
        parser,
        DEFAULT_RESULTS_DIR / EXP / "attention_backend_breakdown.csv",
        DEFAULT_RESULTS_DIR / EXP / "figures",
    )
    args = parser.parse_args()
    configure_matplotlib()

    rows = sorted(read_rows(Path(args.csv)), key=row_sort_key)
    formats = parse_formats(args.formats)
    out_dir = Path(args.out_dir)
    rows = [row for row in rows if row.get("path", "") in PATH_ORDER]
    if not rows:
        raise SystemExit("No sparse_gpu rows found")

    rows = sorted(rows, key=lambda row: to_int(row, "max_ctx"))
    components = active_components(rows)
    y_positions = list(range(len(rows)))
    y_labels = [f"context {to_int(row, 'max_ctx')}" for row in rows]

    fig, ax = plt.subplots(figsize=(7.0, max(3.6, 0.75 * len(rows) + 1.5)))
    lefts = [0.0 for _ in rows]
    totals = [sum(finite_component(row, key) for key, _, _ in components) for row in rows]
    max_total = max(totals) if totals else 0.0

    for key, label, color in components:
        values = [finite_component(row, key) for row in rows]
        bars = ax.barh(y_positions, values, left=lefts, label=label, color=color, height=0.58)
        for bar, value in zip(bars, values):
            if value <= 0.0 or max_total <= 0.0:
                continue
            if value >= max_total * 0.075:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_y() + bar.get_height() / 2.0,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                )
        lefts = [left + value for left, value in zip(lefts, values)]

    label_pad = max_total * 0.025 if max_total > 0.0 else 0.01
    for y, total in zip(y_positions, totals):
        ax.text(total + label_pad, y, f"{total:.3f} ms", va="center", fontsize=9)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.invert_yaxis()
    ax.set_xlabel("stage time (ms)")
    ax.set_title("Sparse GPU Attention Time Breakdown", fontweight="bold")
    ax.set_xlim(0.0, max_total * 1.22 if max_total > 0.0 else 1.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=len(components))
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    save_figure(fig, out_dir, "attention_backend_breakdown_stack", formats)

    print(f"wrote figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
