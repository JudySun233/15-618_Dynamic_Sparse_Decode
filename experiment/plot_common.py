#!/usr/bin/env python3
"""Shared helpers for report experiment plotting scripts."""

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required for plotting but is not installed in this Python environment.\n"
        "Install it with `python3 -m pip install matplotlib`, or load/use an environment that has matplotlib."
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "results" / "report_experiments"


def add_common_args(
    parser: argparse.ArgumentParser,
    default_csv: Path,
    default_out_dir: Path,
) -> None:
    parser.add_argument("--csv", default=str(default_csv), help="Input CSV from the experiment script.")
    parser.add_argument("--out-dir", default=str(default_out_dir), help="Directory for generated figures.")
    parser.add_argument(
        "--formats",
        default="png",
        help="Comma-separated output formats, for example: png or png,svg",
    )


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "font.size": 10,
            "legend.frameon": False,
        }
    )


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise SystemExit(
            f"Input CSV not found: {path}\n"
            "Run the corresponding experiment script first, or pass --csv."
        )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"Input CSV has no data rows: {path}")
    return rows


def to_float(row: Dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def to_int(row: Dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def finite_pairs(x_values: Sequence[float], y_values: Sequence[float]) -> Tuple[List[float], List[float]]:
    pairs = [(x, y) for x, y in zip(x_values, y_values) if math.isfinite(x) and math.isfinite(y)]
    return [x for x, _ in pairs], [y for _, y in pairs]


def parse_formats(value: str) -> List[str]:
    formats = [item.strip().lstrip(".") for item in value.split(",") if item.strip()]
    return formats or ["png"]


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, formats: Iterable[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out_dir / f"{stem}.{fmt}", bbox_inches="tight")
    plt.close(fig)


def add_speedup_axis(ax: plt.Axes, x: Sequence[float], dense_ms: Sequence[float], sparse_ms: Sequence[float]) -> None:
    speedup = [
        dense / sparse if math.isfinite(dense) and math.isfinite(sparse) and sparse > 0 else math.nan
        for dense, sparse in zip(dense_ms, sparse_ms)
    ]
    x_clean, y_clean = finite_pairs(x, speedup)
    if not x_clean:
        return
    ax2 = ax.twinx()
    ax2.plot(x_clean, y_clean, marker="^", color="#c44e52", label="dense/sparse speedup")
    ax2.axhline(1.0, linestyle=":", color="0.35", linewidth=1.0)
    ax2.set_ylabel("speedup")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], loc="best")


def bar_labels(ax: plt.Axes, fmt: str = "{:.2g}") -> None:
    for container in ax.containers:
        labels = []
        for patch in container:
            height = patch.get_height()
            labels.append("" if not math.isfinite(height) else fmt.format(height))
        ax.bar_label(container, labels=labels, padding=2, fontsize=8)
