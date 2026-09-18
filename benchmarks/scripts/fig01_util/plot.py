#!/usr/bin/env python3
"""Plot measured Fig. 1(c) A3 usage without smoothing or dropped points."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


STYLES = (
    ("indexer", "Indexer + Top-k", "#4477AA", "o", "-"),
    ("sparse", "Sparse attention", "#EE7733", "^", "-"),
    ("moe", "MoE", "#AA3377", "D", "-"),
    ("dense32", "Dense attention, 32K", "#555555", "s", "--"),
    ("dense64", "Dense attention, 64K", "#228833", "v", "-."),
)
BATCHES = (4, 8, 12, 16, 24, 32, 48, 64, 96, 128)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_prefix", type=Path)
    args = parser.parse_args()
    with args.input.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(STYLES) * len(BATCHES), len(rows)
    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 8,
        "axes.labelsize": 7, "xtick.labelsize": 6.3, "ytick.labelsize": 6.3,
        "legend.fontsize": 5.1, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": .6,
        "lines.linewidth": 1.2, "pdf.fonttype": 42,
        "svg.fonttype": "none", "savefig.dpi": 220,
        "grid.alpha": .22, "grid.linewidth": .45,
    })
    fig, ax = plt.subplots(figsize=(2.35, 2.45), layout="constrained")
    for stage, label, color, marker, linestyle in STYLES:
        points = sorted((r for r in rows if r["stage"] == stage),
                        key=lambda r: int(r["local_batch"]))
        assert tuple(int(r["local_batch"]) for r in points) == BATCHES
        y = [float(r["aicore_usage_percent"]) for r in points]
        assert all(0 <= value <= 100 for value in y)
        ax.plot(BATCHES, y, label=label, color=color, marker=marker,
                linestyle=linestyle, markersize=2.8)
    ax.set(xlabel="Batch per DP replica", ylabel="AI Core usage (%)",
           xlim=(0, 132), xticks=(4, 32, 64, 96, 128),
           ylim=(0, 100), yticks=(0, 20, 40, 60, 80, 100))
    ax.set_axisbelow(True)
    ax.grid(axis="y")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.02), ncol=2,
              frameon=True, facecolor="white", edgecolor="none",
              framealpha=.92, labelspacing=.18, columnspacing=.5,
              handlelength=1.7, borderpad=.2)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(args.output_prefix.with_suffix("." + suffix),
                    bbox_inches="tight", pad_inches=.035)
    plt.close(fig)


if __name__ == "__main__":
    main()
