#!/usr/bin/env python3
"""Three-panel attention parallelism comparison from measured usage only."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


STAGES = (("dense32", "Dense MLA, 32K"),
          ("dense64", "Dense MLA, 64K"),
          ("sparse", "Sparse MLA, top-k 2048"))
MODES = (("single", "Single die, 128 heads", "#4477AA", "o", "-"),
         ("tp8_no_comm", "TP8 local, 16 heads", "#EE7733", "s", "--"),
         ("tp8_comm", "TP8 + all-reduce", "#AA3377", "^", "-."))
BATCHES = (4, 8, 16, 32, 64, 128)


def main():
    source = Path(sys.argv[1])
    prefix = Path(sys.argv[2])
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(STAGES) * len(MODES) * len(BATCHES)
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none",
                         "savefig.dpi": 220})
    fig, axes = plt.subplots(1, 3, figsize=(8.1, 2.6), sharey=True,
                             layout="constrained")
    for ax, (stage, title) in zip(axes, STAGES):
        for mode, label, color, marker, linestyle in MODES:
            points = sorted((r for r in rows if r["stage"] == stage and r["mode"] == mode),
                            key=lambda r: int(r["local_batch"]))
            assert tuple(int(r["local_batch"]) for r in points) == BATCHES
            ax.plot(BATCHES, [float(r["aicore_usage_percent"]) for r in points],
                    label=label, color=color, marker=marker, linestyle=linestyle,
                    linewidth=1.3, markersize=3)
        ax.set(title=title, xlabel="Batch per DP replica", xlim=(0, 132),
               xticks=(4, 32, 64, 128), ylim=(0, 100),
               yticks=(0, 20, 40, 60, 80, 100))
        ax.grid(axis="y", alpha=.23)
    axes[0].set_ylabel("AI Core usage (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(.5, 1.08), frameon=False, fontsize=8)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(prefix.with_suffix("." + suffix), bbox_inches="tight", pad_inches=.035)
    plt.close(fig)


if __name__ == "__main__":
    main()
