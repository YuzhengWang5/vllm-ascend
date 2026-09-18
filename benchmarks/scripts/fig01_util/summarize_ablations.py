#!/usr/bin/env python3
"""Compare single die, TP8 local compute, and TP8 plus collective."""

from __future__ import annotations

import csv
from pathlib import Path
import statistics
import sys

from summarize import records


STAGES = ("dense32", "dense64", "sparse")
MODES = ("single", "tp8_no_comm", "tp8_comm")
BATCHES = (4, 8, 16, 32, 64, 128)


def main():
    run_dir = Path(sys.argv[1])
    samples = list(records(run_dir / "logs" / "usage_samples.jsonl"))
    rows = []
    for stage in STAGES:
        for mode in MODES:
            path = (run_dir / "logs" / f"{stage}.jsonl" if mode == "tp8_no_comm"
                    else run_dir / "logs" / f"ablation_{stage}_{mode}.jsonl")
            runs = list(records(path))
            ranks = 1 if mode == "single" else 16
            active_dies = [0] if mode == "single" else list(range(16))
            for batch in BATCHES:
                measurements = [r for r in runs if r.get("stage") == stage
                                and r.get("mode") == mode
                                and r.get("batch") == batch and r.get("status") == "pass"]
                assert len(measurements) == ranks, (stage, mode, batch, len(measurements))
                assert {r["rank"] for r in measurements} == set(range(ranks))
                points = [s for s in samples if s.get("stage") == stage
                          and s.get("mode") == mode and s.get("batch") == batch]
                assert len(points) >= 6, (stage, mode, batch, len(points))
                points.sort(key=lambda x: x["monotonic_start"])
                trimmed = points[1:-1]
                for sample in trimmed:
                    owners = sample["processes"]
                    assert len(owners) == ranks and {p["die"] for p in owners} == set(active_dies), (
                        stage, mode, batch, owners
                    )
                usage = [statistics.mean(float(s["aicore_percent_by_die"][str(i)])
                                         for i in active_dies) for s in trimmed]
                rows.append({
                    "stage": stage, "mode": mode, "local_batch": batch,
                    "attention_heads_per_die": 128 if mode == "single" else 16,
                    "active_dies": ranks,
                    "aicore_usage_percent": round(statistics.median(usage), 3),
                    "aicore_usage_percent_min_sample": round(min(usage), 3),
                    "aicore_usage_percent_max_sample": round(max(usage), 3),
                    "usage_samples_total": len(points),
                    "usage_samples_retained": len(trimmed),
                    "device_ms_rank_max": round(max(r["device_ms_total"] for r in measurements), 3),
                    "wall_seconds_rank_max": round(max(r["wall_seconds"] for r in measurements), 3),
                    "graph_replays": measurements[0]["graph_replays"],
                })
    output = run_dir / "logs" / "attention_mode_comparison.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
