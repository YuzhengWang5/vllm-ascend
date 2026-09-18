#!/usr/bin/env python3
"""Validate all-rank stage runs and summarize measured A3 usage samples."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics


STAGES = ("indexer", "sparse", "moe", "dense32", "dense64")
BATCHES = (4, 8, 12, 16, 24, 32, 48, 64, 96, 128)


def records(path: Path):
    for line in path.read_text().splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue  # torchrun and CANN emit non-JSON diagnostic lines.
        if isinstance(value, dict):
            yield value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    samples = list(records(args.run_dir / "logs" / "usage_samples.jsonl"))
    rows = []
    for stage in STAGES:
        runs = list(records(args.run_dir / "logs" / f"{stage}.jsonl"))
        for batch in BATCHES:
            rank_runs = [r for r in runs if r.get("stage") == stage
                         and r.get("batch") == batch and r.get("status") == "pass"]
            assert len(rank_runs) == 16, (stage, batch, len(rank_runs))
            assert {r["rank"] for r in rank_runs} == set(range(16)), (stage, batch)
            assert min(r["wall_seconds"] for r in rank_runs) >= 15, (stage, batch)
            replays = {r["graph_replays"] for r in rank_runs}
            assert len(replays) == 1, (stage, batch, replays)
            points = [s for s in samples if s["stage"] == stage and s["batch"] == batch
                      and s.get("mode") == "tp8_no_comm"]
            assert len(points) >= 6, (stage, batch, len(points))
            points.sort(key=lambda s: s["monotonic_start"])
            # Ignore one sample at each transition. Each retained sample was
            # fully collected while the active marker contained this stage.
            trimmed = points[1:-1]
            assert len(trimmed) >= 4, (stage, batch, len(trimmed))
            for sample in trimmed:
                # The benchmark runs one rank on every die. Any extra owner
                # makes device-level usage attribution ambiguous.
                owners = sample["processes"]
                assert len(owners) == 16 and {p["die"] for p in owners} == set(range(16)), (
                    stage, batch, owners
                )
            per_sample_mean = [statistics.mean(float(s["aicore_percent_by_die"][str(i)])
                                               for i in range(16)) for s in trimmed]
            die_means = [statistics.mean(float(s["aicore_percent_by_die"][str(i)])
                                         for s in trimmed) for i in range(16)]
            rows.append({
                "stage": stage, "local_batch": batch, "global_batch": batch * 2,
                "aicore_usage_percent": round(statistics.median(per_sample_mean), 3),
                "aicore_usage_percent_min_sample": round(min(per_sample_mean), 3),
                "aicore_usage_percent_max_sample": round(max(per_sample_mean), 3),
                "aicore_usage_percent_die_mean_min": round(min(die_means), 3),
                "aicore_usage_percent_die_mean_max": round(max(die_means), 3),
                "usage_samples_total": len(points),
                "usage_samples_retained": len(trimmed),
                "graph_replays": replays.pop(),
                "device_ms_rank_max": round(max(r["device_ms_total"] for r in rank_runs), 3),
                "wall_seconds_rank_max": round(max(r["wall_seconds"] for r in rank_runs), 3),
            })
    target = args.run_dir / "logs" / "stage_aicore_utilization.csv"
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(target)


if __name__ == "__main__":
    main()
