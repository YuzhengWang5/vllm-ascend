#!/usr/bin/env python3
"""Post-process full-concurrency overlap throughput from raw streaming timestamps."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


WARMUP_TOKENS = 50


def overlap_for_repeat(repeat: dict) -> dict:
    requests = repeat["requests"]
    boundaries = [item["token_timestamps"][WARMUP_TOKENS - 1] for item in requests]
    ends = [item["token_timestamps"][-1] for item in requests]
    start = max(boundaries)
    end = min(ends)
    if end <= start:
        raise ValueError("no interval in which every request is measuring")
    count = sum(
        start < timestamp <= end
        for item in requests
        for timestamp in item["token_timestamps"][WARMUP_TOKENS:]
    )
    return {
        "attempt": repeat["attempt"],
        "overlap_duration_s": end - start,
        "overlap_token_count": count,
        "overlap_throughput_tok_s": count / (end - start),
        "original_throughput_tok_s": repeat["throughput_tok_s"],
        "warmup_boundary_skew_ms": repeat["warmup_boundary_skew_ms"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrices", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for matrix_path in args.matrices:
        for point in json.loads(matrix_path.read_text(encoding="utf-8")):
            if point["status"] != "ok":
                continue
            repeats = [overlap_for_repeat(item) for item in point["repeats"]]
            rows.append({
                "variant": point["variant"],
                "context_tokens": point["context_tokens"],
                "global_batch": point["global_batch"],
                "overlap_throughput_tok_s_mean": statistics.fmean(
                    item["overlap_throughput_tok_s"] for item in repeats
                ),
                "original_throughput_tok_s_mean": point["aggregate"]["throughput_tok_s_mean"],
                "warmup_boundary_skew_ms_max": max(
                    item["warmup_boundary_skew_ms"] for item in repeats
                ),
                "repeats": repeats,
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "variant", "context_tokens", "global_batch",
            "overlap_throughput_tok_s_mean", "original_throughput_tok_s_mean",
            "warmup_boundary_skew_ms_max",
        ])
        writer.writeheader()
        writer.writerows({key: value for key, value in row.items() if key != "repeats"} for row in rows)


if __name__ == "__main__":
    main()
