#!/usr/bin/env python3
"""Recompute steady decode metrics from archived per-token client timestamps.

The initial streaming response may contain several token IDs in one event.
Starting after token 100 avoids most such startup bursts; tokens sharing the
boundary timestamp are excluded from both the count and the time window.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


WARMUP_TOKENS = 100
TOTAL_TOKENS = 250


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    return ordered[low] * (high - pos) + ordered[high] * (pos - low)


def recalculate(point: dict) -> dict:
    batch = point["global_batch"]
    dp_size = point["dp_size"]
    attempts = []
    all_tpot_intervals_ms = []
    all_effective_tpot_ms = []
    for repeat in point["repeats"]:
        requests = repeat["requests"]
        assert len(requests) == batch
        observations = []
        for request in requests:
            stamps = request["token_timestamps"]
            assert len(stamps) == TOTAL_TOKENS
            boundary = stamps[WARMUP_TOKENS - 1]
            measured = [stamp for stamp in stamps[WARMUP_TOKENS:] if stamp > boundary]
            assert len(measured) >= 130, (point["context_tokens"], batch, len(measured))
            intervals = [(right - left) * 1000 for left, right in zip([boundary] + measured[:-1], measured)]
            all_tpot_intervals_ms.extend(intervals)
            all_effective_tpot_ms.append((measured[-1] - boundary) * 1000 / len(measured))
            observations.append({
                "request_id": request["request_id"],
                "dp_rank": request["request_id"] % dp_size,
                "boundary": boundary,
                "end": measured[-1],
                "measured_tokens": len(measured),
                "timestamps": measured,
            })

        boundaries = [item["boundary"] for item in observations]
        ends = [item["end"] for item in observations]
        assert max(boundaries) < min(ends), "no all-DP overlap"
        duration = max(ends) - min(boundaries)
        count = sum(item["measured_tokens"] for item in observations)
        per_dp_rates = []
        for rank in range(dp_size):
            members = [item for item in observations if item["dp_rank"] == rank]
            per_dp_rates.append(
                sum(item["measured_tokens"] for item in members)
                / (max(item["end"] for item in members) - min(item["boundary"] for item in members))
            )
        overlap_start = max(boundaries)
        overlap_end = min(ends)
        overlap_count = sum(
            overlap_start < stamp <= overlap_end
            for item in observations
            for stamp in item["timestamps"]
        )
        attempts.append({
            "attempt": repeat["attempt"],
            "throughput_tok_s": count / duration,
            "per_dp_sum_throughput_tok_s": sum(per_dp_rates),
            "overlap_throughput_tok_s": overlap_count / (overlap_end - overlap_start),
            "measured_tokens": count,
            "expected_tokens_without_boundary_batching": batch * (TOTAL_TOKENS - WARMUP_TOKENS),
            "min_measured_tokens_per_request": min(item["measured_tokens"] for item in observations),
            "max_measured_tokens_per_request": max(item["measured_tokens"] for item in observations),
            "duration_s": duration,
            "warmup_boundary_skew_ms": (max(boundaries) - min(boundaries)) * 1000,
        })

    rates = [attempt["throughput_tok_s"] for attempt in attempts]
    return {
        "status": "ok",
        "variant": point["variant"],
        "setup_id": point["setup_id"],
        "tested_commit": point["tested_commit"],
        "context_tokens": point["context_tokens"],
        "global_batch": batch,
        "dp_size": dp_size,
        "tp_size": point["tp_size"],
        "warmup_tokens": WARMUP_TOKENS,
        "generated_tokens": TOTAL_TOKENS,
        "measurement_policy": "count only tokens strictly after token 100 timestamp",
        "repeats": attempts,
        "aggregate": {
            "throughput_tok_s_mean": statistics.fmean(rates),
            "throughput_tok_s_median": statistics.median(rates),
            "throughput_tok_s_min": min(rates),
            "throughput_tok_s_max": max(rates),
            "per_dp_sum_throughput_tok_s_mean": statistics.fmean(
                item["per_dp_sum_throughput_tok_s"] for item in attempts
            ),
            "overlap_throughput_tok_s_mean": statistics.fmean(
                item["overlap_throughput_tok_s"] for item in attempts
            ),
            "tpot_ms_p50": percentile(all_tpot_intervals_ms, 0.5),
            "tpot_ms_p95": percentile(all_tpot_intervals_ms, 0.95),
            "effective_tpot_per_request_ms_p50": percentile(all_effective_tpot_ms, 0.5),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--point-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    points = []
    for source_dir in args.point_dir:
        for source_file in sorted(source_dir.glob("seq*_batch*.json")):
            original = json.loads(source_file.read_text())
            if original.get("status") != "ok":
                continue
            corrected = recalculate(original)
            corrected["original_point_path"] = str(source_file)
            output_file = args.output_dir / source_file.name
            if output_file.exists():
                raise FileExistsError(output_file)
            output_file.write_text(json.dumps(corrected, indent=2) + "\n")
            points.append(corrected)
    points.sort(key=lambda item: (item["context_tokens"], item["global_batch"]))
    (args.output_dir / "matrix.json").write_text(json.dumps(points, indent=2) + "\n")
    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        fields = ["context_tokens", "global_batch", "variant", "setup_id", "tested_commit",
                  "throughput_tok_s_mean", "throughput_tok_s_min", "throughput_tok_s_max",
                  "tpot_ms_p50", "tpot_ms_p95", "effective_tpot_per_request_ms_p50",
                  "per_dp_sum_throughput_tok_s_mean", "overlap_throughput_tok_s_mean"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for point in points:
            writer.writerow({
                **{key: point[key] for key in fields[:5]},
                **{key: point["aggregate"][key] for key in fields[5:]},
            })
    print(f"recomputed {len(points)} complete points into {args.output_dir}")


if __name__ == "__main__":
    main()
