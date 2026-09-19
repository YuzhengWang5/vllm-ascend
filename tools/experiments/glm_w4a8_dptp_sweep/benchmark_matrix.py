#!/usr/bin/env python3
"""GLM-5 BF16 index, 4096 hot buffer, 819/2048 miss matrix."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import aiohttp


CONTEXTS = (32 * 1024, 64 * 1024, 128 * 1024)
BATCHES = (16, 32, 48, 64)
WARMUP_TOKENS = 100
MEASURE_TOKENS = 150
MAX_TOKENS = WARMUP_TOKENS + MEASURE_TOKENS
MIN_STEADY_TOKENS = 120
BLOCK_SIZE = 128
MLA_KV_BYTES_PER_TOKEN = 78 * (512 + 64) * 2


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


async def wait_ready(session: aiohttp.ClientSession, base_url: str) -> None:
    deadline = time.monotonic() + 1200
    while time.monotonic() < deadline:
        try:
            async with session.get(f"{base_url}/health") as response:
                if response.status == 200:
                    return
        except aiohttp.ClientError:
            pass
        await asyncio.sleep(1)
    raise TimeoutError("server is not healthy after 1200 seconds")


async def one_request(
    session: aiohttp.ClientSession,
    base_url: str,
    request_id: int,
    context: int,
    start: asyncio.Event,
    dp_size: int,
) -> dict[str, Any]:
    body = {
        "model": "GLM-5",
        "prompt": [100 + request_id] * context,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "return_token_ids": True,
    }
    await start.wait()
    submitted_at = time.perf_counter()
    timestamps: list[float] = []
    token_ids: list[int] = []
    headers = {"X-data-parallel-rank": str(request_id % dp_size)}
    async with session.post(
        f"{base_url}/v1/completions", json=body, headers=headers
    ) as response:
        if response.status != 200:
            raise RuntimeError(
                f"request {request_id}: HTTP {response.status}: {await response.text()}"
            )
        async for raw_line in response.content:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            item = json.loads(line[6:])
            if not item.get("choices"):
                continue
            ids = item["choices"][0].get("token_ids") or []
            token_ids.extend(ids)
            timestamps.extend([time.perf_counter()] * len(ids))
    return {
        "request_id": request_id,
        "submitted_at": submitted_at,
        "finished_at": time.perf_counter(),
        "token_timestamps": timestamps,
        "generated_token_ids": token_ids,
    }


def steady_window(request: dict[str, Any]) -> tuple[float, list[float], list[float]]:
    """Measure only tokens that arrived after the warmup token's SSE event."""
    stamps = request["token_timestamps"]
    boundary = stamps[WARMUP_TOKENS - 1]
    measured = [stamp for stamp in stamps[WARMUP_TOKENS:] if stamp > boundary]
    if len(measured) < MIN_STEADY_TOKENS:
        raise RuntimeError(
            f"only {len(measured)} steady tokens after warmup boundary for "
            f"request {request['request_id']}"
        )
    intervals_ms = [
        (right - left) * 1000
        for left, right in zip([boundary] + measured[:-1], measured)
    ]
    return boundary, measured, intervals_ms


async def run_attempt(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    context: int,
    batch: int,
    attempt: int,
) -> dict[str, Any]:
    start = asyncio.Event()
    tasks = [
        asyncio.create_task(
            one_request(session, args.base_url, request_id, context, start, args.dp_size)
        )
        for request_id in range(batch)
    ]
    await asyncio.sleep(0)
    start.set()
    requests = await asyncio.gather(*tasks)
    lengths = [len(request["token_timestamps"]) for request in requests]
    if lengths != [MAX_TOKENS] * batch:
        raise RuntimeError(f"token count mismatch: {lengths}")

    windows = [steady_window(request) for request in requests]
    boundaries = [window[0] for window in windows]
    ends = [window[1][-1] for window in windows]
    intervals_ms = [interval for window in windows for interval in window[2]]
    measured_count = sum(len(window[1]) for window in windows)
    duration_s = max(ends) - min(boundaries)
    per_dp_rates = []
    for dp_rank in range(args.dp_size):
        dp_windows = [
            window for request, window in zip(requests, windows)
            if request["request_id"] % args.dp_size == dp_rank
        ]
        dp_start = min(window[0] for window in dp_windows)
        dp_end = max(window[1][-1] for window in dp_windows)
        per_dp_rates.append(sum(len(window[1]) for window in dp_windows) / (dp_end - dp_start))
    overlap_start = max(boundaries)
    overlap_end = min(ends)
    overlap_duration_s = max(0.0, overlap_end - overlap_start)
    overlap_tokens = sum(
        overlap_start < timestamp <= overlap_end
        for window in windows
        for timestamp in window[1]
    )
    return {
        "attempt": attempt,
        "throughput_tok_s": measured_count / duration_s,
        "measured_tokens": measured_count,
        "min_measured_tokens_per_request": min(len(window[1]) for window in windows),
        "max_measured_tokens_per_request": max(len(window[1]) for window in windows),
        "per_dp_sum_throughput_tok_s": sum(per_dp_rates),
        "per_dp_throughput_tok_s": per_dp_rates,
        "overlap_throughput_tok_s": (
            overlap_tokens / overlap_duration_s if overlap_duration_s > 0 else None
        ),
        "overlap_duration_s": overlap_duration_s,
        "overlap_tokens": overlap_tokens,
        "measurement_duration_s": duration_s,
        "warmup_boundary_skew_ms": (max(boundaries) - min(boundaries)) * 1000,
        "all_requests_active_at_measurement_start": max(boundaries) < min(ends),
        "tpot_ms_mean": statistics.fmean(intervals_ms),
        "tpot_ms_p50": percentile(intervals_ms, 0.50),
        "tpot_ms_p95": percentile(intervals_ms, 0.95),
        "requests": requests,
    }


async def run_point(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    context: int,
    batch: int,
) -> dict[str, Any]:
    repeats: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    attempt = 0
    while len(repeats) < args.repeats and attempt < args.max_attempts:
        item = await run_attempt(session, args, context, batch, attempt)
        print(json.dumps({k: v for k, v in item.items() if k != "requests"}), flush=True)
        (repeats if item["all_requests_active_at_measurement_start"] else rejected).append(item)
        attempt += 1
    if len(repeats) != args.repeats:
        raise RuntimeError(
            f"only {len(repeats)}/{args.repeats} all-active repeats after {attempt} attempts"
        )

    throughputs = [item["throughput_tok_s"] for item in repeats]
    per_dp_throughputs = [item["per_dp_sum_throughput_tok_s"] for item in repeats]
    overlap_throughputs = [
        item["overlap_throughput_tok_s"]
        for item in repeats
        if item["overlap_throughput_tok_s"] is not None
    ]
    all_intervals = [
        interval
        for item in repeats
        for request in item["requests"]
        for interval in steady_window(request)[2]
    ]
    return {
        "status": "ok",
        "variant": args.variant,
        "setup_id": args.setup_id,
        "tested_commit": args.tested_commit,
        "tp_size": args.tp_size,
        "resident_policy": (
            "service_directmap_oracle_819_miss"
            if args.variant == "iaas"
            else "local_lru_oracle_819_miss"
        ),
        "context_tokens": context,
        "global_batch": batch,
        "dp_size": args.dp_size,
        "local_batch_per_dp": batch // args.dp_size,
        "warmup_tokens": WARMUP_TOKENS,
        "measure_tokens": MEASURE_TOKENS,
        "measurement_policy": "count only tokens strictly after token 100 timestamp",
        "repeats": repeats,
        "rejected_repeats": rejected,
        "aggregate": {
            "throughput_tok_s_mean": statistics.fmean(throughputs),
            "throughput_tok_s_median": statistics.median(throughputs),
            "throughput_tok_s_min": min(throughputs),
            "throughput_tok_s_max": max(throughputs),
            "per_dp_sum_throughput_tok_s_mean": statistics.fmean(per_dp_throughputs),
            "overlap_throughput_tok_s_mean": (
                statistics.fmean(overlap_throughputs) if overlap_throughputs else None
            ),
            "tpot_ms_mean": statistics.fmean(all_intervals),
            "tpot_ms_p50": percentile(all_intervals, 0.50),
            "tpot_ms_p95": percentile(all_intervals, 0.95),
            "tpot_ms_p99": percentile(all_intervals, 0.99),
        },
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("iaas", "baseline1"), required=True)
    parser.add_argument("--dp-size", type=int, required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--setup-id", required=True)
    parser.add_argument("--tested-commit", required=True)
    parser.add_argument("--capacity-blocks", type=int, required=True)
    parser.add_argument("--dram-pool-gib-per-dp", type=int, default=48)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18300")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=9)
    parser.add_argument("--contexts", type=int, nargs="+", default=CONTEXTS)
    parser.add_argument("--batches", type=int, nargs="+", default=BATCHES)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    if args.dp_size <= 0 or args.tp_size <= 0 or args.dp_size * args.tp_size != 16:
        parser.error("--dp-size * --tp-size must equal 16")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    matrix: list[dict[str, Any]] = []
    connector = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=None)
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        read_bufsize=32 << 20,
        max_line_size=32 << 20,
    ) as session:
        await wait_ready(session, args.base_url)
        for context in args.contexts:
            for batch in args.batches:
                if context <= 0 or batch <= 0 or batch % args.dp_size:
                    raise ValueError("contexts must be positive and batches positive multiples of dp-size")
                local_batch = batch // args.dp_size
                required_blocks = math.ceil((context + MAX_TOKENS) / BLOCK_SIZE) * local_batch
                required_dram_bytes = (context + MAX_TOKENS) * local_batch * MLA_KV_BYTES_PER_TOKEN
                point_path = args.output_dir / f"seq{context}_batch{batch}.json"
                if args.resume and point_path.exists():
                    previous = json.loads(point_path.read_text())
                    expected = {
                        "setup_id": args.setup_id,
                        "tested_commit": args.tested_commit,
                        "variant": args.variant,
                        "dp_size": args.dp_size,
                        "tp_size": args.tp_size,
                        "context_tokens": context,
                        "global_batch": batch,
                    }
                    mismatches = {
                        key: (previous.get(key), value)
                        for key, value in expected.items()
                        if previous.get(key) != value
                    }
                    if mismatches:
                        raise RuntimeError(f"refusing incompatible resume at {point_path}: {mismatches}")
                    if previous.get("status") == "ok":
                        matrix.append(previous)
                        print(json.dumps({"resume": str(point_path), "status": previous["status"]}), flush=True)
                        continue
                if required_blocks > args.capacity_blocks:
                    result = {
                        "status": "capacity_oom_not_run",
                        "variant": args.variant,
                        "setup_id": args.setup_id,
                        "tested_commit": args.tested_commit,
                        "tp_size": args.tp_size,
                        "context_tokens": context,
                        "global_batch": batch,
                        "local_batch_per_dp": local_batch,
                        "dp_size": args.dp_size,
                        "required_blocks_per_dp": required_blocks,
                        "capacity_blocks_per_dp": args.capacity_blocks,
                    }
                elif required_dram_bytes > args.dram_pool_gib_per_dp * (1 << 30):
                    result = {
                        "status": "dram_capacity_oom_not_run",
                        "variant": args.variant,
                        "setup_id": args.setup_id,
                        "tested_commit": args.tested_commit,
                        "tp_size": args.tp_size,
                        "context_tokens": context,
                        "global_batch": batch,
                        "local_batch_per_dp": local_batch,
                        "dp_size": args.dp_size,
                        "required_dram_bytes_per_dp": required_dram_bytes,
                        "dram_pool_bytes_per_dp": args.dram_pool_gib_per_dp * (1 << 30),
                    }
                else:
                    try:
                        result = await run_point(session, args, context, batch)
                        result["required_blocks_per_dp"] = required_blocks
                        result["capacity_blocks_per_dp"] = args.capacity_blocks
                        result["required_dram_bytes_per_dp"] = required_dram_bytes
                    except Exception as error:
                        result = {
                            "status": "error",
                            "variant": args.variant,
                            "setup_id": args.setup_id,
                            "tested_commit": args.tested_commit,
                            "tp_size": args.tp_size,
                            "context_tokens": context,
                            "global_batch": batch,
                            "local_batch_per_dp": local_batch,
                            "dp_size": args.dp_size,
                            "required_blocks_per_dp": required_blocks,
                            "capacity_blocks_per_dp": args.capacity_blocks,
                            "error": repr(error),
                        }
                point_path.write_text(json.dumps(result, indent=2) + "\n")
                matrix.append(result)
                print(json.dumps({
                    "context": context,
                    "batch": batch,
                    "status": result["status"],
                    "aggregate": result.get("aggregate"),
                }), flush=True)
                if result["status"] == "error" and not args.continue_on_error:
                    raise RuntimeError(f"benchmark point failed: {point_path}: {result['error']}")

    (args.output_dir / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
