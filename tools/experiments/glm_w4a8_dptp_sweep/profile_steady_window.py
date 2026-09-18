#!/usr/bin/env python3
"""Bracket a steady decode window with vLLM's existing NPU profiler API."""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import aiohttp


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-batch", type=int, required=True)
    parser.add_argument("--dp-size", type=int, required=True)
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--warmup-tokens", type=int, default=50)
    parser.add_argument("--profile-tokens", type=int, default=25)
    parser.add_argument("--max-tokens", type=int, default=125)
    parser.add_argument("--base-url", default="http://127.0.0.1:18300")
    args = parser.parse_args()
    if args.dp_size <= 0 or args.global_batch <= 0 or args.global_batch % args.dp_size:
        parser.error("global batch must be a positive multiple of DP size")
    if args.prompt_tokens <= 0 or args.warmup_tokens < 0 or args.profile_tokens < 10:
        parser.error("prompt must be positive and profile window at least 10 tokens")
    if args.max_tokens <= args.warmup_tokens + args.profile_tokens + 10:
        parser.error("max tokens must leave more than 10 steps after the profile window")

    counts = [0] * args.global_batch
    completions = [False] * args.global_batch
    gate = asyncio.Event()
    payload = {
        "model": "GLM-5",
        "prompt": [100] * args.prompt_tokens,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "return_token_ids": True,
    }
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=None)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        async def request(index: int) -> None:
            await gate.wait()
            async with session.post(
                f"{args.base_url}/v1/completions",
                json=payload,
                headers={"X-data-parallel-rank": str(index % args.dp_size)},
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"request {index}: HTTP {response.status}: {await response.text()}")
                async for raw_line in response.content:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    data = json.loads(line[6:])
                    if data.get("choices"):
                        counts[index] += len(data["choices"][0].get("token_ids") or [])
            completions[index] = True

        tasks = [asyncio.create_task(request(index)) for index in range(args.global_batch)]
        await asyncio.sleep(0)
        gate.set()

        async def wait_for_every_request(target: int) -> None:
            while min(counts) < target:
                for task in tasks:
                    if task.done() and task.exception() is not None:
                        raise task.exception()
                if any(completions):
                    raise RuntimeError(
                        f"a request finished before every stream reached token {target}: "
                        f"min={min(counts)} max={max(counts)}"
                    )
                await asyncio.sleep(0.02)

        profile_active = False
        started = time.perf_counter()
        try:
            await wait_for_every_request(args.warmup_tokens)
            warmup_done = time.perf_counter()
            async with session.post(f"{args.base_url}/start_profile") as response:
                if response.status != 200:
                    raise RuntimeError(f"start_profile HTTP {response.status}: {await response.text()}")
            profile_active = True
            profile_started = time.perf_counter()
            await wait_for_every_request(args.warmup_tokens + args.profile_tokens)
            async with session.post(f"{args.base_url}/stop_profile") as response:
                if response.status != 200:
                    raise RuntimeError(f"stop_profile HTTP {response.status}: {await response.text()}")
            profile_active = False
            profile_stopped = time.perf_counter()
            await asyncio.gather(*tasks)
        finally:
            if profile_active:
                async with session.post(f"{args.base_url}/stop_profile") as response:
                    print(json.dumps({"emergency_stop_profile_http_status": response.status}), flush=True)
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    if counts != [args.max_tokens] * args.global_batch:
        raise RuntimeError(f"incomplete request counts: min={min(counts)} max={max(counts)}")
    print(json.dumps({
        "global_batch": args.global_batch,
        "dp_size": args.dp_size,
        "prompt_tokens": args.prompt_tokens,
        "warmup_tokens": args.warmup_tokens,
        "profile_tokens_at_least_per_request": args.profile_tokens,
        "max_tokens": args.max_tokens,
        "warmup_elapsed_s": warmup_done - started,
        "profile_api_window_s": profile_stopped - profile_started,
        "total_elapsed_s": time.perf_counter() - started,
        "min_tokens_at_end": min(counts),
        "max_tokens_at_end": max(counts),
    }, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
