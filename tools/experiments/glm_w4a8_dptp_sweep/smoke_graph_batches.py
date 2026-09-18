#!/usr/bin/env python3
"""Check all graph batch buckets without requiring DummyKV token determinism."""

import argparse
import asyncio
import json
import time

import aiohttp


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-batch", type=int, required=True)
    parser.add_argument("--dp-size", type=int, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--base-url", default="http://127.0.0.1:18300")
    args = parser.parse_args()
    if args.dp_size <= 0 or 16 % args.dp_size or args.global_batch <= 0 or args.global_batch % args.dp_size:
        parser.error("--dp-size must divide 16 and --global-batch must be a positive multiple of dp-size")
    if args.prompt_tokens <= 0:
        parser.error("--prompt-tokens must be positive")
    payload = {
        "model": "GLM-5",
        "prompt": [100] * args.prompt_tokens,
        "max_tokens": 12,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": False,
        "return_token_ids": True,
    }
    timeout = aiohttp.ClientTimeout(total=300, sock_connect=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        gate = asyncio.Event()

        async def request(index: int) -> dict:
            await gate.wait()
            async with session.post(
                f"{args.base_url}/v1/completions",
                json=payload,
                headers={"X-data-parallel-rank": str(index % args.dp_size)},
            ) as response:
                body = await response.json()
                choice = body.get("choices", [{}])[0]
                return {
                    "index": index,
                    "status": response.status,
                    "finish_reason": choice.get("finish_reason"),
                    "token_ids": choice.get("token_ids"),
                }

        started = time.perf_counter()
        tasks = [asyncio.create_task(request(i)) for i in range(args.global_batch)]
        await asyncio.sleep(0)
        gate.set()
        results = await asyncio.gather(*tasks)

    token_complete = all(
        item["status"] == 200
        and item["finish_reason"] == "length"
        and len(item["token_ids"] or []) == 12
        for item in results
    )
    lane_match = [
        len({tuple(results[rank + lane * args.dp_size]["token_ids"] or []) for rank in range(args.dp_size)}) == 1
        for lane in range(args.global_batch // args.dp_size)
    ]
    print(json.dumps({
        "token_complete": token_complete,
        "global_batch": args.global_batch,
        "dp_size": args.dp_size,
        "local_batch_per_dp": args.global_batch // args.dp_size,
        "same_tokens_across_dp_by_lane": lane_match,
        "elapsed_s": time.perf_counter() - started,
        "results": results,
    }, ensure_ascii=False, indent=2))
    if not token_complete:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
