#!/usr/bin/env python3
"""Record all-die AI Core usage during marked stage graph-replay windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time


def parse_npu_smi(output: str) -> dict[int, int]:
    usage = {}
    for line in output.splitlines():
        fields = line.split("|")
        if len(fields) < 5 or ":" not in fields[2]:
            continue
        identifiers = fields[1].split()
        values = fields[3].split()
        if len(identifiers) != 2 or not all(part.isdigit() for part in identifiers):
            continue
        if not values or not values[0].isdigit():
            continue
        physical_id = int(identifiers[1])
        usage[physical_id] = int(values[0])
    return usage


def parse_processes(output: str) -> list[dict]:
    processes = []
    for line in output.splitlines():
        fields = line.split("|")
        if len(fields) < 6 or not fields[2].strip().isdigit():
            continue
        identifiers = fields[1].split()
        if len(identifiers) != 2 or not all(value.isdigit() for value in identifiers):
            continue
        # The process table uses package ID plus local chip 0/1, whereas the
        # status table above prints the physical die ID 0..15 directly.
        processes.append({"die": 2 * int(identifiers[0]) + int(identifiers[1]),
                          "pid": int(fields[2].strip()),
                          "name": fields[3].strip()})
    return processes


def active_marker(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--stop", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("a") as stream:
        while not args.stop.exists():
            before = active_marker(args.marker)
            if before is None:
                time.sleep(0.2)
                continue
            started = time.monotonic()
            result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True)
            ended = time.monotonic()
            after = active_marker(args.marker)
            if result.returncode != 0 or after != before:
                continue
            usage = parse_npu_smi(result.stdout)
            processes = parse_processes(result.stdout)
            if len(usage) != 16 or set(usage) != set(range(16)):
                raise RuntimeError(f"expected A3 die 0..15; saw {sorted(usage)}")
            stream.write(json.dumps({
                "stage": before["stage"], "batch": before["batch"],
                "mode": before["mode"],
                "monotonic_start": started, "monotonic_end": ended,
                "aicore_percent_by_die": usage,
                "processes": processes,
            }) + "\n")
            stream.flush()


if __name__ == "__main__":
    main()
