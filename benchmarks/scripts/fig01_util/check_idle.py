#!/usr/bin/env python3
"""Fail if any process already owns one of the 16 A3 NPU dies."""

import sys
from pathlib import Path


def main():
    output = Path(sys.argv[1]).read_text()
    processes = []
    for line in output.splitlines():
        fields = line.split("|")
        if len(fields) >= 6 and fields[2].strip().isdigit():
            processes.append((fields[1].strip(), fields[2].strip(), fields[3].strip()))
    if processes:
        for die, pid, name in processes:
            print(f"occupied die={die} pid={pid} process={name}", file=sys.stderr)
        raise SystemExit(1)
    print("all 16 A3 dies report no owning processes")


if __name__ == "__main__":
    main()
