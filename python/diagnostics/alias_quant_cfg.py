#!/usr/bin/env python3
"""Clone a quantization-cfg tensor block under a second tensor name."""

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--from-name", required=True)
    parser.add_argument("--to-name", required=True)
    args = parser.parse_args()

    lines = args.source.read_text(encoding="utf-8").splitlines()
    source_header = f"    {args.from_name}:"
    target_header = f"    {args.to_name}:"

    if target_header in lines:
        raise SystemExit(f"target already exists: {args.to_name}")
    try:
        start = lines.index(source_header)
    except ValueError as exc:
        raise SystemExit(f"source not found: {args.from_name}") from exc

    end = start + 1
    while end < len(lines):
        line = lines[end]
        if line.startswith("    ") and not line.startswith("     "):
            break
        end += 1

    block = lines[start:end]
    block[0] = target_header
    output_lines = [*lines[:end], *block, *lines[end:]]
    args.output.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print(f"aliased {args.from_name} -> {args.to_name} in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
