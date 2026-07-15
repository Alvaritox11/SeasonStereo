#!/usr/bin/env python3
"""Build a left/right pair CSV from a rectified reference directory."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a pairs CSV with columns left,right.")
    parser.add_argument("--reference-root", type=Path, default=Path("data/synchronic_only"),
                        help="Directory containing L/ and R/ subfolders.")
    parser.add_argument("--left-dir", type=Path, default=None)
    parser.add_argument("--right-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("pairs.csv"))
    parser.add_argument("--extension", default=".iio")
    parser.add_argument("--limit", type=int, default=0, help="Debug limit. Use 0 for all pairs.")
    return parser.parse_args()


def main():
    args = parse_args()
    left_dir = args.left_dir or args.reference_root / "L"
    right_dir = args.right_dir or args.reference_root / "R"

    left_files = sorted(left_dir.glob(f"*{args.extension}"))
    right_files = sorted(right_dir.glob(f"*{args.extension}"))

    if args.limit > 0:
        left_files = left_files[:args.limit]
        right_files = right_files[:args.limit]

    if len(left_files) != len(right_files):
        raise RuntimeError(f"Mismatched file counts: left={len(left_files)} right={len(right_files)}")

    mismatches = [
        (left.name, right.name)
        for left, right in zip(left_files, right_files)
        if left.name != right.name
    ]
    if mismatches:
        preview = "\n".join(f"  left={left} right={right}" for left, right in mismatches[:5])
        raise RuntimeError(f"Found {len(mismatches)} misaligned pairs:\n{preview}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["left", "right"])
        writer.writeheader()
        for left, right in zip(left_files, right_files):
            writer.writerow({"left": str(left), "right": str(right)})

    print(f"Wrote {len(left_files)} pairs to {args.output}")


if __name__ == "__main__":
    main()
