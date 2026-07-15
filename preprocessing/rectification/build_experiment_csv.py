#!/usr/bin/env python3
"""Create a minimal SeasonStereo experiment CSV from generated train images."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a small experiment CSV with a filename column for SeasonStereo training."
    )
    parser.add_argument("--train-root", type=Path, default=Path("data/diachronic-stereo-synthetic/train"))
    parser.add_argument("--output", type=Path, default=Path("data/diachronic-stereo-synthetic/experiments/smoke_train.csv"))
    parser.add_argument("--limit", type=int, default=0, help="Use 0 for all generated train samples.")
    parser.add_argument("--extension", default=".npy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    left_dir = args.train_root / "L"

    samples = sorted(left_dir.glob(f"*{args.extension}"))
    if args.limit > 0:
        samples = samples[:args.limit]
    if not samples:
        raise FileNotFoundError(f"No samples found in {left_dir} with extension {args.extension}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename"])
        writer.writeheader()
        for sample in samples:
            writer.writerow({"filename": sample.name})

    print(f"Wrote {len(samples)} rows to {args.output}")


if __name__ == "__main__":
    main()
