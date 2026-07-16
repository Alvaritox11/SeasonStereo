"""
Generate a comparison table from summary.csv using the paper aggregation.

For each model and dataset:
  1. Keep rows with status == "ok" and a valid mae_filtered.
  2. For Omaha, drop complete pairs whose GT disparity max is > 192.
  3. Compute the median mae_filtered for each AOI.
  4. Report mean and sample std of those AOI medians.

Usage:
  python generate_table.py --summary outputs/test_eval/summary.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


METRIC = "mae_filtered"
GT_MAX_DISP = 192.0
GT_MAX_FILTER_DATASETS = {"omaha_diachronic", "omaha_synchronic"}

MODEL_ORDER = [
    "monster",
    "monster++",
    "..."
]

DATASET_ORDER = [
    "buenos_aires",
    "jax",
    "omaha_diachronic",
    "omaha_synchronic",
]


def sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return float("nan")
    return float(np.std(values, ddof=1))


def fmt_float(value: float, digits: int = 4) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def gt_disparity_path_from_left(left_path: str) -> Path | None:
    """Resolve .../L/<pair>.iio to the sibling .../disparity/<pair>.iio."""
    if not left_path:
        return None

    left = Path(left_path)
    candidates = []

    if left.parent.name == "L":
        candidates.append(left.parent.parent / "disparity" / left.name)

    left_str = str(left)
    if "/L/" in left_str:
        candidates.append(Path(left_str.replace("/L/", "/disparity/")))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_gt_max_disparity(disparity_path: Path) -> float:
    try:
        disparity = np.load(disparity_path)
    except Exception:
        import iio

        disparity = iio.read(str(disparity_path))
    return float(np.nanmax(disparity))


def should_keep_by_gt_disparity(row: dict, gt_max_cache: dict[str, float | None]) -> bool:
    dataset = row.get("dataset", "")
    if dataset not in GT_MAX_FILTER_DATASETS:
        return True

    left_path = row.get("left_path", "")
    pair_id = row.get("pair_id", "")
    cache_key = left_path or f"{dataset}:{pair_id}"

    if cache_key not in gt_max_cache:
        disparity_path = gt_disparity_path_from_left(left_path)
        gt_max_cache[cache_key] = (
            None if disparity_path is None else load_gt_max_disparity(disparity_path)
        )

    gt_max = gt_max_cache[cache_key]
    if gt_max is None:
        return True
    return gt_max <= GT_MAX_DISP


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: comparison_table_v2.csv next to summary.csv.",
    )
    args = parser.parse_args()

    output_path = args.output or args.summary.parent / "comparison_table_v2.csv"

    values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # values[model][dataset][aoi] = [pair mae_filtered, ...]
    errors = defaultdict(lambda: defaultdict(list))
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    gt_max_cache: dict[str, float | None] = {}

    with open(args.summary, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{args.summary} has no header")
        if METRIC not in reader.fieldnames:
            raise ValueError(f"Missing required column: {METRIC}")
        if "dataset" not in reader.fieldnames:
            raise ValueError("Missing required column: dataset")

        for row in reader:
            model = row.get("model", "")
            dataset = row.get("dataset", "")
            aoi = row.get("aoi", "") or "unknown_aoi"
            status = row.get("status", "")
            pair_id = row.get("pair_id", "?")

            if status == "error":
                errors[model][dataset].append(pair_id)
                counts[model][dataset]["error"] += 1
                continue

            if status != "ok" or not row.get(METRIC):
                counts[model][dataset]["skipped"] += 1
                continue

            try:
                metric_value = float(row[METRIC])
            except ValueError:
                counts[model][dataset]["skipped"] += 1
                continue

            if not should_keep_by_gt_disparity(row, gt_max_cache):
                counts[model][dataset]["filtered_gt_disp"] += 1
                continue

            values[model][dataset][aoi].append(metric_value)
            counts[model][dataset]["ok"] += 1

    if not values:
        print("No completed rows found in summary.csv.")
        return

    all_models = set(values) | set(errors)
    all_datasets = {dataset for by_dataset in values.values() for dataset in by_dataset}
    all_datasets |= {dataset for by_dataset in errors.values() for dataset in by_dataset}

    models = [model for model in MODEL_ORDER if model in all_models]
    models += sorted(all_models - set(MODEL_ORDER))

    datasets = [dataset for dataset in DATASET_ORDER if dataset in all_datasets]
    datasets += sorted(all_datasets - set(DATASET_ORDER))

    scores = defaultdict(dict)
    for model in models:
        for dataset in datasets:
            aoi_scores = [
                float(np.median(pair_values))
                for pair_values in values[model].get(dataset, {}).values()
                if pair_values
            ]
            if not aoi_scores:
                continue
            scores[model][dataset] = {
                "mean": float(np.mean(aoi_scores)),
                "std": sample_std(aoi_scores),
                "n_aoi": len(aoi_scores),
            }

    model_w = max(len(model) for model in models) + 2
    col_w = 24
    sep_w = model_w + (col_w + 2) * len(datasets)

    print(f"\n{'=' * sep_w}")
    print(f"  AOI SCORE -- mean(median {METRIC} per AOI) +/- std")
    print(f"  Omaha filter: drop complete pairs with GT max disparity > {GT_MAX_DISP:g}")
    print(f"{'=' * sep_w}")

    header = f"  {'model':{model_w}s}"
    for dataset in datasets:
        header += f"  {dataset[:col_w]:>{col_w}s}"
    print(header)
    print(f"  {'-' * model_w}" + (f"  {'-' * col_w}") * len(datasets))

    for model in models:
        row_text = f"  {model:{model_w}s}"
        for dataset in datasets:
            score = scores[model].get(dataset)
            if score:
                cell = f"{fmt_float(score['mean'])} +/- {fmt_float(score['std'])}"
            else:
                cell = "-"
            row_text += f"  {cell:>{col_w}s}"
        print(row_text)
    print(f"{'=' * sep_w}")

    print(f"\n{'=' * sep_w}")
    print("  COUNTS -- ok pairs / AOIs / filtered GT / errors")
    print(f"{'=' * sep_w}")
    print(header)
    print(f"  {'-' * model_w}" + (f"  {'-' * col_w}") * len(datasets))

    for model in models:
        row_text = f"  {model:{model_w}s}"
        for dataset in datasets:
            count = counts[model][dataset]
            score = scores[model].get(dataset)
            n_aoi = score["n_aoi"] if score else 0
            n_err = len(errors[model].get(dataset, []))
            total = count["ok"] + count["filtered_gt_disp"] + n_err
            cell = (
                f"{count['ok']} / {n_aoi} / {count['filtered_gt_disp']} / {n_err}"
                if total
                else "-"
            )
            row_text += f"  {cell:>{col_w}s}"
        print(row_text)
    print(f"{'=' * sep_w}")

    total_errors = sum(len(pairs) for by_dataset in errors.values() for pairs in by_dataset.values())
    if total_errors:
        print(f"\nFAILED PAIRS ({total_errors} total)")
        print("-" * 60)
        for model in models:
            for dataset in datasets:
                pairs = errors[model].get(dataset, [])
                if pairs:
                    print(f"[{model}] [{dataset}] {len(pairs)} errors:")
                    for pair_id in pairs:
                        print(f"    {pair_id}")

    csv_rows = []
    for model in models:
        out_row = {"model": model}
        for dataset in datasets:
            score = scores[model].get(dataset)
            count = counts[model][dataset]
            n_err = len(errors[model].get(dataset, []))

            out_row[f"{dataset}_mean"] = f"{score['mean']:.6f}" if score else ""
            out_row[f"{dataset}_std"] = f"{score['std']:.6f}" if score else ""
            out_row[f"{dataset}_n_aoi"] = score["n_aoi"] if score else 0
            out_row[f"{dataset}_n_ok_pairs"] = count["ok"]
            out_row[f"{dataset}_n_filtered_gt_disp"] = count["filtered_gt_disp"]
            out_row[f"{dataset}_n_err"] = n_err
        csv_rows.append(out_row)

    csv_cols = ["model"] + [
        f"{dataset}_{suffix}"
        for dataset in datasets
        for suffix in ("mean", "std", "n_aoi", "n_ok_pairs", "n_filtered_gt_disp", "n_err")
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
