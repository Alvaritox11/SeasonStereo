#!/usr/bin/env python3
"""Select high-quality pairs by metric and export their intersection."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional


DEFAULT_METRICS = ["dinov3", "combined_score_sum_rgb", "combined_score_sum_lab"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create selected-pair CSVs from per_sample_metrics.csv and export "
            "the intersection of the selected pairs across multiple metrics."
        )
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        required=True,
        help="Path to per_sample_metrics.csv written by compute_metrics.py.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output folder. Defaults to the metrics CSV parent directory.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metric columns to select and intersect.",
    )
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--max-epe", type=float, default=3.0)
    parser.add_argument("--epe-column", default="epe")
    parser.add_argument("--id-column", default="id")
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("data/synchronic_only"),
        help="Reference folder used to build left/right paths in the intersection CSV.",
    )
    parser.add_argument("--extension", default=".iio")
    parser.add_argument(
        "--use-metrics-paths",
        action="store_true",
        help="Use left_path/right_path from the metrics CSV instead of reference-root/L and reference-root/R.",
    )
    parser.add_argument(
        "--intersection-name",
        default=None,
        help="Output basename for the intersection CSV. Defaults to an automatic name.",
    )
    parser.add_argument(
        "--pairs-name",
        default="intersected_pairs.csv",
        help="Filename for the lightweight id,left,right intersection manifest.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value: str) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def is_finite(value: float) -> bool:
    return math.isfinite(value)


def metric_slug(metric: str) -> str:
    if metric.startswith("combined_score_sum_"):
        return "comb_score_" + metric.removeprefix("combined_score_sum_")
    if metric.startswith("combined_score_prod_"):
        return "comb_prod_" + metric.removeprefix("combined_score_prod_")
    return metric.replace("+", "plus").replace("-", "_")


def default_intersection_name(metrics: List[str]) -> str:
    combined_sum_suffixes = [
        metric.removeprefix("combined_score_sum_")
        for metric in metrics
        if metric.startswith("combined_score_sum_")
    ]
    other_parts = [
        metric_slug(metric)
        for metric in metrics
        if not metric.startswith("combined_score_sum_")
    ]

    parts = []
    if combined_sum_suffixes:
        parts.extend(["comb_score", *combined_sum_suffixes])
    parts.extend(other_parts)
    return "intersection_" + "_".join(parts)


def metric_value_column(metric: str) -> str:
    if metric == "dinov3":
        return "sim_val_DINOv3"
    if metric.startswith("combined_score_sum_"):
        return "sim_val_comb_score_sum_" + metric.removeprefix("combined_score_sum_")
    if metric.startswith("combined_score_prod_"):
        return "sim_val_comb_score_prod_" + metric.removeprefix("combined_score_prod_")
    return f"sim_val_{metric}"


def require_columns(rows: List[Dict[str, str]], columns: Iterable[str], csv_path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    available = set(rows[0].keys())
    missing = [column for column in columns if column not in available]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")


def select_top_pairs(
    rows: List[Dict[str, str]],
    metric: str,
    id_column: str,
    epe_column: str,
    max_epe: float,
    top_k: int,
) -> List[Dict[str, str]]:
    candidates = []

    for row in rows:
        if row.get("status", "ok") != "ok":
            continue

        metric_value = as_float(row.get(metric, ""))
        epe_value = as_float(row.get(epe_column, ""))
        pair_id = row.get(id_column, "")

        if not pair_id or not is_finite(metric_value) or not is_finite(epe_value):
            continue
        if epe_value > max_epe:
            continue

        candidates.append(
            {
                "id": pair_id,
                "similarity_metric": metric,
                "similarity_value": metric_value,
                "epe_value": epe_value,
            }
        )

    candidates.sort(key=lambda r: (-float(r["similarity_value"]), float(r["epe_value"]), str(r["id"])))

    selected = candidates[:top_k]
    for index, row in enumerate(selected, start=1):
        row["rank"] = index

    return selected


def write_selected(path: Path, rows: List[Dict[str, str]]) -> None:
    fieldnames = ["rank", "id", "similarity_metric", "similarity_value", "epe_value"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def pair_paths(row: Dict[str, str], args: argparse.Namespace) -> Dict[str, str]:
    pair_id = row[args.id_column]

    if args.use_metrics_paths:
        return {
            "left": row.get("left_path", ""),
            "right": row.get("right_path", ""),
        }

    return {
        "left": str(args.reference_root / "L" / f"{pair_id}{args.extension}"),
        "right": str(args.reference_root / "R" / f"{pair_id}{args.extension}"),
    }


def build_intersection(
    selected_by_metric: Dict[str, List[Dict[str, str]]],
    original_by_id: Dict[str, Dict[str, str]],
    metric_order: List[str],
    args: argparse.Namespace,
) -> List[Dict[str, str]]:
    selected_id_sets = [set(row["id"] for row in selected_by_metric[metric]) for metric in metric_order]
    common_ids = set.intersection(*selected_id_sets) if selected_id_sets else set()

    selected_lookup = {
        metric: {row["id"]: row for row in selected_rows}
        for metric, selected_rows in selected_by_metric.items()
    }

    output_rows = []
    for pair_id in common_ids:
        source_row = original_by_id[pair_id]
        paths = pair_paths(source_row, args)

        row = {
            "id": pair_id,
            "left": paths["left"],
            "right": paths["right"],
            "epe_value": selected_lookup[metric_order[0]][pair_id]["epe_value"],
        }

        for metric in metric_order:
            row[metric_value_column(metric)] = selected_lookup[metric][pair_id]["similarity_value"]

        output_rows.append(row)

    output_rows.sort(
        key=lambda row: (
            *[-float(row[metric_value_column(metric)]) for metric in metric_order],
            float(row["epe_value"]),
            row["id"],
        )
    )
    return output_rows


def write_intersection(path: Path, rows: List[Dict[str, str]], metric_order: List[str]) -> None:
    fieldnames = ["id", "left", "right", "epe_value"]
    fieldnames.extend(metric_value_column(metric) for metric in metric_order)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_pairs_manifest(path: Path, rows: List[Dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "left", "right"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"id": row["id"], "left": row["left"], "right": row["right"]})


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or args.metrics_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.metrics_csv)
    require_columns(rows, [args.id_column, args.epe_column, *args.metrics], args.metrics_csv)
    if args.use_metrics_paths:
        require_columns(rows, ["left_path", "right_path"], args.metrics_csv)

    original_by_id: Dict[str, Dict[str, str]] = {}
    for row in rows:
        original_by_id.setdefault(row[args.id_column], row)

    selected_by_metric: Dict[str, List[Dict[str, str]]] = {}
    for metric in args.metrics:
        selected = select_top_pairs(
            rows=rows,
            metric=metric,
            id_column=args.id_column,
            epe_column=args.epe_column,
            max_epe=args.max_epe,
            top_k=args.top_k,
        )
        selected_by_metric[metric] = selected
        selected_path = out_dir / f"selected_pairs__{metric}.csv"
        write_selected(selected_path, selected)
        print(f"Wrote {len(selected)} selected pairs: {selected_path}")

    intersection_rows = build_intersection(
        selected_by_metric=selected_by_metric,
        original_by_id=original_by_id,
        metric_order=list(args.metrics),
        args=args,
    )

    intersection_name = args.intersection_name
    if intersection_name is None:
        intersection_name = default_intersection_name(list(args.metrics))

    intersection_path = out_dir / f"{intersection_name}.csv"
    write_intersection(intersection_path, intersection_rows, list(args.metrics))
    print(f"Wrote {len(intersection_rows)} intersected pairs: {intersection_path}")

    pairs_path = out_dir / args.pairs_name
    write_pairs_manifest(pairs_path, intersection_rows)
    print(f"Wrote lightweight pairs manifest: {pairs_path}")


if __name__ == "__main__":
    main()
