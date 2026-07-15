import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from metrics_common import (
    DinoV3Metric,
    compute_color_distance_stats,
    compute_epe,
    compute_ssim_ab,
    compute_ssim_lab,
    compute_ssim_rgb,
    load_disp,
    load_image,
    prepare_warped_pair,
)


def read_pairs_csv(pairs_list: Path) -> List[Dict[str, Path]]:
    samples = []
    with open(pairs_list, "r", newline="", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("left,right"):
                continue

            parts = line.split(",")
            if len(parts) != 2:
                raise ValueError(
                    f"Line {line_idx} in {pairs_list} must have exactly 2 columns: left,right"
                )

            left_path = Path(parts[0])
            right_path = Path(parts[1])
            sample_id = left_path.stem

            samples.append(
                {
                    "id": sample_id,
                    "left_path": left_path,
                    "right_path": right_path,
                }
            )
    return samples


def resolve_gt_disp_path(gt_disp_dir: Path, sample_id: str) -> Path:
    return gt_disp_dir / f"{sample_id}.iio"


def resolve_pred_disp_path(pred_dir: Path, sample_id: str) -> Path:
    return pred_dir / f"{sample_id}.iio"


def finalize_color_scores(rows: List[Dict], diff_max_percentile: float, family: str) -> float:
    base_key = f"color_distance_{family}"
    vals = np.array(
        [
            r[base_key]
            for r in rows
            if base_key in r and np.isfinite(r[base_key])
        ],
        dtype=np.float32,
    )

    if vals.size == 0:
        return float("nan")

    diff_max = float(np.percentile(vals, diff_max_percentile))
    if diff_max <= 0:
        diff_max = 1.0

    for r in rows:
        if base_key not in r or not np.isfinite(r[base_key]) or f"ssim_{family}" not in r:
            continue

        cd = min(float(r[base_key]), diff_max)
        color_score = 1.0 - (cd / diff_max)

        r[f"diff_max_{family}"] = diff_max
        r[f"color_score_{family}"] = color_score
        r[f"combined_score_sum_{family}"] = 0.5 * r[f"ssim_{family}"] + 0.5 * color_score
        r[f"combined_score_prod_{family}"] = r[f"ssim_{family}"] * color_score

    return diff_max


def summarize_rows(rows: List[Dict]) -> Dict:
    summary = {}
    if not rows:
        return summary

    skip = {
        "id",
        "left_path",
        "right_path",
        "gt_disp_path",
        "pred_disp_path",
        "status",
        "error_message",
        "warp_source",
        "prediction_name",
        "experiment_name",
    }

    numeric_keys = [k for k in rows[0].keys() if k not in skip]

    for k in numeric_keys:
        vals = np.array(
            [r[k] for r in rows if k in r and isinstance(r[k], (int, float)) and np.isfinite(r[k])],
            dtype=np.float64,
        )
        if vals.size == 0:
            continue

        summary[k] = {
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "std": float(vals.std()),
        }

    summary["num_samples_total"] = len(rows)
    summary["num_samples_ok"] = sum(r.get("status", "ok") == "ok" for r in rows)
    summary["num_samples_error"] = sum(r.get("status", "ok") == "error" for r in rows)
    return summary


def infer_prediction_name(pred_dir: Path) -> str:
    return pred_dir.name


def build_experiment_name(args, pred_dir: Path) -> str:
    if args.experiment_name != "auto":
        return args.experiment_name
    pred_name = infer_prediction_name(pred_dir)
    return f"pred-{pred_name}__warp-{args.warp_source}__metric-{args.similarity_metric}"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pairs_list", type=str, required=True)
    parser.add_argument("--gt_disp_dir", type=str, required=True)
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, default="auto")

    parser.add_argument(
        "--similarity_metric",
        type=str,
        default="all",
        choices=[
            "ssim_rgb",
            "ssim_lab",
            "ssim_ab",
            "ssim_lab_color",
            "ssim_ab_color",
            "ssim_rgb_color",
            "dinov3",
            "all",
        ],
    )
    parser.add_argument(
        "--warp_source",
        type=str,
        default="gt",
        choices=["gt", "pred"],
    )

    parser.add_argument("--diff_max_percentile", type=float, default=95.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--require_positive_gt", action="store_true")
    parser.add_argument("--require_positive_warp", action="store_true")

    parser.add_argument("--dinov3_repo_path", type=str, default=None)
    parser.add_argument("--dinov3_ckpt_path", type=str, default=None)
    parser.add_argument("--dinov3_model_name", type=str, default="facebook/dinov3-vitl16-pretrain-sat493m")
    parser.add_argument("--dinov3_arch", type=str, default="dinov3_vitl16")

    return parser.parse_args()


def main():
    args = parse_args()

    pairs_list = Path(args.pairs_list)
    gt_disp_dir = Path(args.gt_disp_dir)
    pred_dir = Path(args.pred_dir)
    out_root = Path(args.out_dir)

    experiment_name = build_experiment_name(args, pred_dir)
    out_dir = out_root / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = read_pairs_csv(pairs_list)

    dino_metric = None
    if args.similarity_metric in ("dinov3", "all"):
        dino_metric = DinoV3Metric(
            model_name=args.dinov3_model_name,
            device=args.device,
            repo_path=args.dinov3_repo_path,
            ckpt_path=args.dinov3_ckpt_path,
            arch=args.dinov3_arch,
        )

    rows: List[Dict] = []

    for sample in tqdm(samples, desc="Computing metrics"):
        sid = sample["id"]
        left_path = sample["left_path"]
        right_path = sample["right_path"]
        gt_disp_path = resolve_gt_disp_path(gt_disp_dir, sid)
        pred_disp_path = resolve_pred_disp_path(pred_dir, sid)

        row = {
            "id": sid,
            "left_path": str(left_path),
            "right_path": str(right_path),
            "gt_disp_path": str(gt_disp_path),
            "pred_disp_path": str(pred_disp_path),
            "warp_source": args.warp_source,
            "prediction_name": infer_prediction_name(pred_dir),
            "experiment_name": experiment_name,
            "status": "ok",
            "error_message": "",
        }

        try:
            left_img = load_image(left_path)
            right_img = load_image(right_path)
            gt_disp = load_disp(gt_disp_path)
            pred_disp = load_disp(pred_disp_path)

            row["epe"] = compute_epe(
                gt_disp=gt_disp,
                pred_disp=pred_disp,
                require_positive_gt=args.require_positive_gt,
            )

            warp_disp = gt_disp if args.warp_source == "gt" else pred_disp

            prepared = prepare_warped_pair(
                left_img=left_img,
                right_img=right_img,
                warp_disp=warp_disp,
                gt_disp=gt_disp,
                require_positive_warp=args.require_positive_warp,
                require_positive_gt=args.require_positive_gt,
            )

            left = prepared["left"]
            left_hat = prepared["left_hat"]
            final_mask = prepared["final_mask"]

            row["valid_ratio"] = float(final_mask.mean())

            if args.similarity_metric in ("ssim_rgb", "ssim_rgb_color", "all"):
                row["ssim_rgb"] = compute_ssim_rgb(left, left_hat, final_mask)

            if args.similarity_metric in ("ssim_rgb_color", "all"):
                row.update(compute_color_distance_stats(left, left_hat, final_mask, mode="rgb_l2"))

            if args.similarity_metric in ("ssim_lab", "ssim_lab_color", "all"):
                row["ssim_lab"] = compute_ssim_lab(left, left_hat, final_mask)

            if args.similarity_metric in ("ssim_ab", "ssim_ab_color", "all"):
                row["ssim_ab"] = compute_ssim_ab(left, left_hat, final_mask)

            if args.similarity_metric in ("ssim_lab_color", "all"):
                row.update(compute_color_distance_stats(left, left_hat, final_mask, mode="lab_l2"))

            if args.similarity_metric in ("ssim_ab_color", "all"):
                row.update(compute_color_distance_stats(left, left_hat, final_mask, mode="ab_l2"))

            if args.similarity_metric == "all":
                row.update(compute_color_distance_stats(left, left_hat, final_mask, mode="ciede2000"))

            if args.similarity_metric in ("dinov3", "all"):
                row["dinov3"] = dino_metric.compute(left, left_hat, final_mask)

        except Exception as e:
            row["status"] = "error"
            row["error_message"] = str(e)

        rows.append(row)

    if args.similarity_metric in ("ssim_lab_color", "all"):
        finalize_color_scores(rows, diff_max_percentile=args.diff_max_percentile, family="lab")

    if args.similarity_metric in ("ssim_ab_color", "all"):
        finalize_color_scores(rows, diff_max_percentile=args.diff_max_percentile, family="ab")
    
    if args.similarity_metric in ("ssim_rgb_color", "all"):
        finalize_color_scores(rows, diff_max_percentile=args.diff_max_percentile, family="rgb")

    csv_path = out_dir / "per_sample_metrics.csv"
    if rows:
        fieldnames = sorted({k for row in rows for k in row.keys()})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    summary = summarize_rows(rows)

    run_config = vars(args).copy()
    run_config["pairs_list"] = str(pairs_list)
    run_config["gt_disp_dir"] = str(gt_disp_dir)
    run_config["pred_dir"] = str(pred_dir)
    run_config["out_dir"] = str(out_dir)
    run_config["experiment_name"] = experiment_name
    run_config["prediction_name"] = infer_prediction_name(pred_dir)

    with open(out_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "summary_metrics.txt", "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()