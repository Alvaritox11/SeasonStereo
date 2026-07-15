#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_MODELS = ["monster", "monster++", "foundationstereo"]
DEFAULT_WARP_SOURCES = ["gt", "pred"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch all stereo metric experiment combinations automatically."
    )

    parser.add_argument("--compute_script", type=str, default="compute_metrics.py",
                        help="Path to the metrics computation script.")
    parser.add_argument("--pairs_list", type=str, required=True,
                        help="Path to pairs CSV/list.")
    parser.add_argument("--gt_disp_dir", type=str, required=True,
                        help="Directory with GT disparities.")
    parser.add_argument("--predictions_root", type=str, required=True,
                        help="Root directory containing one subfolder per model prediction.")
    parser.add_argument("--results_root", type=str, required=True,
                        help="Root directory where experiment folders will be created.")

    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="Prediction model folders to use under predictions_root.")
    parser.add_argument("--warp_sources", nargs="+", default=DEFAULT_WARP_SOURCES,
                        choices=["gt", "pred"],
                        help="Warp sources to evaluate.")
    parser.add_argument("--similarity_metric", type=str, default="all",
                        choices=["ssim_rgb", "ssim_lab", "ssim_ab", "ssim_lab_color", "ssim_ab_color", "ssim_rgb_color", "dinov3", "all"],
                        help="Metric mode passed to compute_metrics.py.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device passed to compute_metrics.py.")
    # parser.add_argument("--color_distance_mode", type=str, default="ab_l2",
    #                     choices=["ab_l2", "lab_l2"],
    #                     help="Color distance mode passed through.")
    parser.add_argument("--diff_max_percentile", type=float, default=95.0,
                        help="Dataset-level percentile for color normalization.")
    parser.add_argument("--require_positive_gt", action="store_true",
                        help="Pass through to compute_metrics.py.")
    parser.add_argument("--require_positive_warp", action="store_true",
                        help="Pass through to compute_metrics.py.")

    # DINO options: keep optional if compute_metrics supports them
    parser.add_argument("--dinov3_repo_path", type=str, default=None)
    parser.add_argument("--dinov3_ckpt_path", type=str, default=None)
    parser.add_argument("--dinov3_arch", type=str, default=None)

    parser.add_argument("--python_bin", type=str, default=sys.executable,
                        help="Python executable used to launch each run.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run experiments even if the output folder already exists.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing them.")
    parser.add_argument("--stop_on_error", action="store_true",
                        help="Stop immediately if one experiment fails.")

    return parser.parse_args()

def build_experiment_name(model_name: str, warp_source: str, similarity_metric: str) -> str:
    safe_model = model_name.replace("+", "plus")
    return f"pred-{safe_model}__warp-{warp_source}__metric-{similarity_metric}"


def build_command(args, pred_dir: Path, experiment_name: str, warp_source: str):
    cmd = [
        args.python_bin,
        args.compute_script,
        "--pairs_list", args.pairs_list,
        "--gt_disp_dir", args.gt_disp_dir,
        "--pred_dir", str(pred_dir),
        "--out_dir", args.results_root,
        "--similarity_metric", args.similarity_metric,
        "--warp_source", warp_source,
        "--device", args.device,
        # "--color_distance_mode", args.color_distance_mode,
        "--diff_max_percentile", str(args.diff_max_percentile),
        "--experiment_name", experiment_name,
    ]

    if args.require_positive_gt:
        cmd.append("--require_positive_gt")
    if args.require_positive_warp:
        cmd.append("--require_positive_warp")

    if args.dinov3_repo_path:
        cmd.extend(["--dinov3_repo_path", args.dinov3_repo_path])
    if args.dinov3_ckpt_path:
        cmd.extend(["--dinov3_ckpt_path", args.dinov3_ckpt_path])
    if args.dinov3_arch:
        cmd.extend(["--dinov3_arch", args.dinov3_arch])

    return cmd


def write_launcher_manifest(results_root: Path, runs):
    results_root.mkdir(parents=True, exist_ok=True)
    manifest_path = results_root / "launcher_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2)
    return manifest_path


def main():
    args = parse_args()

    predictions_root = Path(args.predictions_root)
    results_root = Path(args.results_root)
    compute_script = Path(args.compute_script)

    if not compute_script.exists():
        raise FileNotFoundError(f"compute script not found: {compute_script}")

    planned_runs = []
    for model_name in args.models:
        pred_dir = predictions_root / model_name
        if not pred_dir.exists():
            print(f"[WARN] prediction directory does not exist, skipping: {pred_dir}")
            continue

        for warp_source in args.warp_sources:
            experiment_name = build_experiment_name(
                model_name=model_name,
                warp_source=warp_source,
                similarity_metric=args.similarity_metric,
            )
            experiment_dir = results_root / experiment_name

            planned_runs.append({
                "model": model_name,
                "warp_source": warp_source,
                "pred_dir": str(pred_dir),
                "experiment_name": experiment_name,
                "experiment_dir": str(experiment_dir),
            })

    if not planned_runs:
        raise RuntimeError("No runs were planned. Check predictions_root and models.")

    manifest_path = write_launcher_manifest(results_root, planned_runs)
    print(f"[INFO] wrote manifest to: {manifest_path}")

    num_ok = 0
    num_failed = 0
    num_skipped = 0

    for run in planned_runs:
        experiment_dir = Path(run["experiment_dir"])

        if experiment_dir.exists() and not args.overwrite:
            print(f"[SKIP] {run['experiment_name']} already exists")
            num_skipped += 1
            continue

        cmd = build_command(
            args=args,
            pred_dir=Path(run["pred_dir"]),
            experiment_name=run["experiment_name"],
            warp_source=run["warp_source"],
        )

        print("\n" + "=" * 80)
        print(f"[RUN]  {run['experiment_name']}")
        print(f"[MODEL] {run['model']}")
        print(f"[WARP] {run['warp_source']}")
        print("[CMD]  " + " ".join(cmd))

        if args.dry_run:
            continue

        completed = subprocess.run(cmd)
        if completed.returncode == 0:
            print(f"[OK]   {run['experiment_name']}")
            num_ok += 1
        else:
            print(f"[FAIL] {run['experiment_name']} (code {completed.returncode})")
            num_failed += 1
            if args.stop_on_error:
                break

    print("\n" + "=" * 80)
    print("[SUMMARY]")
    print(f"planned : {len(planned_runs)}")
    print(f"ok      : {num_ok}")
    print(f"failed  : {num_failed}")
    print(f"skipped : {num_skipped}")


if __name__ == "__main__":
    main()
