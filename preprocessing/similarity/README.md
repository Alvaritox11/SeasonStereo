# Similarity Metrics

This folder computes image-similarity metrics between rectified left images and right images warped into the left view. It can also generate disparity maps with the supported stereo backbones before metric computation.

## Files

- `generate_pairs_list.py`: builds a `left,right` CSV from a reference stereo folder.
- `generate_disp_maps.py`: predicts disparity maps for each pair.
- `compute_metrics.py`: computes EPE, SSIM variants, color-distance scores, and optional DINOv3 similarity.
- `run_all_metrics_experiments.py`: launches metric combinations over several prediction folders.
- `select_pair_intersections.py`: selects high-quality pairs by metric and exports their intersection.
- `metrics_common.py`: shared metric and warping utilities.
- `similarity.ipynb`: small notebook for inspecting one warped pair and metric row.

## Pair List

Run commands from the repository root.

```bash
python preprocessing/similarity/generate_pairs_list.py \
  --reference-root data/synchronic_only \
  --output outputs/similarity/pairs.csv
```

The CSV contains:

```text
left,right
path/to/left.iio,path/to/right.iio
```

For a small smoke test, add `--limit 5`.

## Generate Disparities

MonSter / MonSter++:

```bash
python preprocessing/similarity/generate_disp_maps.py \
  --pairs_list outputs/similarity/pairs.csv \
  --out_dir outputs/similarity/disparities \
  --model monster++ \
  --ckpt_path checkpoints/monster++-mix_all.pth \
  --depth_anything_v2_path checkpoints/depth_anything_v2_vitl.pth \
  --device cuda \
  --max_side 0 \
  --limit 5
```

Use `--max_side 0` when the disparities will be used as SeasonStereo training pseudo-GT, because the output disparity shape must match the rectified train images and masks.

FoundationStereo:

```bash
python preprocessing/similarity/generate_disp_maps.py \
  --pairs_list outputs/similarity/pairs.csv \
  --out_dir outputs/similarity/disparities \
  --model foundationstereo \
  --ckpt_path checkpoints/foundationstereo/model_best_bp2.pth \
  --device cuda
```

## Compute Metrics

```bash
python preprocessing/similarity/compute_metrics.py \
  --pairs_list outputs/similarity/pairs.csv \
  --gt_disp_dir data/synchronic_only/disparity \
  --pred_dir outputs/similarity/disparities/monster++ \
  --out_dir outputs/similarity/results \
  --similarity_metric all \
  --warp_source pred \
  --device cuda
```

For DINOv3 without network downloads, pass a local repo and checkpoint:

```bash
python preprocessing/similarity/compute_metrics.py \
  --pairs_list outputs/similarity/pairs.csv \
  --gt_disp_dir data/synchronic_only/disparity \
  --pred_dir outputs/similarity/disparities/monster++ \
  --out_dir outputs/similarity/results \
  --similarity_metric dinov3 \
  --dinov3_repo_path thirdparty/dinov3 \
  --dinov3_ckpt_path checkpoints/dinov3_sat493m.pth
```

## Batch Launcher

```bash
python preprocessing/similarity/run_all_metrics_experiments.py \
  --pairs_list outputs/similarity/pairs.csv \
  --gt_disp_dir data/synchronic_only/disparity \
  --predictions_root outputs/similarity/disparities \
  --results_root outputs/similarity/results \
  --models monster++ foundationstereo \
  --warp_sources gt pred \
  --similarity_metric all
```

## Pair Selection And Intersections

After `compute_metrics.py` writes `per_sample_metrics.csv`, use this script to reproduce the pair-selection logic from the analysis notebooks. By default it selects the top 1000 pairs with `epe <= 3.0` for each metric, then writes the intersection of those selected sets.

```bash
python preprocessing/similarity/select_pair_intersections.py \
  --metrics-csv outputs/similarity/results/pred-monsterplusplus__warp-pred__metric-all/per_sample_metrics.csv \
  --out-dir outputs/similarity/results/pred-monsterplusplus__warp-pred__metric-all \
  --reference-root data/synchronic_only \
  --metrics dinov3 combined_score_sum_rgb combined_score_sum_lab \
  --top-k 1000 \
  --max-epe 3.0
```

This writes:

- `selected_pairs__dinov3.csv`
- `selected_pairs__combined_score_sum_rgb.csv`
- `selected_pairs__combined_score_sum_lab.csv`
- `intersection_comb_score_rgb_lab_dinov3.csv`
- `intersected_pairs.csv`

For a two-metric intersection like the notebook's RGB+DINOv3 selection:

```bash
python preprocessing/similarity/select_pair_intersections.py \
  --metrics-csv outputs/similarity/results/pred-monsterplusplus__warp-pred__metric-all/per_sample_metrics.csv \
  --out-dir outputs/similarity/results/pred-monsterplusplus__warp-pred__metric-all \
  --reference-root data/synchronic_only \
  --metrics dinov3 combined_score_sum_rgb \
  --intersection-name intersection_comb_score_rgb_dinov3
```

## Outputs

Each experiment writes:

- `per_sample_metrics.csv`: one row per pair.
- `summary_metrics.json`: aggregate means, medians, and standard deviations.
- `run_config.json`: command and path metadata for the metric run.

## Dependencies

```bash
pip install numpy torch iio scikit-image tqdm transformers
```

FoundationStereo and MonSter require their respective third-party code and checkpoints. MonSter also requires the Depth-Anything V2 checkpoint expected by the original model.
