# Reproduce From Cropped Images

This guide is for users who do not want to download the full preprocessed SeasonStereo dataset. It verifies the full pipeline on a tiny subset: semantic masks, pair list generation, pseudo-ground-truth disparity prediction, rectification, experiment CSV creation, and a two-step training run.

The commands assume you run them from the repository root.

## Required Inputs

You still need these assets:

```text
data/
  Train-Track3-cropped/
  Train-Track3-cropped-synthetic/
  synchronic_only/
    L/
    R/
    homography/

checkpoints/
  monster++-mix_all.pth
  depth_anything_v2_vitl.pth
  openearthmap_segformer_mit-b2.pt
```

The cropped real/synthetic images alone are not enough. Rectification also needs the released `synchronic_only` reference pairs and homographies.

The smoke test writes generated files under:

```text
smoke_data/
```

You can delete that folder after the test.

## 1. Environment

```bash
conda create -n seasonstereo python=3.10 -y
conda activate seasonstereo

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 2. Semantic Masks

The rectification script can fall back to safe default masks if a mask is missing, but running these commands confirms that the segmentation scripts and checkpoint load correctly.

```bash
python preprocessing/segmentation/infer_water_openearthmap.py \
  --input data/Train-Track3-cropped \
  --output smoke_data/water_segmentation \
  --checkpoint checkpoints/openearthmap_segformer_mit-b2.pt \
  --device cuda \
  --limit 20

python preprocessing/segmentation/infer_tree_openearthmap.py \
  --input data/Train-Track3-cropped \
  --output smoke_data/tree_segmentation \
  --checkpoint checkpoints/openearthmap_segformer_mit-b2.pt \
  --device cuda \
  --limit 20

python preprocessing/segmentation/infer_building_openearthmap.py \
  --input data/Train-Track3-cropped \
  --output smoke_data/building_segmentation \
  --checkpoint checkpoints/openearthmap_segformer_mit-b2.pt \
  --device cuda \
  --limit 20
```

## 3. Pair List And Pseudo-GT Disparities

```bash
python preprocessing/similarity/generate_pairs_list.py \
  --reference-root data/synchronic_only \
  --output smoke_data/pairs.csv \
  --limit 5

python preprocessing/similarity/generate_disp_maps.py \
  --pairs_list smoke_data/pairs.csv \
  --out_dir smoke_data/diachronic-stereo-synthetic/disparity_maps \
  --model monster++ \
  --ckpt_path checkpoints/monster++-mix_all.pth \
  --depth_anything_v2_path checkpoints/depth_anything_v2_vitl.pth \
  --device cuda:0 \
  --max_side 0 \
  --limit 5 \
  --overwrite
```

Use `--max_side 0` for training pseudo-GT. The disparity shape must match the rectified train images and masks exactly. If you previously generated cropped disparities, rerun with `--overwrite`.

## 4. Rectify A Tiny Train Split

```bash
python preprocessing/rectification/dataset_generation.py \
  --train-reference-dir data/synchronic_only \
  --output-root smoke_data/diachronic-stereo-synthetic \
  --track3-root data/Train-Track3-cropped \
  --synthetic-root data/Train-Track3-cropped-synthetic \
  --water-masks-dir smoke_data/water_segmentation/masks \
  --tree-masks-dir smoke_data/tree_segmentation/masks \
  --building-masks-dir smoke_data/building_segmentation/masks \
  --max-reference-pairs 5 \
  --no-preview-png
```

## 5. Build A Smoke Experiment CSV

```bash
python preprocessing/rectification/build_experiment_csv.py \
  --train-root smoke_data/diachronic-stereo-synthetic/train \
  --output smoke_data/diachronic-stereo-synthetic/experiments/smoke_train.csv \
  --limit 20
```

## 6. Train For Two Steps

This only checks that the dataset, model, loss, optimizer, checkpoint loading, and training loop work. It is not intended to produce a useful model.

```bash
python season_stereo/train_monster.py \
  --config-name exp4-pseudoGT_0.05-photo_0.1_buildings-smooth_0.1 \
  skip_validation=true \
  logging.trackers=[] \
  logging.run_name=smoke_test \
  max_step=2 \
  train_iters=2 \
  valid_iters=2 \
  batch_size=1 \
  image_size=384 \
  augmentation.crop_size=384 \
  restore_ckpt=checkpoints/monster++-mix_all.pth \
  depth_anything_v2_path=checkpoints/depth_anything_v2_vitl.pth \
  dfc.root=smoke_data/diachronic-stereo-synthetic \
  dfc.experiment_csv=smoke_data/diachronic-stereo-synthetic/experiments/smoke_train.csv \
  dfc.train_aois_csv=null \
  dfc.disparity_dir=smoke_data/diachronic-stereo-synthetic/disparity_maps/monster++
```

Expected result:

```text
outputs/training/smoke_test/
  hparams.yml
  final.pth
```

## Notes From The Smoke Test

- `skip_validation` is declared in the release configs, so Hydra accepts `skip_validation=true`.
- If you generate smoke data outside `data/diachronic-stereo-synthetic`, override `dfc.root`, `dfc.experiment_csv`, and `dfc.disparity_dir` together.
- Do not center-crop pseudo-GT disparity maps for training. Use `--max_side 0`.
- `dataset_generation.py` saves train masks and images with matching rectified shapes.
