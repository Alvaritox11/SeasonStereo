# Rectification

This folder builds the rectified training split used by SeasonStereo. It expects the raw release data to already be downloaded locally.

## Files

- `dataset_generation.py`: training split generation. It rectifies RGB pairs, shared validity masks, water/tree keep masks, building masks, and homographies.
- `build_experiment_csv.py`: creates a small `filename` CSV from generated train samples for smoke-test training.
- `run_dataset_generation.sh`: example shell wrapper with direct editable paths.
- `dataset_check.py`: consistency/debug utility kept from the original workflow.

## Expected Input Layout

The default paths assume this structure under `data/`:

```text
data/
  synchronic_only/
    L/
    R/
    homography/
  Train-Track3-cropped/
    Track3-RGB-1/
    Track3-RGB-2/
  Train-Track3-cropped-synthetic/
    Track3-RGB-1/
    Track3-RGB-2/
  water_segmentation/masks/
  tree_segmentation/masks/
  building_segmentation/masks/
```

`synchronic_only/` is useful to keep in the data release because it documents the reference pairs and stores the homographies used for rectification.

## Mask Conventions

- Geometry masks: `1 = valid rectified pixel`, `0 = outside warp`.
- Water/tree masks: `1 = keep pixel`, `0 = masked class or outside warp`.
- Building masks: `1 = building`, `0 = non-building or outside warp`.

If a water/tree mask is missing, the generator writes an all-ones keep mask inside the valid warp area and logs the missing file. If a building mask is missing, it writes an all-zero building mask and logs it.

## Usage

Run from the repository root:

```bash
bash preprocessing/rectification/run_dataset_generation.sh
```

or explicitly:

```bash
python preprocessing/rectification/dataset_generation.py \
  --train-reference-dir data/synchronic_only \
  --output-root data/diachronic-stereo-synthetic \
  --track3-root data/Train-Track3-cropped \
  --synthetic-root data/Train-Track3-cropped-synthetic \
  --water-masks-dir data/water_segmentation/masks \
  --tree-masks-dir data/tree_segmentation/masks \
  --building-masks-dir data/building_segmentation/masks
```

Useful debug options:

```bash
python preprocessing/rectification/dataset_generation.py --max-reference-pairs 10 --no-preview-png
python preprocessing/rectification/dataset_generation.py --no-synthetic
```

For smoke-test training, create a minimal experiment CSV from the generated images:

```bash
python preprocessing/rectification/build_experiment_csv.py \
  --train-root data/diachronic-stereo-synthetic/train \
  --output data/diachronic-stereo-synthetic/experiments/smoke_train.csv \
  --limit 20
```

## Output Layout

The default output root is `data/diachronic-stereo-synthetic/`:

```text
diachronic-stereo-synthetic/
  train/
    L/
    R/
    masks/{L,R}/
    water_masks/{L,R}/
    tree_masks/{L,R}/
    building_masks/{L,R}/
    homography/
    png/                 # optional previews
```

The generated train split includes the real-real pair and all available real/synthetic combinations. The validation split used by training is expected to come already rectified in the released dataset.

## Dependencies

```bash
pip install numpy opencv-python iio scikit-image Pillow
```
