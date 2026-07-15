# Segmentation Masks

These scripts generate semantic masks used by the rectification step. They all load the same 9-class OpenEarthMap SegFormer checkpoint and write class-specific masks.

## Files

- `infer_water_openearthmap.py`
- `infer_tree_openearthmap.py`
- `infer_building_openearthmap.py`
- `segmentation.ipynb`: small notebook to inspect an image, mask, and probability map.

## Mask Conventions

All output masks are named:

```text
{image_stem}_mask.png
```

Water and tree masks are keep masks for training:

- water/tree pixel: `0`
- non-water/non-tree pixel: `1` or `255`, depending on the script output scale

Building masks are written in the same raw convention expected by rectification:

- building pixel: `0`
- non-building pixel: `1`

The rectification generator normalizes these conventions and saves the final rectified masks as NumPy arrays.

## Usage

Run from the repository root:

```bash
python preprocessing/segmentation/infer_water_openearthmap.py \
  --input data/Train-Track3-cropped \
  --output data/water_segmentation \
  --checkpoint checkpoints/openearthmap_segformer_mit-b2.pt \
  --device cuda

python preprocessing/segmentation/infer_tree_openearthmap.py \
  --input data/Train-Track3-cropped \
  --output data/tree_segmentation \
  --checkpoint checkpoints/openearthmap_segformer_mit-b2.pt \
  --device cuda

python preprocessing/segmentation/infer_building_openearthmap.py \
  --input data/Train-Track3-cropped \
  --output data/building_segmentation \
  --checkpoint checkpoints/openearthmap_segformer_mit-b2.pt \
  --device cuda
```

For a quick smoke test:

```bash
python preprocessing/segmentation/infer_water_openearthmap.py --input data/Train-Track3-cropped --output tmp/water --checkpoint checkpoints/openearthmap_segformer_mit-b2.pt --limit 5
```

Each output folder contains:

```text
masks/
probs/
*_percentages.txt
```

## Dependencies

```bash
pip install numpy Pillow scipy torch transformers
```
