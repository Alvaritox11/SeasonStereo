# Synthetic Seasonal Data Generation

This folder contains the scripts used to generate seasonal variants of the cropped Track3 satellite images. It supports a small preview workflow and a batch workflow.

Configure access to Google GenAI in your runtime before running these scripts. No credentials should be stored in this repository.

## Files

- `single_generation.py`: preview prompts on one image or a few AOIs.
- `batch_generation.py`: upload selected AOI images and submit a batch generation job.
- `extract_batch_results.py`: extract generated images from a batch JSONL result.
- `utils.py`: prompts and shared AOI/image helpers.
- `selected_aois.txt`: AOI names to process in batch mode.
- `run_batch.sh`: example batch wrapper.

## Expected Input Layout

```text
data/Train-Track3-cropped/
  Track3-RGB-1/
    JAX_004/
      JAX_004_001_RGB.tif
  Track3-RGB-2/
    OMA_042/
      OMA_042_004_RGB.tif
```

Each AOI folder contains one or more `*_RGB.tif` images.

## Preview Generation

Use this for prompt checks before launching a full batch.
Run from the repository root.

```bash
python preprocessing/synthetic_data_generation/single_generation.py \
  --image-path data/Train-Track3-cropped/Track3-RGB-1/JAX_004/JAX_004_001_RGB.tif \
  --seasons WINTER SUMMER \
  --output-dir outputs/synthetic_preview
```

Without `--image-path`, the script previews the first AOI folders it finds:

```bash
python preprocessing/synthetic_data_generation/single_generation.py \
  --track3-root data/Train-Track3-cropped \
  --num-aois 2 \
  --output-dir outputs/synthetic_preview
```

## Batch Generation

Edit `selected_aois.txt` with one AOI folder name per line:

```text
JAX_004
JAX_028
OMA_042
```

Then run:

```bash
python preprocessing/synthetic_data_generation/batch_generation.py \
  --track3-root data/Train-Track3-cropped \
  --aoi-list preprocessing/synthetic_data_generation/selected_aois.txt \
  --batch-dir outputs/synthetic_batch
```

When the batch result file is available:

```bash
python preprocessing/synthetic_data_generation/extract_batch_results.py --batch-dir outputs/synthetic_batch
```

Generated images are saved as:

```text
outputs/synthetic_batch/generated_images/{image_stem}_{SEASON}.png
```

The rectification pipeline expects synthetic variants to be placed under:

```text
data/Train-Track3-cropped-synthetic/
  Track3-RGB-1/{AOI}/{image_stem}_{SEASON}.png
  Track3-RGB-2/{AOI}/{image_stem}_{SEASON}.png
```

## Prompts

Prompts are defined in `utils.py` and enforce:

- unchanged geometry, camera viewpoint, scale, and alignment
- unchanged roads, buildings, boundaries, river banks, shorelines, and water contours
- season-specific vegetation, lighting, and shadow changes
- realistic satellite-image appearance without structural edits

## Dependencies

```bash
pip install google-genai Pillow
```
