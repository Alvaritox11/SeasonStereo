# SeasonStereo

Diachronic stereo matching for multi-date satellite imagery.

This repository contains the release code for the SeasonStereo paper: training, evaluation, dataset preprocessing, semantic masks, synthetic seasonal data utilities, and similarity/pair-selection tools. Large datasets, generated products, and model checkpoints are distributed separately through Hugging Face.

## Links

- Project page: coming soon
- Paper: coming soon
- Dataset: coming soon
- Checkpoints: coming soon

## Repository Layout

```text
.
├── season_stereo/                         # training and test evaluation
├── preprocessing/
│   ├── rectification/                     # rectified training split generation
│   ├── segmentation/                      # water/tree/building mask inference
│   ├── similarity/                        # pair metrics and pseudo-GT disparity helpers
│   └── synthetic_data_generation/         # optional seasonal image generation
├── others/project_page/                   # static project page draft
├── REPRODUCE_FROM_CROPPED_IMAGES.md       # minimal from-scratch smoke test
├── requirements.txt
└── README.md
```

## Installation

Create a fresh environment and install PyTorch for your CUDA version. The example below uses CUDA 12.1 wheels.

```bash
conda create -n seasonstereo python=3.10 -y
conda activate seasonstereo

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

The MonSter/MonSter++ wrapper requires the vendored MonSter code and a Depth-Anything V2 checkpoint. The expected checkpoint paths are listed below.

## Data And Checkpoints

Expected local layout after downloading the release assets:

```text
data/
  diachronic-stereo-synthetic/
    train/
    val/
    test/
    experiments/
    train_aois.csv
    val_aois.csv
  synchronic_only/
    L/
    R/
    homography/
  Train-Track3-cropped/
  Train-Track3-cropped-synthetic/
  water_segmentation/
  tree_segmentation/
  building_segmentation/

checkpoints/
  monster++-mix_all.pth
  depth_anything_v2_vitl.pth
  season-stereo-final.pth
  openearthmap_segformer_mit-b2.pt
```

The scripts and configs use these relative paths directly. If your folders are elsewhere, edit the path lines in the relevant `.sh` file or YAML config, or pass Hydra overrides on the command line.

If you do not want to download the full preprocessed dataset, follow [REPRODUCE_FROM_CROPPED_IMAGES.md](REPRODUCE_FROM_CROPPED_IMAGES.md). That guide uses only a small subset of cropped real/synthetic images, reference homographies, and checkpoints to verify the full pipeline.

## Training

The main training configs are in `season_stereo/training_configs/`.

```bash
bash season_stereo/run_experiments.sh
```

Equivalent direct command:

```bash
python season_stereo/train_monster.py \
  --config-name exp4-pseudoGT_0.05-photo_0.1_buildings-smooth_0.1
```

For a minimal training smoke test without the released validation split, use `skip_validation=true` as shown in [REPRODUCE_FROM_CROPPED_IMAGES.md](REPRODUCE_FROM_CROPPED_IMAGES.md).

## Evaluation

```bash
bash season_stereo/run_evaluation.sh
```

The default evaluation script compares the MonSter++ baseline checkpoint and the SeasonStereo checkpoint on the released test subsets.

## Preprocessing

Each preprocessing folder has a dedicated README:

- [Rectification](preprocessing/rectification/README.md)
- [Segmentation masks](preprocessing/segmentation/README.md)
- [Similarity metrics and pair selection](preprocessing/similarity/README.md)
- [Synthetic seasonal generation](preprocessing/synthetic_data_generation/README.md)

## Citation

The final citation will be added after publication metadata is available.

```bibtex
@inproceedings{seasonstereo2026,
  title     = {SeasonStereo: Diachronic Stereo Matching for Multi-Date Satellite Imagery},
  author    = {Authors},
  booktitle = {Venue},
  year      = {2026}
}
```

## Acknowledgements

This release builds on open stereo and monocular-depth work, including MonSter/MonSter++, Depth-Anything V2, FoundationStereo, and DINOv3. Semantic masks are produced with an OpenEarthMap-style SegFormer model.
