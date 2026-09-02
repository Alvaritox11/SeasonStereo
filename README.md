# SeasonStereo

Diachronic stereo matching for multi-date satellite imagery.

This repository contains the release code for the SeasonStereo paper: training, evaluation, dataset preprocessing, semantic masks, synthetic seasonal data utilities, and similarity/pair-selection tools. Large datasets, generated products, and model checkpoints are distributed separately through Hugging Face.

## Links

- [Project page](https://alvaritox11.github.io/SeasonStereo/)
- [Paper](https://arxiv.org/abs/2607.27139)
- [Dataset](https://huggingface.co/datasets/Alvaritox/seasonstereo-data)
- [Checkpoints](https://huggingface.co/Alvaritox/seasonstereo)

## Repository Layout

```text
.
├── season_stereo/                         # training and test evaluation
├── preprocessing/
│   ├── rectification/                     # rectified training split generation
│   ├── segmentation/                      # water/tree/building mask inference
│   ├── similarity/                        # pair metrics and pseudo-GT disparity helpers
│   └── synthetic_data_generation/         # optional seasonal image generation
├── docs/                                  # static project page draft
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

| Asset | Location |
| --- | --- |
| Data (test/val splits, cropped tiles, seasonal variants, masks, train subset) | [Alvaritox/seasonstereo-data](https://huggingface.co/datasets/Alvaritox/seasonstereo-data) |
| SeasonStereo checkpoint | [Alvaritox/seasonstereo](https://huggingface.co/Alvaritox/seasonstereo) |

```bash
pip install -U "huggingface_hub[cli]"

hf download Alvaritox/seasonstereo-data --repo-type dataset --local-dir data
hf download Alvaritox/seasonstereo season-stereo-final.pth --local-dir checkpoints
```

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

The download layout differs slightly from the layout the scripts expect. After downloading, move `segmentation_masks/{water,tree,building}_segmentation/` to `data/` and `diachronic-stereo-synthetic/synchronic_only/` to `data/synchronic_only/`.

The scripts and configs use these relative paths directly. If your folders are elsewhere, edit the path lines in the relevant `.sh` file or YAML config, or pass Hydra overrides on the command line.


### Full training split

The full rectified training split (~700 GB across 77 AOIs) is **not distributed**. The dataset repository ships `diachronic-stereo-synthetic/train_subset/`, a single-AOI sample with the exact same structure, so the data format and the training loop can be inspected and run end to end.

To build the complete training split, run the generation pipeline on the released cropped tiles as described in [REPRODUCE_FROM_CROPPED_IMAGES.md](REPRODUCE_FROM_CROPPED_IMAGES.md). Every input it needs is in the dataset repository: real crops, seasonal variants, `synchronic_only` reference pairs and homographies, and semantic masks. Dropping the `--limit` and `--max-reference-pairs` flags reproduces the full split rather than the smoke-test subset.


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
@inproceedings{diaz-laureano2026seasonstereo,
  title     = {SeasonStereo: Robust Dense Stereo Matching for Multi-Date Satellite Imagery via Generative AI},
  author    = {D{\'\i}az-Laureano, {\'A}lvaro and Mar{\'\i}, Roger and Masquil, El{\'\i}as and Arias, Pablo and Facciolo, Gabriele},
  booktitle = {Proceedings of the TerraBytes II Workshop at the European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## Acknowledgments
 
The non-synthetic satellite images used in this project originate from the [2019 IEEE GRSS Data Fusion Contest (DFC2019)](https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019),
Track 3. The original WorldView-3 imagery is provided courtesy of **DigitalGlobe** (now Maxar). The DFC2019 dataset was organized by the IEEE GRSS Image Analysis and Data Fusion Technical Committee (IADF TC), the **Johns Hopkins University Applied Physics Laboratory** (JHU/APL), and the **Intelligence Advanced Research Projects Activity** (IARPA). All non-synthetic images and their copyright remain the property of their respective owners.
 
We are grateful to the DFC2019 organizers and data providers for making this large-scale satellite stereo dataset publicly available and enabling further research in 3D reconstruction from satellite imagery.
 
If you use our dataset, please also cite the original DFC2019 references:
 
```bibtex
@inproceedings{bosch2019semantic,
  title     = {Semantic Stereo for Incidental Satellite Images},
  author    = {Bosch, Marc and Foster, Kevin and Christie, Gordon and Wang, Sean and Hager, Gregory D. and Brown, Myron},
  booktitle = {2019 IEEE Winter Conference on Applications of Computer Vision (WACV)},
  pages     = {1524--1532},
  year      = {2019},
  organization = {IEEE}
}
 
@article{le_saux2019dfc,
  title   = {2019 IEEE GRSS Data Fusion Contest: Large-Scale Semantic 3D Reconstruction},
  author  = {Le Saux, Bertrand and Yokoya, Naoto and H{\"a}nsch, Ronny and Brown, Myron},
  journal = {IEEE Geoscience and Remote Sensing Magazine},
  volume  = {7},
  number  = {4},
  pages   = {33--36},
  year    = {2019}
}
```
