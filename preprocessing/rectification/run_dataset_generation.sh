#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Change these paths if your downloaded data lives elsewhere.
python preprocessing/rectification/dataset_generation.py \
  --train-reference-dir data/synchronic_only \
  --output-root data/diachronic-stereo-synthetic \
  --track3-root data/Train-Track3-cropped \
  --synthetic-root data/Train-Track3-cropped-synthetic \
  --water-masks-dir data/water_segmentation/masks \
  --tree-masks-dir data/tree_segmentation/masks \
  --building-masks-dir data/building_segmentation/masks
