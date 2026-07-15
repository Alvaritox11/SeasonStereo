#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Change these paths if your data, checkpoints, or output folder are elsewhere.
CUDA_VISIBLE_DEVICES=0 python season_stereo/eval_test.py \
  --models \
    "baseline:checkpoints/monster++-mix_all.pth" \
    "season_stereo:checkpoints/season-stereo-final.pth" \
  --test-root data/diachronic-stereo-synthetic/test \
  --output-dir outputs/test_eval \
  --depth-anything checkpoints/depth_anything_v2_vitl.pth \
  --datasets omaha_sync omaha_diach jax buenos_aires \
  --device cuda \
  --skip-existing
