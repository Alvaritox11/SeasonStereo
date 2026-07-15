#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Change the config name here to launch a different experiment.
CUDA_VISIBLE_DEVICES=0 python season_stereo/train_monster.py --config-name exp4-pseudoGT_0.05-photo_0.1_buildings-smooth_0.1
