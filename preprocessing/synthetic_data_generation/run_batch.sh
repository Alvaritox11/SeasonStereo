#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Change these paths if your data or output folder are elsewhere.
python preprocessing/synthetic_data_generation/batch_generation.py \
  --track3-root data/Train-Track3-cropped \
  --aoi-list preprocessing/synthetic_data_generation/selected_aois.txt \
  --batch-dir outputs/synthetic_batch

python preprocessing/synthetic_data_generation/extract_batch_results.py \
  --batch-dir outputs/synthetic_batch
