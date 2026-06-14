#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
set +u
source activate_env.sh
set -u

SCENE_DIR="${SCENE_DIR:-data/vinyes_sam3_200}"
MODEL_DIR="${MODEL_DIR:-output/vinyes_sam3_200}"

python train.py \
  -s "$SCENE_DIR" \
  -m "$MODEL_DIR" \
  --config_file config/gaussian_dataset/vinyes_sam3_200.json \
  --iterations 40000 \
  --test_iterations 1000 10000 30000 40000 \
  --save_iterations 30000 40000 \
  --resolution 4 \
  --eval \
  "$@"
