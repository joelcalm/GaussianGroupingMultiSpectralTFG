#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
set +u
source activate_env.sh
set -u

mkdir -p logs

echo "[1/3] Training started at $(date -Is)"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python train.py \
  -s data/vinyes_20260418_rgb_colmap_shared \
  -m output/vinyes_20260418_rgb_balanced \
  --config_file config/gaussian_dataset/vinyes_20260418_shared_rgb_balanced.json \
  --iterations 40000 \
  --test_iterations 1000 10000 30000 40000 \
  --save_iterations 30000 40000 \
  --resolution 4 \
  --eval

echo "[1/3] Training finished at $(date -Is)"

echo "[2/3] Render started at $(date -Is)"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python render.py \
  -m output/vinyes_20260418_rgb_balanced \
  --iteration 40000 \
  --quiet

echo "[2/3] Render finished at $(date -Is)"

echo "[3/3] Metrics started at $(date -Is)"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python metrics.py \
  -m output/vinyes_20260418_rgb_balanced

echo "[3/3] Metrics finished at $(date -Is)"
