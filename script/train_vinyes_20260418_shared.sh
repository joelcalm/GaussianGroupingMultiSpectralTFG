#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
set +u
source activate_env.sh
set -u

export PYTHONPATH="submodules/diff-gaussian-rasterization:submodules/simple-knn/build/lib.linux-x86_64-cpython-38:${PYTHONPATH:-}"

python train.py \
  -s data/vinyes_20260418_rgb_colmap_shared \
  -m output/vinyes_20260418_rgb_colmap_shared \
  --config_file config/gaussian_dataset/vinyes_20260418_shared.json \
  --iterations 40000 \
  --test_iterations 1000 10000 30000 40000 \
  --save_iterations 30000 40000 \
  --resolution 4 \
  --eval \
  "$@"
