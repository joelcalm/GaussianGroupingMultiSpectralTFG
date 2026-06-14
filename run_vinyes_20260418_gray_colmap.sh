#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
set +u
source activate_env.sh
set -u

exec python \
  script/rebuild_gray_colmap_from_scene.py \
  --source_scene data/vinyes_fulles_1 \
  --output_scene data/vinyes_20260418 \
  --bands rgb,b470,b505,b525,b590,b635,b660 \
  --exclude_bands b850 \
  --cuda_visible_devices 1 \
  --colmap_gpu_index 0 \
  --max_num_features 16000 \
  --estimate_affine_shape 0 \
  --domain_size_pooling 0 \
  --matching_use_gpu 1 \
  --abs_pose_min_num_inliers 250 \
  --abs_pose_max_error 2.0 \
  --mapper_min_num_matches 100
