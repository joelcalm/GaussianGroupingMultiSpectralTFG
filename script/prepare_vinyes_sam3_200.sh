#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python prepare_vinyes_sam3_200.py \
  --source_scene ../vineyard_posematch/vinyes_partial200 \
  --sam3_dir ../vineyard_posematch/sam3_video_vinyes \
  --output_scene ../vineyard_posematch/vinyes_sam3_200 \
  --scene_name vinyes_sam3_200 \
  --label_mode instance \
  "$@"
