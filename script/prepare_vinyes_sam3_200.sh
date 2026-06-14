#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SOURCE_SCENE="${SOURCE_SCENE:-data/vinyes_partial200}"
SAM3_DIR="${SAM3_DIR:-data/vinyes_partial200/sam3_rgb}"
OUTPUT_SCENE="${OUTPUT_SCENE:-data/vinyes_sam3_200}"

python prepare_vinyes_sam3_200.py \
  --source_scene "$SOURCE_SCENE" \
  --sam3_dir "$SAM3_DIR" \
  --output_scene "$OUTPUT_SCENE" \
  --scene_name vinyes_sam3_200 \
  --label_mode instance \
  "$@"
