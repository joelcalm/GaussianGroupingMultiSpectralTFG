#!/bin/bash
# Phase 3: Train Gaussian Grouping on MMS-DATA multispectral (all 9 bands, all images)
#
# Prerequisites:
#   1. Run prepare_mms_dataset.py to create COLMAP-format data + 9-ch images
#   2. Run prepare_pseudo_label.sh to generate object masks
#   3. Rebuild diff-gaussian-rasterization with NUM_CHANNELS=9
#
# Usage: bash script/train_mms.sh <scene_name> [output_name]
#   e.g.: bash script/train_mms.sh birdhouse phase3_birdhouse

SCENE=${1:-birdhouse}
OUTPUT=${2:-phase3_${SCENE}}

python train.py \
    -s data/multi-modal-studio/${SCENE} \
    -m output/${OUTPUT} \
    --eval \
    --config_file config/gaussian_dataset/train_mms.json \
    --save_iterations 1000 7000 15000 30000 \
    --test_iterations 1000 7000 15000 30000
