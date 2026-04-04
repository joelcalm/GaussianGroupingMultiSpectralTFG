#!/bin/bash

source activate_env.sh

set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3

OUTPUT_ROOT="../../data/tmp/jcalm/output"

EXPERIMENTS=(
    "exp_a_capped_longer_densify:exp_a_capped_longer_densify"
    "exp_b_strong_reg:exp_b_strong_reg"
    "exp_c_tuned_baseline:exp_c_tuned_baseline"
    "exp_d_compact_strong_reg:exp_d_compact_strong_reg"
)

echo "============================================="
echo "  Overnight experiment batch"
echo "  Started: $(date)"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo "  Experiments: ${#EXPERIMENTS[@]}"
echo "============================================="

for entry in "${EXPERIMENTS[@]}"; do
    CONFIG_NAME="${entry%%:*}"
    OUTPUT_NAME="${entry##*:}"
    CONFIG_FILE="config/gaussian_dataset/${CONFIG_NAME}.json"

    echo ""
    echo "============================================="
    echo "  Experiment: $OUTPUT_NAME"
    echo "  Config: $CONFIG_FILE"
    echo "  Start: $(date)"
    echo "============================================="

    echo "[TRAIN] Starting training..."
    bash script/train_mms.sh \
        --scene birdhouse \
        --output "$OUTPUT_NAME" \
        --config_file "$CONFIG_FILE" \
        --iterations 40000

    echo "[RENDER] Rendering test views..."
    python render.py -m "$OUTPUT_ROOT/$OUTPUT_NAME" --num_classes 256 --images images

    echo "[METRICS] Computing metrics..."
    python metrics.py -m "$OUTPUT_ROOT/$OUTPUT_NAME"

    echo "[DONE] Experiment $OUTPUT_NAME finished at $(date)"
    echo ""
done

echo ""
echo "============================================="
echo "  All experiments finished: $(date)"
echo "============================================="

echo ""
echo "=== RESULTS SUMMARY ==="
for entry in "${EXPERIMENTS[@]}"; do
    OUTPUT_NAME="${entry##*:}"
    RESULTS_FILE="$OUTPUT_ROOT/$OUTPUT_NAME/results.json"
    echo ""
    echo "--- $OUTPUT_NAME ---"
    if [ -f "$RESULTS_FILE" ]; then
        cat "$RESULTS_FILE"
    else
        echo "  (no results.json found)"
    fi
done
