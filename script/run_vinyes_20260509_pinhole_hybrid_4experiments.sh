#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."
set +u
source activate_env.sh
set -u
mkdir -p logs

DEVICE="${DEVICE:-2}"
ITERATIONS="${ITERATIONS:-40000}"
RESOLUTION="${RESOLUTION:-4}"
OBJECT_PATH="object_mask_experiments"
SCENE="data/vinyes_20260509_pinhole"
SUBSET="$SCENE/manual_eval_gt/target_two_vines_all31"
FINAL_ALL="$SUBSET/final_15_all_objects"
FINAL_TARGET="$SUBSET/final_15"
EVAL_ALL_OUT="output/vinyes_20260509_pinhole_hybrid_object_eval_all_objects"
EVAL_TARGET_OUT="output/vinyes_20260509_pinhole_hybrid_object_eval_target_compact"

variants=(no_color rgb ms rgb_ms)
ports=(8710 8711 8712 8713)
models=()

printf '===== Building hybrid all-object masks at %s =====\n' "$(date -Is)"
python tools/manual_eval/build_hybrid_target_all_masks.py \
  --scene_dir "$SCENE" \
  --subset_dir "$SUBSET" \
  > logs/vinyes_20260509_pinhole_hybrid_build_masks.log 2>&1

printf '===== Preparing hybrid split/configs at %s =====\n' "$(date -Is)"
python tools/experiments/prepare_vinyes_20260509_pinhole_experiments.py \
  --scene_dir "$SCENE" \
  --manual_eval_dir "$FINAL_ALL" \
  --object_mask_dir "$SCENE/manual_eval_gt/object_mask_hybrid_target_all" \
  --config_dir config/gaussian_dataset \
  > logs/vinyes_20260509_pinhole_hybrid_prepare_experiments.log 2>&1

for i in "${!variants[@]}"; do
  variant="${variants[$i]}"
  port="${ports[$i]}"
  model="output/vinyes_20260509_pinhole_hybrid_${variant}"
  config="config/gaussian_dataset/vinyes_20260509_pinhole_${variant}.json"
  models+=("$model")

  if [[ "${CLEAN_OUTPUT:-0}" == "1" ]]; then
    rm -rf "$model"
  fi

  printf '===== [%s] Training hybrid %s started at %s =====\n' "$((i+1))/${#variants[@]}" "$variant" "$(date -Is)"
  CUDA_VISIBLE_DEVICES="$DEVICE" python train.py \
    -s "$SCENE" \
    -m "$model" \
    --config_file "$config" \
    --object_path "$OBJECT_PATH" \
    --iterations "$ITERATIONS" \
    --test_iterations 1000 10000 30000 "$ITERATIONS" \
    --save_iterations 30000 "$ITERATIONS" \
    --resolution "$RESOLUTION" \
    --eval \
    --train_split \
    --quiet \
    --port "$port" \
    > "logs/vinyes_20260509_pinhole_hybrid_${variant}_train.log" 2>&1
  printf '===== [%s] Training hybrid %s finished at %s =====\n' "$((i+1))/${#variants[@]}" "$variant" "$(date -Is)"

  printf '===== [%s] Rendering hybrid RGB test views for %s started at %s =====\n' "$((i+1))/${#variants[@]}" "$variant" "$(date -Is)"
  CUDA_VISIBLE_DEVICES="$DEVICE" python render.py \
    -m "$model" \
    --iteration "$ITERATIONS" \
    --skip_train \
    --only_prefix rgb \
    --quiet \
    > "logs/vinyes_20260509_pinhole_hybrid_${variant}_render_rgb_test.log" 2>&1
  printf '===== [%s] Rendering hybrid RGB test views for %s finished at %s =====\n' "$((i+1))/${#variants[@]}" "$variant" "$(date -Is)"
done

printf '===== All-object evaluation started at %s =====\n' "$(date -Is)"
python tools/eval/evaluate_final15_object_ids.py \
  --model_paths "${models[@]}" \
  --gt_dir "$FINAL_ALL/object_mask" \
  --selected_frames_csv "$FINAL_ALL/selected_frames.csv" \
  --iteration "$ITERATIONS" \
  --label_mode all_gt \
  --per_model_output_name manual_eval_final15_all_objects \
  --output_dir "$EVAL_ALL_OUT" \
  > logs/vinyes_20260509_pinhole_hybrid_object_eval_all_objects.log 2>&1
printf '===== All-object evaluation finished at %s =====\n' "$(date -Is)"

printf '===== Target-compact evaluation started at %s =====\n' "$(date -Is)"
python tools/eval/evaluate_final15_object_ids.py \
  --model_paths "${models[@]}" \
  --gt_dir "$FINAL_TARGET/object_mask" \
  --selected_frames_csv "$FINAL_TARGET/selected_frames.csv" \
  --iteration "$ITERATIONS" \
  --label_mode target_compact \
  --per_model_output_name manual_eval_final15_target_compact \
  --output_dir "$EVAL_TARGET_OUT" \
  > logs/vinyes_20260509_pinhole_hybrid_object_eval_target_compact.log 2>&1
printf '===== Target-compact evaluation finished at %s =====\n' "$(date -Is)"

printf 'All-object summary: %s/summary.csv\n' "$EVAL_ALL_OUT"
printf 'Target-compact summary: %s/summary.csv\n' "$EVAL_TARGET_OUT"
