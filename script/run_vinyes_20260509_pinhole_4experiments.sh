#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/msiau/workspace/jcalm
set +u
source activate_env.sh
set -u
mkdir -p logs

DEVICE="${DEVICE:-2}"
ITERATIONS="${ITERATIONS:-40000}"
RESOLUTION="${RESOLUTION:-4}"
OBJECT_PATH="object_mask_experiments"
SCENE="data/vinyes_20260509_pinhole"
EVAL_OUT="output/vinyes_20260509_pinhole_object_eval"

variants=(no_color rgb ms rgb_ms)
ports=(8610 8611 8612 8613)
models=()

printf '===== Preparing split/configs at %s =====\n' "$(date -Is)"
python tools/experiments/prepare_vinyes_20260509_pinhole_experiments.py \
  --scene_dir "$SCENE" \
  --manual_eval_dir "$SCENE/manual_eval_gt/target_two_vines_all31/final_15" \
  --config_dir config/gaussian_dataset \
  > logs/vinyes_20260509_pinhole_prepare_experiments.log 2>&1

for i in "${!variants[@]}"; do
  variant="${variants[$i]}"
  port="${ports[$i]}"
  model="output/vinyes_20260509_pinhole_${variant}"
  config="config/gaussian_dataset/vinyes_20260509_pinhole_${variant}.json"
  models+=("$model")

  if [[ "${CLEAN_OUTPUT:-0}" == "1" ]]; then
    rm -rf "$model"
  fi

  printf '===== [%s] Training %s started at %s =====\n' "$((i+1))/${#variants[@]}" "$variant" "$(date -Is)"
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
    > "logs/vinyes_20260509_pinhole_${variant}_train.log" 2>&1
  printf '===== [%s] Training %s finished at %s =====\n' "$((i+1))/${#variants[@]}" "$variant" "$(date -Is)"

  printf '===== [%s] Rendering RGB test views for %s started at %s =====\n' "$((i+1))/${#variants[@]}" "$variant" "$(date -Is)"
  CUDA_VISIBLE_DEVICES="$DEVICE" python render.py \
    -m "$model" \
    --iteration "$ITERATIONS" \
    --skip_train \
    --only_prefix rgb \
    --quiet \
    > "logs/vinyes_20260509_pinhole_${variant}_render_rgb_test.log" 2>&1
  printf '===== [%s] Rendering RGB test views for %s finished at %s =====\n' "$((i+1))/${#variants[@]}" "$variant" "$(date -Is)"
done

printf '===== Object-ID evaluation started at %s =====\n' "$(date -Is)"
python tools/eval/evaluate_final15_object_ids.py \
  --model_paths "${models[@]}" \
  --iteration "$ITERATIONS" \
  --output_dir "$EVAL_OUT" \
  > logs/vinyes_20260509_pinhole_object_eval.log 2>&1
printf '===== Object-ID evaluation finished at %s =====\n' "$(date -Is)"
printf 'Summary: %s/summary.csv\n' "$EVAL_OUT"
