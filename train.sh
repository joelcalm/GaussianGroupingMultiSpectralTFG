#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

usage() {
  cat <<'USAGE'
Usage: bash train.sh <run> [extra train.py args]

Runs:
  bear_rgb
  bear_rgb_round_robin
  basement_all9
  basement_round_robin_9
  vines_20260321_rgb_ms
  vines_20260418_rgb_ms
  vines_20260509_rgb_ms
  vines_20260509_object_no_color
  vines_20260509_object_rgb
  vines_20260509_object_ms
  vines_20260509_object_rgb_ms

Environment overrides:
  SCENE_DIR, MODEL_DIR, ITERATIONS, TEST_ITERATIONS, SAVE_ITERATIONS, RESOLUTION, CUDA_VISIBLE_DEVICES
USAGE
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

RUN="$1"
shift

if [[ -f activate_env.sh ]]; then
  set +u
  source activate_env.sh
  set -u
fi

ITERATIONS="${ITERATIONS:-30000}"
TEST_ITERATIONS="${TEST_ITERATIONS:-1000 10000 30000}"
SAVE_ITERATIONS="${SAVE_ITERATIONS:-10000 30000}"
EXTRA_ARGS=(--eval)
RESOLUTION_DEFAULT=""

case "$RUN" in
  bear_rgb)
    DEFAULT_SCENE="data/bear"
    DEFAULT_MODEL="output/bear_rgb"
    CONFIG="config/gaussian_dataset/bear_rgb.json"
    ;;
  bear_rgb_round_robin)
    DEFAULT_SCENE="data/bear"
    DEFAULT_MODEL="output/bear_rgb_round_robin"
    CONFIG="config/gaussian_dataset/bear_rgb_round_robin.json"
    ;;
  basement_all9)
    DEFAULT_SCENE="data/basement"
    DEFAULT_MODEL="output/basement_all9"
    CONFIG="config/gaussian_dataset/basement_all9.json"
    RESOLUTION_DEFAULT="2"
    ;;
  basement_round_robin_9)
    DEFAULT_SCENE="data/basement"
    DEFAULT_MODEL="output/basement_round_robin_9"
    CONFIG="config/gaussian_dataset/basement_round_robin_9.json"
    RESOLUTION_DEFAULT="2"
    ;;
  vines_20260321_rgb_ms)
    DEFAULT_SCENE="data/vinyes_20260321"
    DEFAULT_MODEL="output/vines_20260321_rgb_ms"
    CONFIG="config/gaussian_dataset/vines_20260321_rgb_ms.json"
    RESOLUTION_DEFAULT="4"
    EXTRA_ARGS+=(--train_split)
    ;;
  vines_20260418_rgb_ms)
    DEFAULT_SCENE="data/vinyes_20260418_rgb_colmap_shared"
    DEFAULT_MODEL="output/vines_20260418_rgb_ms"
    CONFIG="config/gaussian_dataset/vines_20260418_rgb_ms.json"
    RESOLUTION_DEFAULT="4"
    EXTRA_ARGS+=(--train_split)
    ;;
  vines_20260509_rgb_ms)
    DEFAULT_SCENE="data/vinyes_20260509"
    DEFAULT_MODEL="output/vines_20260509_rgb_ms"
    CONFIG="config/gaussian_dataset/vines_20260509_rgb_ms.json"
    RESOLUTION_DEFAULT="4"
    EXTRA_ARGS+=(--train_split)
    ;;
  vines_20260509_object_no_color)
    DEFAULT_SCENE="data/vinyes_20260509"
    DEFAULT_MODEL="output/vines_20260509_object_no_color"
    CONFIG="config/gaussian_dataset/vines_20260509_object_no_color.json"
    RESOLUTION_DEFAULT="4"
    EXTRA_ARGS+=(--train_split)
    ;;
  vines_20260509_object_rgb)
    DEFAULT_SCENE="data/vinyes_20260509"
    DEFAULT_MODEL="output/vines_20260509_object_rgb"
    CONFIG="config/gaussian_dataset/vines_20260509_object_rgb.json"
    RESOLUTION_DEFAULT="4"
    EXTRA_ARGS+=(--train_split)
    ;;
  vines_20260509_object_ms)
    DEFAULT_SCENE="data/vinyes_20260509"
    DEFAULT_MODEL="output/vines_20260509_object_ms"
    CONFIG="config/gaussian_dataset/vines_20260509_object_ms.json"
    RESOLUTION_DEFAULT="4"
    EXTRA_ARGS+=(--train_split)
    ;;
  vines_20260509_object_rgb_ms)
    DEFAULT_SCENE="data/vinyes_20260509"
    DEFAULT_MODEL="output/vines_20260509_object_rgb_ms"
    CONFIG="config/gaussian_dataset/vines_20260509_object_rgb_ms.json"
    RESOLUTION_DEFAULT="4"
    EXTRA_ARGS+=(--train_split)
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown run: $RUN" >&2
    usage
    exit 2
    ;;
esac

SCENE_DIR="${SCENE_DIR:-$DEFAULT_SCENE}"
MODEL_DIR="${MODEL_DIR:-$DEFAULT_MODEL}"
RESOLUTION="${RESOLUTION:-$RESOLUTION_DEFAULT}"
read -r -a TEST_ITERATION_ARGS <<< "$TEST_ITERATIONS"
read -r -a SAVE_ITERATION_ARGS <<< "$SAVE_ITERATIONS"

CMD=(python train.py
  -s "$SCENE_DIR"
  -m "$MODEL_DIR"
  --config_file "$CONFIG"
  --iterations "$ITERATIONS"
  --test_iterations "${TEST_ITERATION_ARGS[@]}"
  --save_iterations "${SAVE_ITERATION_ARGS[@]}"
)

if [[ -n "$RESOLUTION" ]]; then
  CMD+=(--resolution "$RESOLUTION")
fi

CMD+=("${EXTRA_ARGS[@]}" "$@")

echo "Run: $RUN"
echo "Scene: $SCENE_DIR"
echo "Model: $MODEL_DIR"
echo "Config: $CONFIG"
exec "${CMD[@]}"
