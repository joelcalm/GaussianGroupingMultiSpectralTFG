#!/bin/bash
# Phase 3: Train Gaussian Grouping on MMS-DATA multispectral (all 9 bands, all images)
#
# Prerequisites:
#   1. Run prepare_mms_dataset.py to create COLMAP-format data + 9-ch images
#   2. Run prepare_pseudo_label.sh to generate object masks
#   3. Rebuild diff-gaussian-rasterization with NUM_CHANNELS=9

set -euo pipefail

DATA_ROOT="../../data/tmp/jcalm/data/multi-modal-studio"
OUTPUT_ROOT="../../data/tmp/jcalm/output"
DEFAULT_CONFIG="config/gaussian_dataset/train_mms_msiau.json"
DEFAULT_SCENE="birdhouse"

usage() {
    cat <<EOF
Usage:
  bash script/train_mms.sh <scene_name> [output_name] [config_file] [extra train args...]
  bash script/train_mms.sh --scene <scene_name> [--output <output_name>] [--config_file <config_file>] [extra train args...]

Examples:
  bash script/train_mms.sh birdhouse phase3_birdhouse
  bash script/train_mms.sh --scene birdhouse --output msiau_base_40k --config_file config/gaussian_dataset/train_mms_msiau.json --iterations 40000

Notes:
  - Any extra args are forwarded to train.py (e.g. --iterations 40000, --ip 0.0.0.0).
    - If --scene is omitted and the first positional arg is not a valid scene, it is treated as output_name.
        The script then falls back to DEFAULT_SCENE (${DEFAULT_SCENE}) if available, else the first available scene.
EOF
}

SCENE=""
OUTPUT=""
CONFIG_FILE="$DEFAULT_CONFIG"
EXTRA_ARGS=()
SCENE_SET_BY_FLAG=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --scene)
            SCENE="$2"
            SCENE_SET_BY_FLAG=1
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --config_file|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --*)
            EXTRA_ARGS+=("$1")
            if [[ $# -gt 1 && "$2" != --* ]]; then
                EXTRA_ARGS+=("$2")
                shift 2
            else
                shift 1
            fi
            ;;
        *)
            if [[ -z "$SCENE" ]]; then
                SCENE="$1"
            elif [[ -z "$OUTPUT" ]]; then
                if [[ "$1" == *.json ]]; then
                    CONFIG_FILE="$1"
                else
                    OUTPUT="$1"
                fi
            elif [[ "$1" == *.json && "$CONFIG_FILE" == "$DEFAULT_CONFIG" ]]; then
                CONFIG_FILE="$1"
            else
                EXTRA_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "Error: Dataset root '$DATA_ROOT' does not exist."
    exit 2
fi

mapfile -t AVAILABLE_SCENES < <(find "$DATA_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)

if [[ -z "$SCENE" ]]; then
    echo "Error: scene_name is required."
    usage
    echo "Available scenes: ${AVAILABLE_SCENES[*]:-<none>}"
    exit 2
fi

if [[ ! -d "$DATA_ROOT/$SCENE" ]]; then
    if [[ $SCENE_SET_BY_FLAG -eq 0 && -z "$OUTPUT" ]]; then
        SELECTED_SCENE=""
        if [[ -d "$DATA_ROOT/$DEFAULT_SCENE" ]]; then
            SELECTED_SCENE="$DEFAULT_SCENE"
        elif [[ ${#AVAILABLE_SCENES[@]} -gt 0 ]]; then
            SELECTED_SCENE="${AVAILABLE_SCENES[0]}"
        fi

        if [[ -n "$SELECTED_SCENE" ]]; then
            echo "Warning: '$SCENE' is not a valid scene in $DATA_ROOT."
            echo "Treating '$SCENE' as output_name and using scene '$SELECTED_SCENE'."
            OUTPUT="$SCENE"
            SCENE="$SELECTED_SCENE"
        else
            echo "Error: Scene '$SCENE' was not found under '$DATA_ROOT'."
            echo "Available scenes: ${AVAILABLE_SCENES[*]:-<none>}"
            exit 2
        fi
    elif [[ ${#AVAILABLE_SCENES[@]} -eq 1 ]]; then
        echo "Warning: '$SCENE' is not a valid scene in $DATA_ROOT."
        echo "Using the only available scene '${AVAILABLE_SCENES[0]}' and treating '$SCENE' as output_name."
        OUTPUT="${OUTPUT:-$SCENE}"
        SCENE="${AVAILABLE_SCENES[0]}"
    else
        echo "Error: Scene '$SCENE' was not found under '$DATA_ROOT'."
        echo "Available scenes: ${AVAILABLE_SCENES[*]:-<none>}"
        exit 2
    fi
fi

if [[ -z "$OUTPUT" ]]; then
    OUTPUT="phase3_${SCENE}"
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Config file '$CONFIG_FILE' does not exist."
    exit 2
fi

python train.py \
    -s "$DATA_ROOT/$SCENE" \
    -m "$OUTPUT_ROOT/$OUTPUT" \
    --eval \
    -r 2 \
    --config_file "$CONFIG_FILE" \
    --save_iterations 1000 7000 15000 30000 40000 \
    --test_iterations 1000 7000 15000 30000 40000 \
    "${EXTRA_ARGS[@]}"
