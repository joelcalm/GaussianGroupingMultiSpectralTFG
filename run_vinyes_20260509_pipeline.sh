#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/msiau/workspace/jcalm
mkdir -p logs

PIPELINE_LOG="logs/vinyes_20260509_pipeline.log"
LOCK="logs/vinyes_20260509_pipeline.lock"

exec >> "$PIPELINE_LOG" 2>&1

finish() {
  status=$?
  if [[ $status -eq 0 ]]; then
    echo "========== Pipeline completed successfully at $(date -Is) =========="
  else
    echo "========== Pipeline failed/stopped with exit code $status at $(date -Is) =========="
  fi
  rm -f "$LOCK"
}
trap finish EXIT
trap 'echo "ERROR near line $LINENO at $(date -Is)"' ERR

if [[ -e "$LOCK" ]]; then
  old_pid="$(cat "$LOCK" 2>/dev/null || true)"
  if [[ -n "${old_pid:-}" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Another pipeline appears to be running with PID $old_pid. Exiting."
    exit 1
  fi
fi

echo $$ > "$LOCK"

echo "========== Pipeline started at $(date -Is) =========="

source activate_env.sh

echo "[1/5] SAM3 Masks started at $(date -Is)"

rm -rf data/vinyes_20260509/sam3_rgb

CUDA_VISIBLE_DEVICES=0 python sam3_vine_video.py \
  --images_dir data/vinyes_20260509/images_rgb \
  --output_dir data/vinyes_20260509/sam3_rgb \
  --model weights/sam3.pt \
  --class_config config/sam3/vine_parts_posts.json \
  --save_class_outputs \
  --device 0 \
  --imgsz 1024 \
  --conf 0.25 \
  --score_threshold_detection 0.55 \
  --new_det_thresh 0.05 \
  --assoc_iou_thresh 0.55 \
  --trk_assoc_iou_thresh 0.55 \
  --init_trk_keep_alive 10 \
  --max_trk_keep_alive 15 \
  --min_trk_keep_alive -4 \
  --max_num_objects 128 \
  --mask_threshold 0.5 \
  --min_component_area 80 \
  --morph_kernel_size 5 \
  --overlay_alpha 0.45 \
  --fps 5 \
  --keep_video data/vinyes_20260509/sam3_rgb/input_rgb.mp4 \
  > logs/vinyes_20260509_sam3.log 2>&1

echo "[1/5] SAM3 Masks finished at $(date -Is)"


echo "[2/5] Prepare Hierarchical Masks started at $(date -Is)"

python - > logs/vinyes_20260509_prepare_masks_python.log 2>&1 <<'PY'
import json, shutil
from pathlib import Path

scene = Path("data/vinyes_20260509")
sam = scene / "sam3_rgb"
semantic = scene / "semantic_mask"
metadata = scene / "metadata"

shutil.rmtree(semantic, ignore_errors=True)
semantic.mkdir(parents=True, exist_ok=True)
metadata.mkdir(parents=True, exist_ok=True)

registered = []
for line in (scene / "sparse/0/images.txt").read_text().splitlines():
    parts = line.split()
    if len(parts) >= 10 and parts[9].endswith(".png"):
        registered.append(parts[9])

for name in registered:
    if name.startswith("rgb_"):
        src = sam / "semantic_index_masks" / name
        if src.exists():
            shutil.copy2(src, semantic / name)

active = {}
for name in registered:
    stem = Path(name).stem
    band = stem.split("_", 1)[0]
    channels = {
        "rgb": [0, 1, 2],
        "b470": [3],
        "b505": [4],
        "b525": [5],
        "b590": [6],
        "b635": [7],
        "b660": [8],
    }[band]
    active[stem] = channels

(metadata / "active_channels.json").write_text(json.dumps(active, indent=2) + "\n")
PY

python compose_hierarchical_vineyard_labels.py \
  --scene_dir data/vinyes_20260509 \
  --sam3_dir data/vinyes_20260509/sam3_rgb \
  --scene_name vinyes_20260509 \
  --config_out config/gaussian_dataset/vinyes_20260509_sam3.json \
  --whole_vine_class vine_plant \
  --post_class wooden_post \
  --part_class leaf=vine_leaf \
  --part_class trunk=vine_trunk \
  --semantic_class ground \
  --semantic_class sky \
  --semantic_class tree \
  --semantic_class stone_wall \
  --semantic_class shrub_or_other_vegetation \
  --semantic_class hedge_or_wall_vegetation \
  --semantic_class building \
  --association_dilate_pixels 7 \
  --min_part_overlap_pixels 20 \
  --min_instance_pixels 100 \
  --overwrite \
  > logs/vinyes_20260509_prepare_masks.log 2>&1

echo "[2/5] Prepare Hierarchical Masks finished at $(date -Is)"


echo "[3/5] Training started at $(date -Is)"

CUDA_VISIBLE_DEVICES=0 python train.py \
  -s data/vinyes_20260509 \
  -m output/vinyes_20260509_sam3 \
  --config_file config/gaussian_dataset/vinyes_20260509_sam3.json \
  --iterations 40000 \
  --test_iterations 1000 10000 30000 40000 \
  --save_iterations 30000 40000 \
  --resolution 4 \
  --eval \
  > logs/vinyes_20260509_train.log 2>&1

echo "[3/5] Training finished at $(date -Is)"


echo "[4/5] Render started at $(date -Is)"

CUDA_VISIBLE_DEVICES=0 python render.py \
  -m output/vinyes_20260509_sam3 \
  --iteration 40000 \
  --quiet \
  > logs/vinyes_20260509_render.log 2>&1

echo "[4/5] Render finished at $(date -Is)"


echo "[5/5] Metrics started at $(date -Is)"

CUDA_VISIBLE_DEVICES=0 python metrics.py \
  -m output/vinyes_20260509_sam3 \
  > logs/vinyes_20260509_metrics.log 2>&1

echo "[5/5] Metrics finished at $(date -Is)"
