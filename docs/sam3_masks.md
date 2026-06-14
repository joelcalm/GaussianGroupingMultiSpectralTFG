# SAM3 Mask Generation

SAM3 is an offline preprocessing stage. It detects, segments, and tracks vineyard components across ordered RGB frames; it is not trained jointly with the Gaussian model.

## Inputs

- ordered RGB frames, for example `data/vinyes_20260509/images_rgb`;
- a SAM3 checkpoint, for example `weights/sam3.pt`; and
- a class/prompt configuration under `config/sam3/`.

Model checkpoints and generated masks are ignored by Git.

## Vineyard Example

This is the mask-generation stage used by the May vineyard pipeline:

```bash
CUDA_VISIBLE_DEVICES=0 python sam3_vine_video.py \
  --images_dir data/vinyes_20260509/images_rgb \
  --output_dir data/vinyes_20260509/sam3_rgb \
  --model weights/sam3.pt \
  --class_config config/sam3/vine_parts_posts.json \
  --save_class_outputs \
  --device 0 --imgsz 1024 --conf 0.25 \
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
  --overlay_alpha 0.45 --fps 5
```

Inspect overlays and track IDs before training. Cross-view consistency matters more than isolated single-frame quality for the object-feature branch.

## Compose Training Labels

Convert SAM3 outputs into class-aware instance masks and metadata:

```bash
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
  --overwrite
```

Generated integer masks must match registered image stems. Keep class maps, instance maps, and active-channel metadata alongside the scene.

## Manual Evaluation Masks

The scripts in `tools/manual_eval/` support selecting frames, correcting target vines, and building the 15-frame held-out subset used in the report. These corrections are pseudo-ground truth, not exhaustive biological annotations.
