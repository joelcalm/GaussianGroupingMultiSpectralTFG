# Shared COLMAP Temporal Tracking

This milestone prepares one RGB-first COLMAP coordinate system for temporal vine tracking across:

- `data/vinyes_20260321` -> `output/vinyes_20260321`
- `data/vinyes_20260418_rgb_colmap_shared` -> `output/vinyes_20260418_rgb_colmap_shared`
- `data/vinyes_20260509_pinhole` -> `output/vinyes_20260509`

It does not merge Gaussian models, transform Gaussians, or assume Gaussian IDs are consistent across dates.

## Why Shared Poses

Independently trained 3DGS scenes do not share a world coordinate frame. Each date-specific COLMAP reconstruction can choose its own origin, axes, scale, and local bundle-adjusted geometry. A vine trunk base at `(x, y, z)` in March is therefore not directly comparable to `(x, y, z)` in April or May unless the camera poses were estimated in a shared reconstruction or aligned afterward.

Gaussian IDs are also not reliable temporal identities. Even with similar imagery, separate 3DGS training runs create and optimize different Gaussian sets, with different splits, pruning, densification history, ordering, and learned features. Temporal matching should therefore start from geometry, row position, trunk base location, and masks/metadata, not Gaussian array indices.

The preferred first step is a shared RGB COLMAP model:

```text
March RGB images + April RGB images + May RGB images
-> one COLMAP sparse model
-> all registered RGB cameras in one world coordinate system
-> per-date pose splits for separate 3DGS training
```

RGB is used as the backbone because cross-date RGB feature matching is usually more stable than mixing narrowband or multispectral images into the initial feature graph.

## Added Tools

- `tools/temporal_tracking/audit_temporal_scenes.py`
- `tools/temporal_tracking/prepare_shared_colmap_rgb.py`
- `tools/temporal_tracking/export_shared_pose_splits.py`
- `tools/temporal_tracking/shared_colmap_utils.py`

The scripts reuse the repo's existing COLMAP loader in `scene/colmap_loader.py`. Training code is not modified.

## Audit The Scenes

From `/home/msiau/workspace/jcalm`:

```bash
python tools/temporal_tracking/audit_temporal_scenes.py
```

This writes:

```text
output/temporal_tracking/audit_summary.md
```

The audit reports candidate RGB folders, RGB image counts, discovered COLMAP sparse folders, binary/text model files, trained output folders, and obvious missing files. It checks conventional `sparse/0` locations directly so symlinked sparse folders such as March's posematch sparse model are detected.

## Prepare The Shared RGB Workspace

Dry-run first:

```bash
python tools/temporal_tracking/prepare_shared_colmap_rgb.py \
  --dry-run \
  --stride 1
```

Create the workspace with symlinks:

```bash
python tools/temporal_tracking/prepare_shared_colmap_rgb.py \
  --stride 1 \
  --link-mode symlink
```

Useful caps while testing:

```bash
python tools/temporal_tracking/prepare_shared_colmap_rgb.py \
  --stride 5 \
  --max-frames-per-scene 100 \
  --link-mode symlink
```

The default workspace is:

```text
output/temporal_tracking/shared_colmap_rgb/
    images/
    lists/all_images.txt
    image_manifest.csv
    image_manifest.json
    sparse/
```

Image names are prefixed to avoid collisions:

```text
20260321__rgbp_00001.png
20260418__frame_00001.jpg
20260509__rgb_00001.png
```

The script stages only RGB candidate folders by default. It prefers `images_rgb/`, then `frames_raw/rgb`, `frames_raw/RGB`, `images/`, and `input/`.

## Run COLMAP

The preparation script prints a concrete command template. A typical run is:

```bash
colmap feature_extractor \
  --database_path output/temporal_tracking/shared_colmap_rgb/database.db \
  --image_path output/temporal_tracking/shared_colmap_rgb/images \
  --image_list_path output/temporal_tracking/shared_colmap_rgb/lists/all_images.txt \
  --ImageReader.camera_model OPENCV \
  --SiftExtraction.max_image_size 3200 \
  --SiftExtraction.max_num_features 12000

colmap exhaustive_matcher \
  --database_path output/temporal_tracking/shared_colmap_rgb/database.db \
  --SiftMatching.use_gpu 0

colmap mapper \
  --database_path output/temporal_tracking/shared_colmap_rgb/database.db \
  --image_path output/temporal_tracking/shared_colmap_rgb/images \
  --image_list_path output/temporal_tracking/shared_colmap_rgb/lists/all_images.txt \
  --output_path output/temporal_tracking/shared_colmap_rgb/sparse \
  --Mapper.multiple_models 1 \
  --Mapper.ba_refine_principal_point 0
```

Existing repo COLMAP logic in `prepare_vineyard_video_colmap.py` uses similar RGB-first ideas, diagnostics, and registration concepts. This milestone keeps the temporal workspace separate so date-specific training code stays untouched.

## Verify Registration

After COLMAP writes a model, verify it:

```bash
python tools/temporal_tracking/prepare_shared_colmap_rgb.py \
  --verify \
  --model-dir output/temporal_tracking/shared_colmap_rgb/sparse/0
```

This reads `images.bin` or `images.txt` and writes:

```text
output/temporal_tracking/shared_colmap_rgb/registration_report.md
```

The report includes registered images per date prefix, whether all three dates appear in the same sparse model, and staged images that failed to register according to the manifest.

If COLMAP creates multiple models under `sparse/`, inspect the largest connected model and pass that folder as `--model-dir`, for example `sparse/1`.

## Export Per-Date Pose Splits

Once one shared model contains the dates you need:

```bash
python tools/temporal_tracking/export_shared_pose_splits.py \
  --shared-model output/temporal_tracking/shared_colmap_rgb/sparse/0
```

This writes:

```text
output/temporal_tracking/shared_pose_splits/vinyes_20260321/sparse/0
output/temporal_tracking/shared_pose_splits/vinyes_20260418_rgb_colmap_shared/sparse/0
output/temporal_tracking/shared_pose_splits/vinyes_20260509/sparse/0
```

Each split contains `cameras.txt` and `images.txt` for one date. The qvec/tvec poses are unchanged from the shared COLMAP model, so they stay in the shared world frame. By default, prefixed names are restored to original image names, which makes the sparse folders easier to reuse with the original date-specific `images_rgb/` or staged image folders. Pass `--keep-prefixed-names` if you want the split files to keep `YYYYMMDD__` names.

The exporter also copies the best available shared point file (`points3D.ply`, `points3D.bin`, or `points3D.txt`) into each split. That provides a shared-frame point cloud for later initialization or inspection; the camera/image records remain per-date.

## Multispectral Pose Transfer

Full multispectral pose transfer is intentionally left for the next milestone.

The intended strategy is:

- Register RGB frames first into the shared COLMAP model.
- Use RGB poses as the geometric backbone.
- Attach multispectral band images to corresponding RGB poses by frame index, timestamp, or existing metadata.
- Avoid putting all MS bands into COLMAP initially because spectral differences can weaken feature matching.
- If sensor offsets are known, later apply `T_world_ms = T_world_rgb @ T_rgb_to_ms`.
- If offsets are unknown, start with `T_world_ms ~= T_world_rgb` and document that approximation.

## Later Training And Tracking

The shared pose splits can later be used to train each date-specific 3DGS scene separately while keeping camera poses in one world frame. That makes vine matching by 3D proximity, row position, trunk base location, and projected masks more meaningful than matching independent Gaussian IDs.

Gaussian transformation, Gaussian merging, and 3D vine identity assignment are deliberately out of scope for this milestone.

## Files Added

- `tools/temporal_tracking/audit_temporal_scenes.py`
- `tools/temporal_tracking/prepare_shared_colmap_rgb.py`
- `tools/temporal_tracking/export_shared_pose_splits.py`
- `tools/temporal_tracking/shared_colmap_utils.py`
- `docs/shared_colmap_temporal_tracking.md`

## Files Modified

- None of the core training or scene-reader files were modified.

## Commands To Run

```bash
python tools/temporal_tracking/audit_temporal_scenes.py
python tools/temporal_tracking/prepare_shared_colmap_rgb.py --dry-run
python tools/temporal_tracking/prepare_shared_colmap_rgb.py --stride 1 --link-mode symlink
python tools/temporal_tracking/prepare_shared_colmap_rgb.py --verify
python tools/temporal_tracking/export_shared_pose_splits.py
```

## Outputs Generated

- `output/temporal_tracking/audit_summary.md`

Expected later outputs:

- `output/temporal_tracking/shared_colmap_rgb/images/`
- `output/temporal_tracking/shared_colmap_rgb/image_manifest.csv`
- `output/temporal_tracking/shared_colmap_rgb/image_manifest.json`
- `output/temporal_tracking/shared_colmap_rgb/lists/all_images.txt`
- `output/temporal_tracking/shared_colmap_rgb/registration_report.md`
- `output/temporal_tracking/shared_pose_splits/*/sparse/0/`

## Assumptions And Missing Inputs

- The May source scene for the output `vinyes_20260509` is `data/vinyes_20260509_pinhole`.
- The shared reconstruction should be RGB-only at this stage.
- The user should choose the final COLMAP matcher/camera settings and inspect whether COLMAP produced one connected model containing all three dates.
- MS-to-RGB frame correspondence metadata and any physical sensor offset are still needed before implementing multispectral pose transfer.
