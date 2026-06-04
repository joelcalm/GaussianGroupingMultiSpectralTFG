# Temporal Vine Tracking Milestone

This milestone starts a temporal tracking pipeline across three independently trained vineyard scenes:

- `vinyes_20260321`
- `vinyes_20260418`
- `vinyes_20260509`

It intentionally does not merge, transform, or edit Gaussian models. The implemented pieces are a selected-frame 2D tracking demo and COLMAP sparse-point Sim(3) alignment utilities.

## Repository Audit

Existing loader and scene conventions:

- COLMAP loading is implemented in `scene/colmap_loader.py` and used by `scene/dataset_readers.py`.
- Training scene loading expects a prepared COLMAP-style scene with `images/`, `sparse/0/`, and optional `object_mask/`; `images_rgb/` is present for RGB-only frame selection in the vineyard datasets.
- `scene/dataset_readers.py` reads `sparse/0/images.bin`/`cameras.bin` first and falls back to `images.txt`/`cameras.txt`; it reads `points3D.ply` or converts `points3D.bin`/`points3D.txt`.
- Mask loading joins by image stem: an image like `rgb_00094.png` uses `object_mask/rgb_00094.png`.
- Existing mask metadata lives under `metadata/instance_label_map.json`, `metadata/class_map.json`, and, for hierarchical vineyard labels, `metadata/hierarchical_label_schema.json`.
- Existing trained outputs and render outputs live under `/home/msiau/data/tmp/jcalm/output/<model_name>/`, with `point_cloud/iteration_*`, `train/ours_*`, `test/ours_*`, and `selections/` artifacts.

Discovered paths used by the example config:

| Scene | Source scene | COLMAP sparse | Mask folder | Trained output | Gaussian / point cloud evidence |
| --- | --- | --- | --- | --- | --- |
| `vinyes_20260321` | `/home/msiau/data/tmp/jcalm/data/vinyes_20260321` | `/home/msiau/data/tmp/jcalm/data/vinyes_20260321/sparse/0` | `/home/msiau/data/tmp/jcalm/data/vinyes_20260321/object_mask` | `/home/msiau/data/tmp/jcalm/output/vinyes_20260321` | `/home/msiau/data/tmp/jcalm/output/vinyes_20260321/selections/point_cloud_3_rows.ply` |
| `vinyes_20260418` | `/home/msiau/data/tmp/jcalm/data/vinyes_20260418_rgb_colmap_shared` | `/home/msiau/data/tmp/jcalm/data/vinyes_20260418_rgb_colmap_shared/sparse/0` | `/home/msiau/data/tmp/jcalm/data/vinyes_20260418_rgb_colmap_shared/object_mask` | `/home/msiau/data/tmp/jcalm/output/vinyes_20260418` | `/home/msiau/data/tmp/jcalm/output/vinyes_20260418/selections/point_cloud_3_rows.ply` |
| `vinyes_20260509` | `/home/msiau/data/tmp/jcalm/data/vinyes_20260509_pinhole` | `/home/msiau/data/tmp/jcalm/data/vinyes_20260509_pinhole/sparse/0` | `/home/msiau/data/tmp/jcalm/data/vinyes_20260509_pinhole/object_mask` | `/home/msiau/data/tmp/jcalm/output/vinyes_20260509` | `/home/msiau/data/tmp/jcalm/output/vinyes_20260509/selections/point_cloud_3_rows.ply` |

Frame and mask counts found through the config:

- `vinyes_20260321`: 200 RGB frames, 200 `object_mask` masks, 200 `semantic_mask` masks, 200 `sam3_instance_mask` masks, 1576 total staged images via symlinked `images/`.
- `vinyes_20260418`: 258 RGB frames, 258 `object_mask` masks, 258 `semantic_mask` masks, 1458 total staged images.
- `vinyes_20260509`: 200 RGB frames, 200 `object_mask` masks, 200 `semantic_mask` masks, 1327 total staged images.

Important mask detail: April and May use hierarchical flat labels where one physical vine can be split into leaf/trunk/other label IDs. The selected-frame demo reads `metadata/instance_label_map.json`, filters labels whose metadata identifies them as vine labels, and merges labels sharing the same physical `instance_id` before assigning temporal IDs. March uses SAM3 vine track labels and is handled through the same metadata path.

## Why Direct Gaussian Merging Is Not Done Yet

The scenes were reconstructed and trained independently, so their COLMAP coordinate systems are independent. Their masks and object IDs are local to each scene and, in some runs, local to SAM3 tracklets or hierarchical label compositions. Their learned Gaussian features and classifiers are also trained independently. For those reasons, this milestone only produces 2D selected-frame associations and a COLMAP sparse alignment utility from manual 3D correspondences.

## Added Files

- `config/temporal_tracking/vinyes_3_dates.yaml`
- `tools/temporal_tracking/__init__.py`
- `tools/temporal_tracking/config.py`
- `tools/temporal_tracking/list_frames.py`
- `tools/temporal_tracking/track_selected_frames_2d.py`
- `tools/temporal_tracking/estimate_sim3_from_correspondences.py`
- `tools/temporal_tracking/export_aligned_colmap_points.py`
- `docs/temporal_vine_tracking.md`

No core training code was modified.

## How To Run

From `/home/msiau/workspace/jcalm`:

```bash
python tools/temporal_tracking/list_frames.py \
  --config config/temporal_tracking/vinyes_3_dates.yaml
```

This writes:

```text
<output_dir>/frame_lists/<scene>_frames.csv
```

Run the selected-frame 2D demo:

```bash
python tools/temporal_tracking/track_selected_frames_2d.py \
  --config config/temporal_tracking/vinyes_3_dates.yaml
```

The example config currently selects:

- `vinyes_20260321`: `rgbp_00063.png`
- `vinyes_20260418`: `frame_00073.jpg` (`images_rgb` has `.jpg`, not `.png`)
- `vinyes_20260509`: `rgb_00025.png` (`images_rgb` uses `rgb_`, not `rgbp_`)

The demo writes:

```text
<output_dir>/instances/instances_<scene>.csv
<output_dir>/temporal_vine_ids_2d.csv
<output_dir>/figures/temporal_vine_tracking_2d.png
```

Estimate Sim(3) transforms from manual correspondences:

```bash
python tools/temporal_tracking/estimate_sim3_from_correspondences.py \
  --config config/temporal_tracking/vinyes_3_dates.yaml \
  --correspondences path/to/manual_correspondences.csv
```

Correspondence CSV format:

```csv
source_scene,source_x,source_y,source_z,ref_x,ref_y,ref_z,comment
vinyes_20260321,0.1,0.2,0.3,1.4,2.1,0.5,post_1
vinyes_20260509,0.5,0.4,0.3,1.9,2.2,0.6,trunk_1
```

This writes one JSON per source scene:

```text
<output_dir>/transforms/<source_scene>_to_<reference_scene>.json
```

Export COLMAP sparse points for visual inspection:

```bash
python tools/temporal_tracking/export_aligned_colmap_points.py \
  --config config/temporal_tracking/vinyes_3_dates.yaml
```

This always writes the reference sparse points and writes aligned source sparse points only when the corresponding transform JSON exists:

```text
<output_dir>/aligned_colmap/<reference_scene>_points_reference.ply
<output_dir>/aligned_colmap/<scene>_points_aligned.ply
```

## Method Summary

The 2D demo loads one selected RGB image per scene, loads the matching integer mask by image stem, extracts vine instances, computes centroid, bounding box, area, and bottom point, sorts visible vine instances left to right by bottom point, and assigns `vine_001`, `vine_002`, etc. across scenes. Consistent colors are used in the 3-panel figure.

The COLMAP alignment script implements Umeyama similarity alignment for:

```text
x_ref = s * R @ x_source + t
```

It saves scale, rotation, translation, RMSE, and correspondence count as JSON. The point exporter applies those JSON transforms to COLMAP sparse points only; Gaussian models are left untouched.

## Outputs Generated During Smoke Test

Using `config/temporal_tracking/vinyes_3_dates.yaml`, the smoke test wrote:

```text
/home/msiau/data/tmp/jcalm/outputs/temporal_vine_tracking/frame_lists/vinyes_20260321_frames.csv
/home/msiau/data/tmp/jcalm/outputs/temporal_vine_tracking/frame_lists/vinyes_20260418_frames.csv
/home/msiau/data/tmp/jcalm/outputs/temporal_vine_tracking/frame_lists/vinyes_20260509_frames.csv
/home/msiau/data/tmp/jcalm/outputs/temporal_vine_tracking/instances/instances_vinyes_20260321.csv
/home/msiau/data/tmp/jcalm/outputs/temporal_vine_tracking/instances/instances_vinyes_20260418.csv
/home/msiau/data/tmp/jcalm/outputs/temporal_vine_tracking/instances/instances_vinyes_20260509.csv
/home/msiau/data/tmp/jcalm/outputs/temporal_vine_tracking/temporal_vine_ids_2d.csv
/home/msiau/data/tmp/jcalm/outputs/temporal_vine_tracking/figures/temporal_vine_tracking_2d.png
/home/msiau/data/tmp/jcalm/outputs/temporal_vine_tracking/aligned_colmap/vinyes_20260418_points_reference.ply
```

The figure is `2720 x 534` RGB and non-empty. The COLMAP exporter skipped March and May aligned PLYs because no manual correspondence transforms exist yet.

## Current Limitations

- 2D temporal IDs are approximate and based on reference-frame image-position matching, not a learned temporal identity model.
- Matching assumes the selected frames show comparable row geometry and similar field-of-view coverage.
- The April masks can include large vine foliage components, so the connected-component extraction uses area filtering to avoid the worst oversized regions.
- The mask meanings differ across dates; metadata is used to normalize vine labels where possible, but this is still not a full temporal identity model.
- Optional image registration was not implemented in this first pass; the core selected-frame visualization and Sim(3) utility were prioritized.
- COLMAP alignment quality depends completely on the quality and spread of manual 3D correspondences.
- Gaussian model transformation and merging are intentionally left for a later milestone.

## Assumptions And Missing Inputs

- `vinyes_20260418` is the reference scene, as specified in the milestone config.
- The discovered output analysis run for `vinyes_20260509` points at the pinhole source scene `/home/msiau/data/tmp/jcalm/data/vinyes_20260509_pinhole`, so that is the example source path.
- Manual 3D correspondences are still needed before March and May COLMAP sparse points can be exported in April coordinates.
- If you want different selected frames, run `list_frames.py`, edit `selected_frame` in `config/temporal_tracking/vinyes_3_dates.yaml`, and rerun `track_selected_frames_2d.py`.
