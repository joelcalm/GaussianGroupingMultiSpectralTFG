# Section 4 Experimental Setup: Implementation Evidence

This document is an evidence extraction note for writing Section 4 of the TFG report. It describes what the current code, scripts, configs, metadata examples, and docs actually do. It is not final report prose.

Important scope notes:

- The current working tree is dirty. This note treats the current files in `/home/msiau/workspace/jcalm` as the implementation source of truth.
- Prepared vineyard data examples are sampled from `/home/msiau/workspace/vineyard_posematch`.
- Several scripts are experimental or dataset-specific. When behavior depends on the dataset/config/run, this is stated explicitly.

---

## 0. High-level summary of Section 4

Implementation-level pipeline:

```text
raw videos/images or existing multispectral tensors/poses
-> frame extraction / band organization / pseudo-RGB generation
-> COLMAP reconstruction or COLMAP-format pose conversion
-> optional RGB-reference band registration / path filtering
-> SAM3 or DEVA mask generation
-> object_mask/ semantic_mask/ instance metadata
-> channel metadata: active_channels.json, frame_info.json, band_info.json
-> training dataset folder with images/, sparse/0/, masks, metadata
-> train.py
```

Raw inputs required, depending on experiment:

- Standard RGB/custom data: an `input/` folder of images for `convert.py`, then `images/`, `sparse/`, and `object_mask/` for training. Implemented in [convert.py](../convert.py) and documented in [docs/train.md](train.md).
- MMS multispectral data: `meta_data.json`, `modalities/multispectral/*.npy`, and optionally `pointcloud.ply`. Converted by [prepare_mms_dataset.py](../prepare_mms_dataset.py), `main`.
- Spec-NeRF data: existing COLMAP `sparse/0/*.bin`, per-pose TIFF channel stacks, and optional point cloud from `points3D.bin`. Converted by [prepare_specnerf_dataset.py](../prepare_specnerf_dataset.py), `main`.
- X-NeRF data: `ms_imgs.npy` and `rgb_poses.npy`. Converted by [prepare_xnerf_dataset.py](../prepare_xnerf_dataset.py), `main`.
- Real vineyard data: RGB and narrowband videos selected by filename tokens. Prepared by [prepare_vineyard_video_colmap.py](../prepare_vineyard_video_colmap.py), `main`.
- Vineyard SAM3 experiments: a prepared COLMAP/multispectral scene plus SAM3 output masks. Combined by [prepare_vinyes_sam3_200.py](../prepare_vinyes_sam3_200.py), `main`.

Training code expects a scene folder that `Scene` recognizes as COLMAP if it contains `sparse/`; see [scene/__init__.py](../scene/__init__.py), `Scene.__init__`. It loads:

- Images from `images/` by default, or another folder passed with `--images`.
- Camera poses and intrinsics from `sparse/0/cameras.bin` plus `images.bin`, falling back to `cameras.txt` plus `images.txt`; see [scene/dataset_readers.py](../scene/dataset_readers.py), `readColmapSceneInfo`.
- Optional masks from `object_mask/` by default, or another folder passed with `--object_path`; see [scene/dataset_readers.py](../scene/dataset_readers.py), `readColmapSceneInfo` and `readColmapCameras`.
- Optional full multispectral tensors from `images_multispectral/<stem>.npy`; see [scene/dataset_readers.py](../scene/dataset_readers.py), `readColmapCameras`, and [utils/camera_utils.py](../utils/camera_utils.py), `loadCam`.
- Optional active-channel metadata from `metadata/active_channels.json`, then `frame_info.json`, then `band_info.json`; see [scene/dataset_readers.py](../scene/dataset_readers.py), `_load_active_channel_map`.

How images, poses, masks, and metadata are linked:

- COLMAP `images.bin/txt` supplies image names. For each COLMAP image name, `readColmapCameras` looks for `images/<basename>`, `object_mask/<stem>.png`, `images_multispectral/<stem>.npy`, and active-channel metadata keyed by `<stem>`.
- The image stem is therefore the join key across RGB/band images, masks, `.npy` tensors, and JSON metadata.
- If no metadata entry exists, `_infer_active_channels` infers channels from filename prefixes: `rgb* -> (0,1,2)`, `b470 -> 3`, `b505 -> 4`, `b525 -> 5`, `b590 -> 6`, `b635 -> 7`, `b660 -> 8`, `b850 -> 9`.

Common to all experiments:

- Training uses the same `train.py` loop, `Scene` loader, `Camera` objects, COLMAP scene reader, Gaussian initialization, photometric loss, optional object-mask loss, and optional 3D object regularization.
- Object features are learned as continuous `num_objects`-dimensional Gaussian attributes; the classifier maps rendered object features to `num_classes`.
- Active channels restrict photometric supervision to channels known to be present for a view.

What changes between configurations:

- RGB-only: `num_channels=3`, active channels normally `[0,1,2]`, standard RGB images.
- Partial RGB or single-channel mode: `single_channel_mode=true` in config causes `train.py` to choose one active channel per iteration; currently this is implemented generally, not only for RGB.
- Multispectral tensor datasets: `images_multispectral/*.npy` stores full `H,W,C` float tensors; configs use `num_channels=9`, `10`, or `20`, but CUDA must be rebuilt to match.
- Real vineyard partial-band experiments: each RGB or narrowband frame is a separate image in one shared COLMAP scene. Per-image active channels identify which output channels are supervised.
- SAM3/vine-id configurations: mask labels and `num_classes` depend on SAM3 output and label compaction or hierarchical composition.

---

## 1. Details for Section 4.1: Common processing pipeline

### 1.1 Expected final dataset structure

Minimum COLMAP-style training scene:

```text
scene/
  images/                         required by default, unless --images points elsewhere
    <image_name>.png|jpg|...
  sparse/
    0/
      cameras.bin|cameras.txt     required
      images.bin|images.txt       required
      points3D.bin|points3D.txt   required unless points3D.ply already exists
      points3D.ply                optional; created by loader if absent and points3D bin/txt exists
  object_mask/                    optional per-view object labels, default mask folder
    <image_stem>.png
  images_multispectral/           optional full-channel tensors
    <image_stem>.npy
  metadata/                       optional label/channel metadata
    active_channels.json
    class_map.json
    class_colors.json
    instance_label_map.json
    registered_images_summary.json
  frame_info.json                 optional active-channel metadata at scene root
  band_info.json                  optional active-channel metadata at scene root
  colmap_quality.json             optional guard checked by train.py
```

Vineyard SAM3 example sampled from `/home/msiau/workspace/vineyard_posematch/vinyes_sam3_vineid_200`:

```text
vinyes_sam3_vineid_200/
  images/                 1576 registered RGB/narrowband images
  images_rgb/             200 RGB frames used by SAM3
  object_mask/            200 training label masks, RGB views only
  semantic_mask/          200 semantic label masks
  sam3_instance_mask/     200 raw aligned SAM3 instance masks
  sparse/0/               4 COLMAP text/model files in this sampled scene
  metadata/               9 metadata files
  frame_info.json         1576 entries
  band_info.json          1576 entries
  partial_channels_summary.json
```

Implementation references:

- Dataset selection: [scene/__init__.py](../scene/__init__.py), `Scene.__init__`.
- COLMAP scene loading: [scene/dataset_readers.py](../scene/dataset_readers.py), `readColmapSceneInfo`.
- Camera and tensor loading: [utils/camera_utils.py](../utils/camera_utils.py), `loadCam`.
- Vineyard SAM3 scene assembly: [prepare_vinyes_sam3_200.py](../prepare_vinyes_sam3_200.py), `main`.

### 1.2 Image folders used

- RGB or pseudo-RGB image folder: `images/` by default. The CLI parameter is `--images`; default is set in [arguments/__init__.py](../arguments/__init__.py), `ModelParams._images = "images"`.
- Real vineyard RGB-only mask-generation folder: `images_rgb/`, created by [prepare_vineyard_video_colmap.py](../prepare_vineyard_video_colmap.py), `extract_video_frames`, and linked/copied by [prepare_vinyes_sam3_200.py](../prepare_vinyes_sam3_200.py), `main`.
- Multispectral tensor folder: `images_multispectral/`, optional, detected in [scene/dataset_readers.py](../scene/dataset_readers.py), `readColmapSceneInfo`.
- Mask folder: `object_mask/` by default. Configurable with `--object_path`; default in [arguments/__init__.py](../arguments/__init__.py), `ModelParams._object_path = "object_mask"`.
- Metadata folder: `metadata/`, optional. Label metadata is copied into the model output by [train.py](../train.py), `copy_label_metadata`.
- COLMAP sparse folder: `sparse/0/`.

Dataset-specific note:

- The sampled `vinyes_pose800` scene has `input/` rather than `images/`, so it would need `--images input` for this loader unless another linked/copied `images/` folder exists. This is not inferred automatically by `Scene`.

### 1.3 Formats expected

RGB/pseudo-RGB images:

- Loaded with PIL via `Image.open`; PNG/JPG and other PIL-supported formats can work.
- Vineyard extraction supports `--image_ext png|jpg`; see [prepare_vineyard_video_colmap.py](../prepare_vineyard_video_colmap.py), `parse_args`.
- COLMAP/SAM3 prep scripts mostly assume `.png`, `.jpg`, `.jpeg` when parsing image lists.

Multispectral images:

- If `images_multispectral/<stem>.npy` exists, it is loaded with `np.load` as `[H, W, C]` and converted to torch `[C, H, W]`; see [utils/camera_utils.py](../utils/camera_utils.py), `loadCam`.
- The loader comment says float32 in `[0,1]`; it does not enforce normalization.
- MMS preparation normalizes uint16 by `2**16` and writes float `.npy`; see [prepare_mms_dataset.py](../prepare_mms_dataset.py), `main`.
- Spec-NeRF preparation normalizes TIFF stacks by `65535.0`; see [prepare_specnerf_dataset.py](../prepare_specnerf_dataset.py), `main`.
- X-NeRF preparation assumes `ms_imgs.npy` is already float32 `[0,1]`; see [prepare_xnerf_dataset.py](../prepare_xnerf_dataset.py), `main`.

Masks:

- Training mask path is `<object_folder>/<image_stem>.png`; see [scene/dataset_readers.py](../scene/dataset_readers.py), `readColmapCameras`.
- Masks are resized with nearest-neighbor interpolation; see [utils/camera_utils.py](../utils/camera_utils.py), `loadCam`.
- Supported mask array shapes in `_convert_object_mask_to_indices`: 2D indexed labels, single-channel 3D, grayscale RGB, or color RGB masks.
- Indexed masks can be uint8 or uint16. SAM3 preparation writes uint16 when max label is above 255; see [prepare_vinyes_sam3_200.py](../prepare_vinyes_sam3_200.py), `save_label_png`.
- For RGB color masks, colors are packed into integer IDs and remapped to compact class indices; see [utils/camera_utils.py](../utils/camera_utils.py), `_pack_rgb_ids`, `_build_rgb_object_id_mapping`, `_convert_object_mask_to_indices`.

Metadata:

- JSON.
- Active-channel metadata may be a direct list or a dict with key `"channels"`; see [scene/dataset_readers.py](../scene/dataset_readers.py), `_load_active_channel_map`.
- Keys are normalized to `Path(name).stem`, so keys may include or omit extensions.

Poses:

- COLMAP binary is tried first: `sparse/0/images.bin`, `cameras.bin`.
- Text fallback: `sparse/0/images.txt`, `cameras.txt`.
- Point cloud reads `points3D.ply` if present, otherwise converts `points3D.bin` or `points3D.txt` to PLY; see [scene/dataset_readers.py](../scene/dataset_readers.py), `readColmapSceneInfo`.

### 1.4 Preprocessing scripts that create the structure

`convert.py`

- Input: `--source_path` containing `input/`.
- Runs COLMAP feature extraction, exhaustive matching, mapper, image undistorter.
- Output: `distorted/database.db`, `distorted/sparse`, `images/`, `sparse/0/`.
- Reference: [convert.py](../convert.py), top-level script.

`prepare_pseudo_label.sh`

- Input: dataset name under `data/`, image scale.
- Runs DEVA `demo_automatic.py` twice, first for color visualization, second for gray training masks.
- Copies `Annotations` to `data/<dataset_name>/object_mask`.
- Reference: [script/prepare_pseudo_label.sh](../script/prepare_pseudo_label.sh).

`prepare_mms_dataset.py`

- Input: MMS `meta_data.json`, multispectral `.npy`, optional point cloud.
- Writes `sparse/0/cameras.txt`, `images.txt`, empty `points3D.txt`, optional `points3D.ply`, pseudo-RGB `images/*.png`, and `images_multispectral/*.npy`.
- Reference: [prepare_mms_dataset.py](../prepare_mms_dataset.py), `main`.

`prepare_specnerf_dataset.py`

- Input: raw Spec-NeRF COLMAP `.bin`, pose directories with TIFF channels.
- Writes text COLMAP files, converts `points3D.bin` to PLY, writes pseudo-RGB PNG and 20-channel `.npy`.
- Reference: [prepare_specnerf_dataset.py](../prepare_specnerf_dataset.py), `main`.

`prepare_xnerf_dataset.py`

- Input: `ms_imgs.npy`, `rgb_poses.npy`.
- Writes COLMAP text files from known poses, empty `points3D.txt`, pseudo-RGB PNG, and per-view `.npy`.
- Reference: [prepare_xnerf_dataset.py](../prepare_xnerf_dataset.py), `main`.

`prepare_vineyard_video_colmap.py`

- Input: vineyard RGB/narrowband videos.
- Extracts frames into `frames_raw/<band>/`, stages images in `images/`, writes `images_rgb/` for RGB, creates image lists and match pairs, runs RGB COLMAP variants, optionally registers narrowband images to the selected RGB model, converts final model to `sparse/0`, and writes channel/COLMAP metadata.
- Reference: [prepare_vineyard_video_colmap.py](../prepare_vineyard_video_colmap.py), `main`.

`sam3_vine_video.py`

- Input: ordered RGB image sequence.
- Runs SAM3 video prediction class by class, with text prompts, and writes semantic, instance, binary, color, overlay, and metadata outputs.
- Reference: [sam3_vine_video.py](../sam3_vine_video.py), `main`.

`prepare_vinyes_sam3_200.py`

- Input: prepared source scene and SAM3 output directory.
- Links/copies `images/`, `images_rgb/`, `sparse/`, copies root metadata, aligns SAM3 masks for registered RGB images, creates `object_mask/`, `semantic_mask/`, optional `sam3_instance_mask/`, `metadata/active_channels.json`, label maps, reports, and a config.
- Reference: [prepare_vinyes_sam3_200.py](../prepare_vinyes_sam3_200.py), `main`.

`compose_hierarchical_vineyard_labels.py`

- Input: prepared scene plus SAM3 class-level outputs saved with `--save_class_outputs`.
- Replaces/creates `object_mask/` with flat composite labels that encode object type, physical instance, and part in metadata.
- Reference: [compose_hierarchical_vineyard_labels.py](../compose_hierarchical_vineyard_labels.py), `main`.

`filter_colmap_scene_by_path.py`

- Input: prepared COLMAP vineyard scene.
- Filters registered band images by nearest distance to reference RGB camera path, rewrites `sparse/0`, filters point tracks, links images/masks, and filters channel metadata.
- Reference: [filter_colmap_scene_by_path.py](../filter_colmap_scene_by_path.py), `main`.

### 1.5 Train/test splits

Implemented in [scene/dataset_readers.py](../scene/dataset_readers.py), `readColmapSceneInfo`.

- If `--eval` is false and `--train_split` is false: all COLMAP images are training images; no test set.
- If `--eval` is true and `--train_split` is false: images sorted by `image_name`; every `llffhold`-th image goes to test, all others to train. The default `llffhold` is 8 in the function signature.
- If `--eval` is true and `--train_split` is true and `images_train/` exists: image stems in `images_train/` define train views; all other COLMAP images are test views.
- If `--eval` is true and `--train_split` is true but `images_train/` is missing: falls back to index-based split.
- If `--eval` is false and `--train_split` is true and `images_train/` exists: only images listed in `images_train/` are used for training; test is empty.
- `n_views` can subsample the train set after the split. Default `n_views=100` means no subsampling in this implementation.

Script evidence:

- `script/train_vinyes_sam3_200.sh` trains with `--eval`, `--resolution 4`, `--iterations 40000`, test iterations `1000 10000 30000 40000`, save iterations `30000 40000`.
- `script/train_mms.sh` trains with `--eval`, `-r 2`, and configurable iterations/config.

### 1.6 How RGB and multispectral views are combined

There are two different mechanisms:

1. Full-tensor multispectral views:
   - `images/<stem>.png` is pseudo-RGB/fallback.
   - `images_multispectral/<stem>.npy` contains all channels.
   - The camera list has one camera per view. The whole tensor is loaded into that camera.
   - Used by MMS, Spec-NeRF, and X-NeRF prep scripts.

2. Real vineyard partial-channel views:
   - Each RGB or narrowband frame is its own image in the shared COLMAP `images/` folder.
   - The same Gaussian scene is trained with all registered images.
   - Active-channel metadata says which output channels each image supervises.
   - Example: `vinyes_sam3_vineid_200` has 1576 `images/`, 200 `images_rgb/`, and active-channel metadata for all 1576 image stems.

### 1.7 Common across experimental configurations

- COLMAP-compatible `sparse/0` scene format.
- `Scene` creates one shared Gaussian model from the scene point cloud.
- Camera objects store pose, intrinsics, image tensor, optional object mask, and active-channel list.
- Image loss is L1 plus DSSIM, restricted to active channels unless no active channels exist.
- Object loss is cross-entropy from a `1x1` classifier over rendered object features, only for views with `objects is not None`.
- 3D object regularization periodically samples Gaussian object predictions.
- Output saving uses `point_cloud/iteration_<N>/point_cloud.ply`, `classifier.pth`, and optionally `color_decoder.pth`.

### 1.8 Changes between configurations

RGB-only:

- `num_channels=3`.
- Images are RGB PNG/JPG, active channels usually `[0,1,2]`.
- Example configs: [config/train_color_embed.json](../config/train_color_embed.json), [config/train_single_channel.json](../config/train_single_channel.json).

Partial RGB / single-channel:

- `single_channel_mode=true` causes `train.py` to choose one active channel at random for the photometric loss each iteration.
- Implemented in [train.py](../train.py), `training`.

Multispectral:

- `num_channels` depends on dataset: 9 for MMS, 20 for Spec-NeRF, 10 for vineyard/X-NeRF configs.
- Full `.npy` tensors or per-band partial images can be used.
- Config examples: [config/gaussian_dataset/train_mms.json](../config/gaussian_dataset/train_mms.json), [config/gaussian_dataset/specnerf_baseline.json](../config/gaussian_dataset/specnerf_baseline.json).

Real vineyard:

- Uses RGB-reference COLMAP variants and optional band registration in `prepare_vineyard_video_colmap.py`.
- Uses channel map `rgb -> [0,1,2]`, `b470 -> [3]`, `b505 -> [4]`, `b525 -> [5]`, `b590 -> [6]`, `b635 -> [7]`, `b660 -> [8]`, `b850 -> [9]`.
- SAM3 masks normally exist only for RGB-like frames, not narrowband frames.

SAM3/vine-id:

- `num_classes` is generated from the chosen label mode and mask compaction.
- Example configs: `vinyes_sam3_200` has 348 classes; `vinyes_sam3_vineid_200` has 362 classes; `vinyes_20260509_sam3` has 1758 classes.

Report-ready bullets for 4.1:

- All experiments are converted into a COLMAP-style scene folder with `images/`, `sparse/0/`, optional masks, and optional channel metadata.
- `train.py` does not load videos directly; videos are first extracted and organized by preparation scripts.
- The COLMAP image name stem is the join key for image pixels, masks, multispectral tensors, and active-channel metadata.
- Multispectral supervision is implemented either as full `.npy` tensors or as separate per-band images with active-channel metadata.
- Object supervision is optional per view. Views without `object_mask/<stem>.png` still contribute photometric loss.
- Train/test splits are created by the loader from `--eval`, `--train_split`, `images_train/`, and the default LLFF holdout stride of 8.
- All configurations share the same Gaussian model, camera construction, loss loop, and output format.
- Dataset-specific differences are mostly in preprocessing, channel count, active-channel maps, and label-map generation.

---

## 2. Details for Section 4.2: Camera pose estimation

### 2.1 Pose estimation method

Methods present in the codebase:

- Standard custom RGB datasets: COLMAP from `convert.py`.
- MMS/Spec-NeRF/X-NeRF: existing poses are converted into COLMAP text format by preparation scripts.
- Real vineyard: `prepare_vineyard_video_colmap.py` builds RGB-only COLMAP variants, selects a variant by diagnostics, and either uses RGB only or registers narrowband images into the selected RGB model.
- Older/sampled `vinyes_pose800`: metadata contains `matched_bands` per RGB frame in `frame_info.json`, but the script that produced it is not present in the inspected repo. Treat this as existing dataset metadata, not current implementation evidence.

### 2.2 COLMAP files expected

Loaded by [scene/dataset_readers.py](../scene/dataset_readers.py), `readColmapSceneInfo`:

- `sparse/0/cameras.bin` and `sparse/0/images.bin`, tried first.
- Fallback: `sparse/0/cameras.txt` and `sparse/0/images.txt`.
- Point initialization:
  - `sparse/0/points3D.ply` if present.
  - Otherwise `sparse/0/points3D.bin`, fallback `points3D.txt`, converted to `points3D.ply`.

### 2.3 How COLMAP poses are loaded

Implemented in [scene/dataset_readers.py](../scene/dataset_readers.py), `readColmapCameras`.

- Iterates over COLMAP extrinsics.
- Gets intrinsics by `extr.camera_id`.
- Converts quaternion to rotation using `qvec2rotmat(extr.qvec)` and transposes the result.
- Uses COLMAP translation vector `extr.tvec`.
- Supports camera models: `SIMPLE_PINHOLE`, `PINHOLE`, `OPENCV`, `OPENCV_FISHEYE`, `FOV`, `SIMPLE_RADIAL`, `RADIAL`, `SIMPLE_RADIAL_FISHEYE`, `RADIAL_FISHEYE`.
- Converts focal lengths to FoV with `focal2fov`.
- Creates `CameraInfo`, then [utils/camera_utils.py](../utils/camera_utils.py), `loadCam`, converts it into [scene/cameras.py](../scene/cameras.py), `Camera`.
- `Camera` builds CUDA `world_view_transform`, `projection_matrix`, `full_proj_transform`, and `camera_center`.

### 2.4 Gaussian position initialization

Implemented in:

- [scene/dataset_readers.py](../scene/dataset_readers.py), `fetchPly`, `storePly`, `readColmapSceneInfo`.
- [scene/__init__.py](../scene/__init__.py), `Scene.__init__`.
- [scene/gaussian_model.py](../scene/gaussian_model.py), `GaussianModel.create_from_pcd`.

Behavior:

- If `random_init=false`, the loader uses `sparse/0/points3D.ply` or creates it from COLMAP `points3D.bin/txt`.
- `Scene.__init__` copies the initial PLY to `model_path/input.ply`.
- `GaussianModel.create_from_pcd` initializes Gaussian positions from the point cloud, color SH coefficients from point colors, random object features, scale from nearest-neighbor distance, identity rotations, and initial opacity `0.1`.
- If `random_init=true`, `readColmapSceneInfo` creates 100,000 random points in a fixed cube and writes `points3D_randinit.ply`.

### 2.5 Real vineyard pose strategy

Current script behavior in [prepare_vineyard_video_colmap.py](../prepare_vineyard_video_colmap.py):

- Default `registration_mode` is `rgb_register`.
- Frames are extracted from one or more videos per band. RGB can use `--rgb_frames_per_video` and dense ranges; non-RGB bands use `--frames_per_video`.
- RGB-only COLMAP variants are run first. Default variants are:
  - `exhaustive:OPENCV`
  - `exhaustive:FOV`
  - `sequential_loop:OPENCV`
  - `sequential_loop:FOV`
- Each RGB variant runs feature extraction, matching, mapper, diagnostics, and writes `variant_config.json`.
- `select_rgb_variant` chooses the best diagnostic by quality flag, registered image count, median observations, and reprojection error.
- If the selected RGB model fails quality checks and `--allow_bad_colmap` is not set, finalization stops.
- For `rgb_register`, non-RGB features are added to the selected database and `colmap image_registrator` registers bands to the fixed RGB model.
- For `rgb_only`, final scene uses the selected RGB model only.
- For `direct`, the script can run one direct all-image reconstruction using `matches_importer` pairs and `mapper`.
- Finalization converts the selected/registered model to text under `sparse/0`.

Run-log evidence from [colmap_run.log](../colmap_run.log):

- A vineyard run extracted 600 RGB frames and 200 frames for each of `b470`, `b505`, `b525`, `b590`, `b635`, `b660`, `b850`.
- It ran an RGB COLMAP variant `exhaustive / OPENCV` using `--ImageReader.camera_model OPENCV`, `--SiftExtraction.max_image_size 3200`, `--SiftExtraction.max_num_features 12000`, affine shape and domain-size pooling enabled.
- The inspected log is long and does not provide a clean final registration summary in the sampled lines. Do not cite final quality from it without checking the final metadata.

### 2.6 How multispectral bands are associated with camera poses

In the current vineyard COLMAP script:

- Band/frame association is filename-index based.
- Staged filenames are stable stems like `rgb_00001`, `b470_00001`, etc.; see [prepare_vineyard_video_colmap.py](../prepare_vineyard_video_colmap.py), `extract_video_frames`.
- `make_pairs` pairs same-band neighboring frames and pairs band frame index `i` with RGB frames around `i` using `cross_band_window` or direct-mode radius.
- Registered COLMAP output contains the final pose for each registered image name.
- Active channels are written per stem by `write_scene_metadata`.

In sampled `vinyes_sam3_vineid_200`:

- `metadata/active_channels.json`, `frame_info.json`, and `band_info.json` have 1576 entries.
- Example: `b470_00001 -> [3]`.
- RGB-like entries map to `[0,1,2]`.

In sampled `vinyes_pose800`:

- `frame_info.json` entries for `rgb_00001` contain `channels: [0,1,2,3,4,5,6,7,8,9]` plus a `matched_bands` dict with nearest band stems and translation/rotation checks.
- The current `scene/dataset_readers.py` only consumes the `channels` field from this metadata. It does not consume `matched_bands` during training.

### 2.7 COLMAP problems observed or handled

Handled explicitly in code:

- Low registered fraction: `write_colmap_diagnostics` flags `low_registered_fraction`.
- High mean reprojection error: flags `high_mean_reprojection_error`.
- Folded/V-shaped trajectory: flags `folded_trajectory_line_angle` if the fitted angle exceeds `--bad_trajectory_line_angle`.
- Weak images near turn: records `weakly_registered_images_near_turn`.
- Bad COLMAP quality can prevent both scene finalization and training:
  - Prep script refuses finalization unless `--allow_bad_colmap`.
  - `train.py` refuses training if `source_path/colmap_quality.json` says `quality_ok=false`, unless `--allow_bad_colmap`.
- Path filtering can remove band images whose camera centers are far from the RGB reference trajectory; see [filter_colmap_scene_by_path.py](../filter_colmap_scene_by_path.py), `main`.

Not directly implemented as named checks:

- Moving vegetation.
- Band-specific exposure/appearance mismatch beyond the registration/pairing strategy.
- Biological vine identity ambiguity.

### 2.8 Grayscale/RGB conversion before COLMAP

In [prepare_vineyard_video_colmap.py](../prepare_vineyard_video_colmap.py):

- `--grayscale_colmap_bands` exists and defaults to false.
- If true, non-RGB staged images are written as single-channel grayscale for COLMAP/training while `frames_raw` remains unchanged.
- If false, frames are staged as extracted/copy-linked PNG or JPG.

In training:

- If no `.npy` exists, images are loaded through PIL and `PILtoTorch`; the first three channels are used before expansion into the full channel tensor.
- For a single active narrowband channel, `_expand_active_channels` takes the first loaded channel and places it into the specified output channel index.

### 2.9 Limitations to state

- Pose quality is a prerequisite for stable training; the trainer has only a quality-file guard, not pose correction.
- RGB-reference registration assumes band images can be registered into a shared geometry.
- Some registered band frames may be far from the RGB trajectory; `filter_colmap_scene_by_path.py` exists to remove them, but it is a separate step.
- In repeated vineyard rows, trajectory quality can be ambiguous; the current code uses folded-trajectory and registration-fraction diagnostics, not semantic reasoning about rows.
- SAM3 masks are produced on RGB frames, so narrowband-only views usually lack object supervision.
- The training formulation assumes a static scene.

What the report should say:

The current real-vineyard pipeline estimates camera geometry with COLMAP, using RGB frames as the reference reconstruction and optionally registering narrowband frames into that model. The training loader consumes the final `sparse/0` COLMAP model and treats every registered RGB or band image as a camera in one shared Gaussian scene. Gaussian positions are initialized from COLMAP `points3D`; if `points3D.ply` is missing, it is generated from COLMAP binary/text points. Pose quality is checked by preprocessing diagnostics and can block training when marked bad.

Exact implementation references:

- `convert.py`: standard COLMAP conversion.
- `prepare_vineyard_video_colmap.py`: vineyard RGB-reference COLMAP, registration, diagnostics, metadata.
- `filter_colmap_scene_by_path.py`: camera path filtering.
- `scene/dataset_readers.py`: COLMAP file loading and point cloud conversion.
- `scene/__init__.py`: scene detection, `input.ply`, `cameras.json`, Gaussian initialization.
- `scene/gaussian_model.py`: initialization from point cloud.
- `train.py`: COLMAP quality guard.

---

## 3. Details for Section 4.3: Mask generation and object IDs

### 3.1 Mask-generation methods

Methods present:

- DEVA automatic masks for generic datasets: [script/prepare_pseudo_label.sh](../script/prepare_pseudo_label.sh).
- SAM3 video semantic prompting for vineyard scenes: [sam3_vine_video.py](../sam3_vine_video.py).
- SAM3 mask alignment and object-mask scene preparation: [prepare_vinyes_sam3_200.py](../prepare_vinyes_sam3_200.py).
- Hierarchical vineyard labels from SAM3 class instance masks: [compose_hierarchical_vineyard_labels.py](../compose_hierarchical_vineyard_labels.py).

SAM3 script details:

- Uses `ultralytics.models.sam.SAM3VideoSemanticPredictor`.
- Default classes: `background`, `vine_plant`, `wooden_post`, `ground`, `sky`, `tree`, `stone_wall`, `shrub_or_other_vegetation`, `building`.
- Default prompts are defined in `CLASS_PROMPTS`.
- The video result boxes are interpreted as `[x1, y1, x2, y2, track_id, score, class]`.
- Global instance IDs are assigned per `(class_name, sam3 track_id)`.

### 3.2 Mask types in the dataset

Observed/implemented folders:

| Folder | Meaning | Produced by | Used directly by training |
|---|---|---|---|
| `object_mask/` | Training label mask folder, label meaning depends on preparation mode | DEVA, `prepare_vinyes_sam3_200.py`, or `compose_hierarchical_vineyard_labels.py` | Yes, by default |
| `semantic_mask/` | Semantic class indexed masks aligned to registered RGB images | `prepare_vinyes_sam3_200.py` | Only if `--object_path semantic_mask` or analysis script requests semantic labels |
| `sam3_instance_mask/` | Raw aligned SAM3 global instance/tracklet IDs | `prepare_vinyes_sam3_200.py` | No, unless copied/selected into `object_mask/` |
| `semantic_index_masks/` | SAM3 semantic outputs before scene alignment | `sam3_vine_video.py` | No |
| `semantic_instance_masks/` | SAM3 tracked instance outputs before scene alignment | `sam3_vine_video.py` | No |
| `semantic_color_masks/` | Color previews | `sam3_vine_video.py` | No |
| `vine_binary_masks/` | Binary vine masks | `sam3_vine_video.py` | No |
| `class_binary_masks/` | Optional per-class binary masks | `sam3_vine_video.py --save_class_outputs` | Used by hierarchical composer, not directly by trainer |
| `class_instance_masks/` | Optional per-class instance masks | `sam3_vine_video.py --save_class_outputs` | Used by hierarchical composer, not directly by trainer |

### 3.3 Mask folder used during training

- Default: `object_mask`.
- Default set in [arguments/__init__.py](../arguments/__init__.py), `ModelParams._object_path`.
- Used in [scene/dataset_readers.py](../scene/dataset_readers.py), `readColmapSceneInfo`, as `object_dir = 'object_mask' if object_path == None else object_path`.
- Analysis can intentionally use `semantic_mask`; see [analysis/multispectral_separability.py](../analysis/multispectral_separability.py), `make_load_args`.

### 3.4 Label meanings

Label meaning depends on preprocessing:

- DEVA masks: automatic tracking/segmentation labels from DEVA output. The script copies DEVA `Annotations` to `object_mask`.
- SAM3 semantic mode: labels are semantic classes from `class_map.json`.
- SAM3 raw instance mode: labels are global SAM3 tracklet IDs from `semantic_instance_masks`.
- SAM3 compact instance mode: selected classes, e.g. `vine_plant`, keep individual track IDs; other tracked classes fold back to semantic class IDs.
- Weak connected-component mode: if SAM3 instance masks are unavailable, `prepare_vinyes_sam3_200.py` builds weak per-frame connected-component tracks from semantic masks using bbox IoU.
- Hierarchical mode: flat integer IDs encode semantic class, physical instance candidate, and part in `instance_label_map.json`.

Concrete sampled examples:

- `vinyes_sam3_200`: `instance_tracking_report.json` source is `semantic_index_masks_connected_components`; 348 labels/classes in sampled metadata.
- `vinyes_sam3_vineid_200`: report source is `sam3_video_tracker_compact`; keeps 353 `vine_plant` instances and folds other tracks; config has `num_classes=362`.

### 3.5 Mask storage

Implemented storage:

- SAM3 binary masks: 0/255 uint8 PNG via `save_binary_mask`.
- SAM3 semantic index masks: uint8 PNG via `save_index_mask`.
- SAM3 semantic instance masks: uint16 PNG via `save_instance_mask`.
- Prepared label masks: uint8 if max label <=255, otherwise uint16 `I;16`; see `save_label_png` in both SAM3 prep and hierarchical label scripts.
- RGB color masks can be converted to labels by the loader; the loader detects non-grayscale RGB and remaps packed RGB colors to compact integer labels.

### 3.6 Object ID mapping

Relevant files:

- `metadata/class_map.json`: class name to semantic class ID.
- `metadata/class_colors.json`: class ID to RGB visualization color.
- `metadata/instance_label_map.json`: flat label ID to metadata such as class, label, source, track ID, physical instance/part fields.
- `metadata/instance_tracking_report.json`: source and summary of instance generation.

Training relationship:

- `num_classes` in config must be greater than the maximum label used by the masks.
- `train.py` creates `classifier = Conv2d(num_objects, num_classes, kernel_size=1)`.
- For grayscale/indexed masks, there is no automatic clamp to `num_classes`; labels outside the classifier range would make cross-entropy invalid.
- For RGB color masks, `_build_rgb_object_id_mapping` maps at most `num_classes` RGB IDs and drops excess foreground IDs to background 0.

### 3.7 Mask availability by view

- `readColmapCameras` looks for `object_mask/<stem>.png` per registered image.
- If no mask exists, `objects=None`.
- `train.py` applies object cross-entropy only when `viewpoint_cam.objects is not None`; otherwise `loss_obj` is zero.
- In sampled vineyard SAM3 scenes, masks are available for 200 RGB-like frames while `images/` has 1576 registered RGB/narrowband images.

### 3.8 Temporal association

SAM3:

- `sam3_vine_video.py` converts the RGB frame sequence to a temporary or kept video and runs SAM3 video prediction.
- It preserves SAM3 `track_id` from result boxes and assigns global IDs per `(class_name, track_id)`.
- Semantic masks are assembled per frame by applying `MERGE_ORDER`; later classes overwrite earlier classes.

Weak fallback:

- `prepare_vinyes_sam3_200.py`, `build_instance_masks`, tracks connected components frame to frame by class and bbox IoU.
- Defaults: `instance_min_area=15000`, `instance_iou_threshold=0.10`.
- The report explicitly warns that this is weak pixel-space association when SAM3 instance masks are unavailable.

Hierarchical:

- `compose_hierarchical_vineyard_labels.py` discovers whole-vine and post instance IDs from SAM3 class instance masks, associates part components to whole-vine instances by overlap/dilation, and writes flat labels plus hierarchy metadata.

### 3.9 Terminology distinctions

- Object ID: in this code, usually the integer mask label supervised through `object_mask/`; exact meaning depends on preprocessing.
- Instance ID: a flat label identifying a SAM3 tracklet, weak connected component, compact instance label, or hierarchical object/part label.
- Class ID: semantic category ID from `class_map.json`.
- Vine ID: not a built-in training primitive unless produced by label compaction/hierarchical/merge scripts. Raw SAM3 vine tracklets are not guaranteed biological vines.
- Background ID: label 0 in all inspected maps.
- Predicted object ID: `argmax(classifier(render_object))` during rendering.
- Learned object feature: continuous `num_objects`-dimensional feature stored per Gaussian as `obj_dc_*`; it is not itself an integer ID.

### 3.10 Points to state clearly in the report

- Masks supervise object features; they are not baked into the Gaussian IDs directly.
- Object features are continuous learned vectors.
- A classifier maps rendered object features to the configured label space.
- Label semantics depend entirely on the mask-preparation mode.
- Background is label 0 and is treated as a normal class in cross-entropy unless a separate analysis script ignores it.
- SAM3 tracklets are not necessarily physical vines.

Report-ready bullets for 4.3:

- Vineyard masks are produced from SAM3 video prompting on RGB frame sequences.
- SAM3 outputs semantic masks and, when available, tracked instance masks using SAM3 track IDs.
- The prepared training scene aligns masks only for RGB images registered in COLMAP.
- `object_mask/` is the folder actually consumed by `train.py`; other mask folders are evidence or intermediate products.
- Views without masks still train photometrically and do not contribute object classification loss.
- Instance labels may mean raw SAM3 tracklets, compact vine-only tracklets, weak connected components, or hierarchical vine/part labels, depending on the scene.
- The Gaussian model learns continuous object features; predicted IDs are produced by a trained `1x1` classifier.

Manual verification needed:

- For each final experiment, check `metadata/instance_tracking_report.json` and `metadata/instance_label_map.json` before describing label semantics.
- For `vinyes_sam3_200`, masks appear to be weak connected-component tracks, not true SAM3 track IDs.
- For `vinyes_sam3_vineid_200`, compact SAM3 tracker labels are present and vine tracklets are kept individually.

---

## 4. Details for Section 4.4: Channel metadata and implementation details

### 4.1 Output channels

Vineyard channel map in current code:

| Index | Meaning | Source in code |
|---:|---|---|
| 0 | R | `BAND_CHANNELS`, `_infer_active_channels` |
| 1 | G | `BAND_CHANNELS`, `_infer_active_channels` |
| 2 | B | `BAND_CHANNELS`, `_infer_active_channels` |
| 3 | b470, nominal 470 nm by name | `BAND_CHANNELS`, `_infer_active_channels` |
| 4 | b505, nominal 505 nm by name | `BAND_CHANNELS`, `_infer_active_channels` |
| 5 | b525, nominal 525 nm by name | `BAND_CHANNELS`, `_infer_active_channels` |
| 6 | b590, nominal 590 nm by name | `BAND_CHANNELS`, `_infer_active_channels` |
| 7 | b635, nominal 635 nm by name | `BAND_CHANNELS`, `_infer_active_channels` |
| 8 | b660, nominal 660 nm by name | `BAND_CHANNELS`, `_infer_active_channels` |
| 9 | b850, nominal 850 nm by name | `BAND_CHANNELS`, `_infer_active_channels` |

Important ambiguity:

- The code does not store a separate numeric wavelength metadata field. Wavelengths are encoded only in band names like `b470`.

Other datasets:

- MMS uses 9 channels in config, but channel wavelengths/names are not defined in the inspected prep script beyond channel indices.
- Spec-NeRF uses 20 channels in config; the prep script stacks sorted TIFFs but does not write wavelength metadata.
- X-NeRF uses however many channels are in `ms_imgs.npy`, but shown configs use 10.

### 4.2 How active channels are known

Priority order in [scene/dataset_readers.py](../scene/dataset_readers.py), `_load_active_channel_map`:

1. `metadata/active_channels.json`
2. `frame_info.json`
3. `band_info.json`

If no metadata entry exists:

- `_infer_active_channels(image_name)` uses filename prefix.

If neither metadata nor prefix inference works:

- `active_channels=None`; then [utils/camera_utils.py](../utils/camera_utils.py), `_expand_active_channels`, defaults to the loaded tensor channels or `range(min(image_tensor.shape[0], num_channels))`.

### 4.3 Active-channel metadata format

Accepted formats:

```json
{
  "b470_00001": [3],
  "rgb_00001": [0, 1, 2],
  "rgb_00002.png": {"channels": [0, 1, 2]}
}
```

Details:

- Keys are normalized to the filename stem.
- Channels are zero-indexed.
- Values can be either a list or an object containing `"channels"`.

Sampled metadata:

- `vinyes_sam3_vineid_200/metadata/active_channels.json`: 1576 entries, e.g. `b470_00001: [3]`.
- `vinyes_sam3_vineid_200/frame_info.json`: entries include `band_key`, `band_name`, `channels`, `source_frame`, and `has_object_mask`.
- `vinyes_pose800/frame_info.json`: entries include `matched_bands`, but the loader only reads `channels`.

### 4.4 Partial RGB and partial multispectral supervision

Implemented in:

- [utils/camera_utils.py](../utils/camera_utils.py), `_expand_active_channels`.
- [train.py](../train.py), `active_channel_loss_tensors`.
- [train.py](../train.py), `training`, `single_channel_mode` branch.

Behavior:

- Loaded image/tensor data is expanded to shape `[num_channels, H, W]`.
- Channels not active for a view are zero in the target tensor.
- Photometric loss is computed only on active channels via `active_channel_loss_tensors`.
- If `single_channel_mode=false`, all active channels contribute together.
- If `single_channel_mode=true`, one active channel is randomly selected for that iteration.
- There is no dense per-pixel channel-validity mask; validity is per-image channel list.

### 4.5 Multispectral image loading

If `images_multispectral/<stem>.npy` exists:

- Loaded with `np.load`.
- Expected shape: `[H, W, C]`.
- Resized to the camera resolution with bilinear `scipy.ndimage.zoom`.
- Converted to torch `[C, H, W]`.
- Expanded into `num_channels` using active-channel indices.

If no `.npy` exists:

- PIL image is loaded and converted by `PILtoTorch`.
- First three channels are used.
- For a single-band active channel, the first loaded image channel is placed into the specified output channel.

Normalization:

- Loader assumes values are already suitable for clamp/loss.
- MMS/Spec-NeRF/X-NeRF prep scripts generate or assume `[0,1]`.
- Real vineyard per-band PNGs are loaded as image intensities through PIL conversion. No dataset-specific radiometric calibration was found in the inspected training path.

### 4.6 Main vineyard experiment configs

| Config | num_channels | num_classes | num_objects | use_color_embed | color_embed_dim | decoder hidden/layers | densify_until | max points | reg3d interval | reg3d max/sample |
|---|---:|---:|---:|---|---:|---|---:|---:|---:|---|
| `vinyes_sam3_200.json` | 10 | 348 | 16 | true | 32 | 128 / 3 | 6000 | 1000000 | 5 | 150000 / 500 |
| `vinyes_sam3_vineid_200.json` | 10 | 362 | 16 | true | 32 | 128 / 3 | 6000 | 1000000 | 5 | 150000 / 500 |
| `vinyes_20260509_sam3.json` | 10 | 1758 | 16 | true | 32 | 128 / 3 | 6000 | 1000000 | 5 | 150000 / 500 |
| `vinyes_fulles_1.json` | 10 | 1128 | 16 | true | 32 | 128 / 3 | 6000 | 1000000 | 5 | 150000 / 500 |

Other relevant configs:

- `train_mms.json`: 9 channels, 256 classes, 16 object features.
- `specnerf_baseline.json`: 20 channels, 256 classes, 16 object features.
- `phase3/xnerf_baseline.json`: 10 channels, 256 classes, 16 object features.
- `phase4/xnerf_baseline.json`: 10 channels, single-channel mode true.

Training command evidence:

- [script/train_vinyes_sam3_200.sh](../script/train_vinyes_sam3_200.sh) uses `--iterations 40000`, `--resolution 4`, `--eval`, and config `vinyes_sam3_200.json`.
- [script/train_mms.sh](../script/train_mms.sh) uses `--eval`, `-r 2`, save/test iterations `1000 7000 15000 30000 40000`, and forwards extra args.

### 4.7 Reproducibility details

Training:

- Default iterations from [arguments/__init__.py](../arguments/__init__.py), `OptimizationParams`, are 30000, but scripts/configs can override.
- Default save iterations in `train.py` include 1000, 7000, 30000, 60000, and `args.iterations` is appended.
- Default test iterations in `train.py`: 1000, 7000, 30000, 30000.
- `script/train_vinyes_sam3_200.sh` overrides both for 40000.

GPU/CUDA:

- `Camera` stores transforms and images on CUDA by default.
- Rasterizer output dimensions are compile-time constants in `submodules/diff-gaussian-rasterization/cuda_rasterizer/config.h`:
  - `NUM_CHANNELS 10`
  - `NUM_OBJECTS 16`
- `script/train_mms.sh` notes that the rasterizer must be rebuilt with `NUM_CHANNELS=9` for MMS.

Outputs:

- `cfg_args`: written by `train.py`, `prepare_output_and_logger`.
- Label metadata copied from `source_path/metadata`: `class_map.json`, `class_colors.json`, `instance_label_map.json`, `instance_tracking_report.json`, `mask_alignment_report.json`, `active_channels.json`, `registered_images_summary.json`, `hierarchical_label_schema.json`, `hierarchical_label_report.json`.
- `input.ply`: copied from initial scene PLY by `Scene.__init__`.
- `cameras.json`: written by `Scene.__init__`.
- `point_cloud/iteration_<N>/point_cloud.ply`: written by `Scene.save`.
- `point_cloud/iteration_<N>/classifier.pth`: saved by `train.py`.
- `point_cloud/iteration_<N>/color_decoder.pth`: saved only when `use_color_embed=true`.
- `chkpnt<N>.pth`: optional checkpoints contain `gaussians.capture()` and iteration only.

What is not saved in checkpoints:

- The classifier optimizer/state and classifier weights are not included in `chkpnt<N>.pth`.
- The color decoder state is not included in `chkpnt<N>.pth`.
- They are saved separately only at save iterations.

### 4.8 CUDA/Python config inconsistency risk

Critical:

- Python configs can set `num_channels` and `num_objects`, but the rasterizer allocates outputs using compile-time `NUM_CHANNELS` and `NUM_OBJECTS`.
- Current `config.h` is fixed at 10 channels and 16 objects.
- Running a 9-channel or 20-channel config without rebuilding or changing `config.h` is inconsistent.
- Running `num_objects` other than 16 without rebuilding is inconsistent because `out_objects` is fixed to `NUM_OBJECTS`.

### 4.9 Appendix material

Move to appendix rather than main text:

- Full command lines for every run.
- Complete folder trees.
- Full JSON examples for `active_channels.json`, `instance_label_map.json`, and `frame_info.json`.
- Full COLMAP diagnostics JSON and trajectory CSV.
- Exhaustive config values for non-main ablations.
- Long lists of SAM3 prompts, unless the report needs mask-generation reproducibility in detail.

Report-ready paragraph for 4.4:

The implementation represents each training view as a full `num_channels` tensor, but supervises only the channels listed as active for that image. For vineyard scenes, RGB occupies channels 0-2 and narrowband images occupy channels 3-9 according to their filename band. Active-channel metadata is read first from `metadata/active_channels.json`, then from `frame_info.json`, then from `band_info.json`; if unavailable, the loader falls back to filename prefixes. The color model uses a learned color embedding and decoder in the main vineyard configs, while object identity is represented by a 16-dimensional Gaussian object feature and a `1x1` classifier. Reproducibility requires matching the Python `num_channels` and `num_objects` settings to the compiled CUDA rasterizer constants.

---

## 5. Suggested figures/tables for Section 4

| Figure/table | What it should show | Likely source files | Main text or appendix |
|---|---|---|---|
| Pipeline diagram | Raw videos/images to training scene to `train.py` | This evidence note; prep scripts | Main text |
| Dataset folder structure | Required and optional folders with join key `<stem>` | `vinyes_sam3_vineid_200/`, `scene/dataset_readers.py` | Main text compact, appendix detailed |
| Channel mapping table | Channel indices 0-9 and RGB/band labels | `BAND_CHANNELS` in prep scripts, `_infer_active_channels` | Main text |
| Mask/object-ID table | `object_mask`, `semantic_mask`, `sam3_instance_mask`, metadata meanings | `prepare_vinyes_sam3_200.py`, sampled metadata | Main text |
| COLMAP pose figure | RGB trajectory top-view and weak frames if available | `trajectory_topview.png` under COLMAP variant folders, `colmap_quality.json` | Main text if it supports a point; appendix otherwise |
| Example RGB and spectral frames | Same stem/prefix examples for RGB and bands | `vineyard_posematch/*/images/`, `images_rgb/`, `frames_raw/` | Main text |
| Example masks | RGB frame, semantic mask, instance/object mask overlay | `metadata/mask_contact_sheet.jpg`, `semantic_mask/`, `object_mask/` | Main text |
| Active-channel metadata example | Short JSON snippet showing `rgb` and one band | `metadata/active_channels.json`, `frame_info.json`, `band_info.json` | Appendix |
| CUDA/config consistency table | Python config vs `NUM_CHANNELS`, `NUM_OBJECTS` | `config/*.json`, `config.h` | Appendix or reproducibility subsection |

---

## 6. Implementation evidence table

| Topic | What happens | File/script | Function/class | Key variables/files |
|---|---|---|---|---|
| Dataset type selection | COLMAP scene if `source_path/sparse` exists, Blender if `transforms_train.json` exists | `scene/__init__.py` | `Scene.__init__` | `args.source_path`, `sparse/` |
| COLMAP loading | Reads binary cameras/images, falls back to text | `scene/dataset_readers.py` | `readColmapSceneInfo` | `cameras.bin/txt`, `images.bin/txt` |
| Camera pose conversion | Converts qvec to R, uses tvec, computes FoV | `scene/dataset_readers.py` | `readColmapCameras` | `qvec2rotmat`, `focal2fov` |
| Camera object creation | Stores image, mask, transforms, active channels on CUDA | `utils/camera_utils.py`, `scene/cameras.py` | `loadCam`, `Camera` | `original_image`, `objects`, `active_channels` |
| Point cloud initialization | Uses `points3D.ply` or converts COLMAP points; optionally random init | `scene/dataset_readers.py`, `scene/gaussian_model.py` | `readColmapSceneInfo`, `create_from_pcd` | `points3D.ply`, `points3D.bin/txt` |
| Mask loading | Opens `object_mask/<stem>.png` if present | `scene/dataset_readers.py` | `readColmapCameras` | `object_path`, `objects` |
| Mask ID conversion | 2D labels pass through; RGB masks are packed/remapped | `utils/camera_utils.py` | `_convert_object_mask_to_indices` | `object_id_mapping`, `num_classes` |
| Missing mask handling | No object CE loss for views without masks | `train.py` | `training` | `viewpoint_cam.objects is None` |
| Active channel loading | Metadata priority: `metadata/active_channels.json`, `frame_info.json`, `band_info.json` | `scene/dataset_readers.py` | `_load_active_channel_map` | `channels` |
| Active channel inference | Prefix maps `rgb`, `b470`, ..., `b850` | `scene/dataset_readers.py` | `_infer_active_channels` | filename stem |
| Active channel expansion | Loaded images/tensors expanded to `[num_channels,H,W]` | `utils/camera_utils.py` | `_expand_active_channels` | `num_channels`, `active_channels` |
| Active channel loss | Selects active channels before L1/SSIM | `train.py` | `active_channel_loss_tensors` | `rendered`, `target` |
| Single-channel training | Randomly selects one active channel per iteration | `train.py` | `training` | `single_channel_mode` |
| Multispectral tensor loading | Loads `.npy` as `[H,W,C]`, resizes, expands | `utils/camera_utils.py` | `loadCam` | `images_multispectral/<stem>.npy` |
| Config parsing | JSON config overrides selected args | `train.py` | main block | `num_channels`, `num_classes`, `num_objects` |
| Training script | Builds Gaussian model, classifier, optional color decoder, losses | `train.py` | `training` | `classifier`, `ColorDecoder` |
| Output saving | Saves PLY, classifier, optional color decoder | `train.py`, `scene/__init__.py` | `training`, `Scene.save` | `point_cloud/iteration_*` |
| Metadata copying | Copies selected metadata files to output model root | `train.py` | `copy_label_metadata` | `class_map.json`, `active_channels.json`, etc. |
| Render outputs | Saves RGB previews, `.npy` full renders, object predictions, frame index | `render.py` | `render_set` | `frames_index.json`, `renders/*.npy` |
| Metrics active channels | Uses `frames_index.json` active channels to evaluate outputs | `metrics.py` | `read_frames_index`, `select_active_channels` | `active_channels` |
| Vineyard frame extraction | Extracts RGB/band frames and stages `images/`, `images_rgb/` | `prepare_vineyard_video_colmap.py` | `extract_video_frames` | `frames_raw`, `images`, `images_rgb` |
| Vineyard COLMAP | RGB variants, diagnostics, band registration | `prepare_vineyard_video_colmap.py` | `run_rgb_colmap_variant`, `run_registration` | `colmap_rgb_*`, `sparse_registered` |
| Vineyard channel metadata | Writes `band_info.json`, `frame_info.json`, summary | `prepare_vineyard_video_colmap.py` | `write_scene_metadata` | `partial_channels_summary.json` |
| SAM3 masks | Runs SAM3 video predictor and writes semantic/instance masks | `sam3_vine_video.py` | `run_class_predictions`, `build_semantic_outputs` | `semantic_instance_masks`, `instance_label_map.json` |
| SAM3 scene assembly | Aligns masks to registered images and writes training masks | `prepare_vinyes_sam3_200.py` | `main` | `object_mask`, `semantic_mask`, `sam3_instance_mask` |
| Hierarchical labels | Composes flat object/part labels with metadata hierarchy | `compose_hierarchical_vineyard_labels.py` | `main` | `hierarchical_label_schema.json` |
| Path filtering | Drops band views far from RGB reference trajectory | `filter_colmap_scene_by_path.py` | `main` | `colmap_path_filter/filter_report.json` |

---

## 7. Ambiguities and manual checks

1. What exactly `object_mask/` means in each final run
   - Where ambiguity appears: `object_mask/` is reused for DEVA, semantic labels, SAM3 instance labels, compact vine IDs, weak components, and hierarchical labels.
   - Why it matters: Section 4.3 must not claim "instance IDs" if the run used weak components or semantic masks.
   - Manual check: inspect each run's `metadata/instance_tracking_report.json`, `metadata/instance_label_map.json`, and config `label_mode`.

2. Whether final vineyard COLMAP used RGB-only, RGB-register, direct all-band, or filtered registration
   - Where ambiguity appears: multiple scripts and prepared scenes exist: `vinyes_partial200`, `vinyes_pose800`, `vinyes_fulles_rgbrobust`, `vinyes_sam3_*`.
   - Why it matters: Section 4.2 should describe the actual final scene.
   - Manual check: inspect `colmap_shared/register_config.json`, `colmap_shared/registration_summary.json`, `colmap_rgb_variants_summary.json`, and `colmap_quality.json` for the exact dataset used.

3. Whether masks exist only for RGB views
   - Where ambiguity appears: sampled SAM3 scenes have 200 masks and 1576 images, but other scenes may differ.
   - Why it matters: object supervision coverage changes loss interpretation.
   - Manual check: count `object_mask/*.png` and compare to registered image names in `sparse/0/images.txt`.

4. Whether `.npy` multispectral tensors are normalized to `[0,1]`
   - Where ambiguity appears: loader assumes but does not enforce; MMS/Spec-NeRF scripts normalize, X-NeRF assumes input is normalized.
   - Why it matters: photometric loss scale and color decoder behavior.
   - Manual check: sample min/max of `images_multispectral/*.npy` for the final dataset.

5. Whether the CUDA rasterizer was compiled with matching constants
   - Where ambiguity appears: `config.h` says `NUM_CHANNELS=10`, `NUM_OBJECTS=16`; configs include 9 and 20 channel runs.
   - Why it matters: mismatched output tensors break or silently invalidate experiments.
   - Manual check: before each experiment, verify `submodules/diff-gaussian-rasterization/cuda_rasterizer/config.h` and rebuild if changing channel/object dimensions.

6. Whether `vinyes_pose800` is part of the final report setup
   - Where ambiguity appears: sampled `vinyes_pose800` has `input/`, `object_mask/`, `sparse/0`, and rich `matched_bands` metadata, but the producing script is not in the inspected repo.
   - Why it matters: it may use a different pose/band association strategy than `prepare_vineyard_video_colmap.py`.
   - Manual check: locate the exact script/run log that created `vinyes_pose800`, or treat it as an older/manual dataset.

7. Whether train/test split is consistent across bands
   - Where ambiguity appears: default split sorts all image names together and takes every 8th image; for band-prefixed files this may split by band blocks rather than matched frame groups.
   - Why it matters: evaluation may not hold out corresponding RGB/band frames consistently.
   - Manual check: inspect generated train/test camera names from `Scene` for final runs, or use `images_train/` if a grouped split is required.

8. Whether root metadata is copied to the model output
   - Where ambiguity appears: `train.py` copies selected files only from `source_path/metadata`, not root `frame_info.json` or `band_info.json`.
   - Why it matters: reproducibility metadata may be split between dataset and model output.
   - Manual check: inspect model output root after training for copied metadata.

9. Whether RGB masks are indexed or color-coded
   - Where ambiguity appears: loader supports both; prep scripts write indexed masks, DEVA may output indexed or colored depending on tool output.
   - Why it matters: RGB remapping can compact labels and drop IDs beyond `num_classes`.
   - Manual check: inspect a sample mask shape/dtype/unique values.

10. Whether `num_rgb_registered` is accurate for scenes using prefix `rgb` vs `rgbp`
   - Where ambiguity appears: `prepare_vinyes_sam3_200.py` writes `num_rgb_registered = summary.get("rgbp", 0)`, but some scenes use `rgb` names.
   - Why it matters: metadata summary may under-report RGB registered counts.
   - Manual check: compare `registered_per_band` to the `num_rgb_registered` field.

---

## 8. Final report outline

### 4.1 Common processing pipeline

- Convert each dataset into a COLMAP-style training scene.
- Use image stems as the join key for pixels, poses, masks, and channel metadata.
- Describe required folders: `images/`, `sparse/0/`, optional `object_mask/`, `images_multispectral/`, `metadata/`.
- Explain RGB/pseudo-RGB images versus full `.npy` multispectral tensors.
- Explain active-channel supervision for partial-band vineyard images.
- State train/test split behavior under `--eval` and `--train_split`.
- State that all experiments share `train.py`, `Scene`, `Camera`, Gaussian initialization, and losses.

### 4.2 Camera pose estimation

- State pose source per dataset: COLMAP, existing converted poses, or RGB-reference vineyard COLMAP.
- List required COLMAP files and binary/text fallback.
- Explain pose loading into `CameraInfo` and `Camera`.
- Explain Gaussian initialization from `points3D.ply` or converted COLMAP points.
- For vineyard, describe RGB COLMAP variants and band registration with `image_registrator`.
- Mention diagnostics: registered fraction, reprojection error, folded trajectory, weak near-turn images.
- State limitations: pose quality dependence, repeated structures, band registration uncertainty, static scene assumption.

### 4.3 Mask generation and object IDs

- State mask source: DEVA for generic data, SAM3 video prompting for vineyard data.
- Explain SAM3 semantic masks and SAM3 tracked instance masks.
- State `object_mask/` is the training folder.
- Explain missing masks produce no object loss for that view.
- Distinguish semantic class IDs, SAM3 tracklet IDs, compact vine IDs, hierarchical labels, and background.
- Explain Gaussian object features are continuous and classifier-predicted labels are derived from them.
- Note that SAM3 tracklets are not guaranteed physical vines.

### 4.4 Channel metadata and implementation details

- Define channel indices for vineyard: RGB 0-2, narrowbands 3-9.
- Explain active-channel metadata priority and zero-indexed format.
- Explain `_expand_active_channels` and active-channel photometric loss.
- Explain `single_channel_mode`.
- Describe `.npy` multispectral loading, resizing, and normalization assumptions.
- Summarize main vineyard hyperparameters: 10 channels, 16 object features, color embedding, decoder, classes per config.
- State saved outputs: `cfg_args`, `input.ply`, `cameras.json`, `point_cloud.ply`, `classifier.pth`, `color_decoder.pth`.
- State CUDA compile-time constants must match `num_channels` and `num_objects`.
