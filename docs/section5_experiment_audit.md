# Section 5 Experiment Audit

This is a factual audit for Section 5, "Experiments and Evaluation", of the TFG report:

**Instance-Aware Multispectral Representation of Vineyards with 3D Gaussian Splatting**

Scope: repository, scripts, configs, logs, output folders, rendered results, metric files, and dataset-preparation scripts visible under `/home/msiau/workspace`, especially `/home/msiau/workspace/jcalm` and linked data/output folders.

Rules followed:

- Verified claims cite the file, script, log, or output folder where they were found.
- Inferred claims are explicitly labeled as inferred.
- Missing, deleted, ambiguous, or unrecoverable details are marked as `UNKNOWN / NEEDS JOEL CONFIRMATION`.
- This document does not write the final report section.

## 1. Executive summary

### Main verified facts

- The current training code predicts `num_channels` output channels and computes photometric loss only on `active_channels` when present. This is implemented in `active_channel_loss_tensors()` and used by the training loop in `jcalm/train.py`.
- Active-channel metadata can be loaded from `metadata/active_channels.json`, `frame_info.json`, or `band_info.json`, and can also be inferred from vineyard filename prefixes such as `rgb`, `b470`, `b505`, etc. This is implemented in `jcalm/scene/dataset_readers.py`.
- The real vineyard channel order is implemented in code as: `0 R`, `1 G`, `2 B`, `3 b470`, `4 b505`, `5 b525`, `6 b590`, `7 b635`, `8 b660`, `9 b850`. Evidence: `BAND_CHANNELS` in `jcalm/prepare_vineyard_video_colmap.py` and `jcalm/prepare_vinyes_sam3_200.py`.
- The renderer supports either spherical-harmonic color or learned color embeddings decoded by an MLP. Evidence: `jcalm/gaussian_renderer/__init__.py`, `jcalm/scene/gaussian_model.py`, and `jcalm/utils/color_decoder.py`.
- Current tracked metric code implements reconstruction metrics SSIM, PSNR, LPIPS, and L1. It also supports per-channel metrics when `single_channel_mode` is true in `cfg_args`. Evidence: `jcalm/metrics.py`.
- Some existing vineyard `results.json` files contain RMSE and Spectral Angle Mapper values, but the current tracked `jcalm/metrics.py` does not implement those metrics. The exact script/version that produced those RMSE/SAM values is `UNKNOWN / NEEDS JOEL CONFIRMATION`.
- Basement data exists at `/home/msiau/data/tmp/jcalm/data/basement` with 50 views, 9 distorted channel folders, 9 undistorted channel folders, 50 masks, and COLMAP sparse reconstructions. Evidence: filesystem counts and COLMAP binary files under that dataset.
- I did not find completed Basement output folders for the listed k/random/round-robin/Bernoulli variants under the visible output roots. Status of those experiments is `UNKNOWN / NEEDS JOEL CONFIRMATION`.
- I did not find the Bear dataset or Bear output folders in the inspected workspace. Bear is referenced by docs/configs, but actual runs and metrics are `UNKNOWN / NEEDS JOEL CONFIRMATION`.
- Several real vineyard models and metrics exist, including `vinyes_20260509_sam3`, `vinyes_20260509_pinhole_sam3_full`, `vinyes_sam3_vineid_200`, and older `vinyes2`, `vinyes3`, `vinyes_partial*` outputs. Evidence: output folders and `results.json` files under `/home/msiau/data/tmp/jcalm/output` and `/home/msiau/workspace/vineyard_posematch/output`.
- A representation-analysis pipeline exists for PCA, LDA, t-SNE, UMAP, and classifier separability over SAM labels, predicted pixel labels, and predicted Gaussian labels. Evidence: `jcalm/analysis/multispectral_separability.py` and outputs under `jcalm/outputs/multispectral_separability*`.

### Main uncertainties

- Which Bear experiments actually completed, and where their outputs/metrics are stored: `UNKNOWN / NEEDS JOEL CONFIRMATION`.
- Whether Basement full and partial-supervision experiments completed, and which output folders are final: `UNKNOWN / NEEDS JOEL CONFIRMATION`.
- Whether the professor's k notation is implemented in the current code or only reflected in older config/result metadata: current code evidence is insufficient; `UNKNOWN / NEEDS JOEL CONFIRMATION`.
- Which vineyard dataset/model should be treated as final for the report: `UNKNOWN / NEEDS JOEL CONFIRMATION`.
- Which checkpoint should be reported when multiple checkpoints exist, for example 30000, 40000, or 60000 iterations: `UNKNOWN / NEEDS JOEL CONFIRMATION`.
- Whether SAM3 masks are semantic, instance, vine-ID, or mixed for each final experiment: partially verified per dataset metadata, but final intended interpretation needs confirmation.
- Whether old vineyard metrics containing RMSE/SAM are trustworthy and produced by the intended evaluation code: `UNKNOWN / NEEDS JOEL CONFIRMATION`.

## 2. Relevant code/scripts

### Training

| Path | Purpose | Key arguments/config | Inputs | Outputs |
|---|---|---|---|---|
| `jcalm/train.py` | Main 3DGS/Gaussian Grouping training loop with active-channel photometric loss, object classifier loss, and optional 3D regularization. | `--source_path/-s`, `--model_path/-m`, `--config_file`, `--eval`, `--resolution/-r`, `--iterations`, `--test_iterations`, `--save_iterations`, `--checkpoint_iterations`, model params from `jcalm/arguments/__init__.py`. | COLMAP scene folders, image folders, optional masks, optional active-channel metadata, JSON config. | Model folder with `point_cloud/iteration_*`, `cfg_args`, `classifier.pth`, optional `color_decoder.pth`, copied metadata. |
| `jcalm/script/train.sh` | Older generic wrapper for one dataset. | Dataset name and scale; uses `config/gaussian_dataset/train.json`; runs train then render. | `data/$dataset`. | `output/$dataset`. |
| `jcalm/script/train_mms.sh` | Wrapper for multi-modal-studio/MMS scenes. | Scene name, resolution, optional config; default `config/gaussian_dataset/train_mms_msiau.json`; uses `--eval`, iterations up to 40000. | `../../data/tmp/jcalm/data/multi-modal-studio/$SCENE`. | `../../data/tmp/jcalm/output/mms_$SCENE`. |
| `jcalm/script/train_vinyes_sam3_200.sh` | Wrapper for SAM3 vineyard dataset training. | Uses `config/gaussian_dataset/vinyes_sam3_200.json`, `--eval`, `-r 4`, iterations 40000. | `../vineyard_posematch/vinyes_sam3_200`. | `output/vinyes_sam3_200`. |
| `jcalm/run_overnight_experiments.sh` | Batch script for MMS/Birdhouse experiment configs. | Runs `exp_a_baseline`, `exp_b_high_reg`, `exp_c_no_single_channel`, `exp_d_sh_baseline` at 40000 iterations, then render and metrics. | MMS/Birdhouse dataset and config files. | Intended output folders under `output/`, but corresponding visible output folders were not found. Status: `UNKNOWN / NEEDS JOEL CONFIRMATION`. |

### Rendering

| Path | Purpose | Key arguments | Inputs | Outputs |
|---|---|---|---|---|
| `jcalm/render.py` | Render train/test views, full output channels, object features, predicted object masks, and GT masks when available. | `--iteration`, `--skip_train`, `--skip_test`, `--quiet`, `--only_prefix`, `--max_train_views`, `--max_test_views`, plus model/source args. | Trained model folder, `classifier.pth`, optional `color_decoder.pth`, scene data. | `train/ours_*/renders`, `gt`, `objects_feature16`, `gt_objects_color`, `objects_pred`, `frames_index.json`, `channel_frames_index.json`, per-channel folders such as `channel_B0`. |

Verified details:

- For `num_channels > 3`, the renderer saves RGB visualization PNGs using channels `[0,1,2]` and saves full-channel `.npy` arrays for both render and GT. Evidence: `jcalm/render.py`.
- Per-channel comparison images are written only for active channels of a frame. Evidence: `jcalm/render.py`.

### Metric computation

| Path | Purpose | Key arguments | Inputs | Outputs |
|---|---|---|---|---|
| `jcalm/metrics.py` | Current reconstruction metric script. Computes SSIM, PSNR, LPIPS, and L1. Supports active-channel selection from `frames_index.json`. | Model paths, `--iteration`. | Rendered test folder, `renders`, `gt`, `frames_index.json`, `cfg_args`. | `results.json` in model folder. |

Verified details:

- Current tracked metrics use the `test` split render folder. Evidence: `jcalm/metrics.py`.
- Current tracked metrics do not implement RMSE, Spectral Angle Mapper, vegetation-index metrics, IoU, or mIoU. Evidence: `jcalm/metrics.py`.
- Existing old vineyard result files contain RMSE/SAM values, but their metric implementation is not present in current tracked `jcalm/metrics.py`. Evidence: `results.json` files under `/home/msiau/data/tmp/jcalm/output/vinyes2`, `/home/msiau/data/tmp/jcalm/output/vinyes3`, and `/home/msiau/workspace/vineyard_posematch/output/vinyes_partial*`.

### Dataset preparation

| Path | Purpose | Key arguments | Inputs | Outputs |
|---|---|---|---|---|
| `jcalm/prepare_xnerf_dataset.py` | Convert X-NeRF-style raw multispectral arrays and poses into COLMAP-format scene. | `--dataset_dir`, `--output_dir`, `--rgb_channels`, `--image_ext`, `--single_camera`. | `ms_imgs.npy`, `rgb_poses.npy`. | `images/`, `images_multispectral/`, `sparse/0/{cameras.txt,images.txt,points3D.txt}`. |
| `jcalm/prepare_specnerf_dataset.py` | Convert Spec-NeRF TIFF data into COLMAP-format multispectral scene. | `--scene_dir`, `--output_dir`, `--rgb_channels`, `--image_ext`, `--undistort`. | Spec-NeRF pose/image folders with per-pose TIFF channels. | `images/`, `images_multispectral/`, `sparse/0`. |
| `jcalm/prepare_mms_dataset.py` | Convert Multi-Modal-Studio scene metadata and multispectral arrays into COLMAP-format scene. | `--dataset_root`, `--scene`, `--output_dir`, `--rgb_channels`, `--image_ext`. | MMS `meta_data.json`, `modalities/multispectral/*.npy`. | `images/`, `images_multispectral/`, `sparse/0`. |
| `jcalm/prepare_vineyard_video_colmap.py` | Extract RGB/MS frames from videos, run COLMAP, register bands, finalize metadata, and diagnostics. | `--video_dir`, `--output_dir`, `--stage`, `--registration_mode`, `--num_frames`, `--camera_model`, `--rgb_matcher`, `--cross_band_window`, etc. | Vineyard videos, one RGB plus narrowband videos. | `frames_raw/`, `images/`, `images_rgb/`, `colmap_*`, `sparse/0`, `metadata`, `band_info.json`, `frame_info.json`, diagnostics JSONs. |
| `jcalm/prepare_vinyes_sam3_200.py` | Prepare SAM3-masked vineyard dataset from registered vineyard images. | `--source_scene_dir`, `--sam3_dir`, `--output_dir`, `--label_mode`, `--object_mask_source`, `--copy_images`. | Registered vineyard scene, SAM3 masks/tracks. | `images`, `images_rgb`, `object_mask`, `semantic_mask`, optional `sam3_instance_mask`, `metadata/active_channels.json`, class maps, reports. |
| `jcalm/compose_hierarchical_vineyard_labels.py` | Compose hierarchical vineyard labels/class metadata for training. | Config template includes `num_channels=10`, `num_objects=16`, color embedding settings. | Vineyard masks and metadata. | Composite label metadata/config outputs. |

### Active-channel metadata generation/loading

Verified code paths:

- Loading active-channel metadata: `jcalm/scene/dataset_readers.py`.
- Active-channel inference from filenames: `jcalm/scene/dataset_readers.py`.
- RGB/single/multispectral tensor expansion into a full-channel tensor: `jcalm/utils/camera_utils.py`.
- Metadata copying into model output folders: `copy_label_metadata()` in `jcalm/train.py`.
- Vineyard metadata generation: `jcalm/prepare_vineyard_video_colmap.py` and `jcalm/prepare_vinyes_sam3_200.py`.

### COLMAP conversion/registration and feature extraction

Verified code paths:

- `jcalm/prepare_vineyard_video_colmap.py` performs frame extraction, feature extraction, matching, mapping, target-band registration, finalization, and diagnostics.
- Matching/feature extraction options include SIFT GPU use, `max_image_size`, `max_num_features`, affine shape/domain-size pooling, sequential matching, exhaustive matching, and cross-band matching windows. Evidence: argument definitions and command builders in `jcalm/prepare_vineyard_video_colmap.py`.
- Logs show repeated COLMAP problems:
  - `jcalm/logs/vinyes_20260418_rgbdense_turn.log`: sequential matcher killed with SIGKILL.
  - `jcalm/logs/vinyes_20260418_rgbdense_turn_map_finalize.log`: 651/651 RGB frames registered but quality flagged bad.
  - `/home/msiau/data/tmp/jcalm/data/vinyes_20260418_rgbdense_turn/colmap_rgb_seqloop_pinhole/colmap_quality.json`: `quality_ok=false` with reason `folded_trajectory_line_angle=78.6`.
  - `vineyard_posematch/logs/vinyes_partial200_fast.log`: RGB-first registration failed for 37 target frames.

### Mask alignment or mask generation

| Path | Purpose | Evidence |
|---|---|---|
| `jcalm/prepare_vinyes_sam3_200.py` | Align SAM3 semantic/index masks to RGB frames; can use SAM3 instance masks or weak connected components. Writes object masks and reports. | Script code and generated vineyard dataset folders. |
| `jcalm/compose_hierarchical_vineyard_labels.py` | Compose hierarchical labels and metadata for vineyard masks. | Script/config template. |

Generated mask-related folders verified in vineyard datasets:

- `object_mask`
- `semantic_mask`
- `sam3_instance_mask` in some datasets
- `metadata/class_map.json`, `metadata/class_map_semantic.json`, and reports in SAM3-prepared datasets

Which masks should be called semantic, instance, vine-ID, or mixed in the final report is partially verified per folder/config but still `UNKNOWN / NEEDS JOEL CONFIRMATION`.

### PCA/LDA/t-SNE/UMAP analysis and classifier separability

| Path | Purpose | Inputs | Outputs |
|---|---|---|---|
| `jcalm/analysis/multispectral_separability.py` | Analyze RGB/MS/RGB+MS separability using PCA, optional LDA/t-SNE/UMAP, and classifier probes over SAM labels, predicted pixel labels, or predicted Gaussian labels. | Trained model, source scene, rendered/predicted labels or Gaussian features. | CSV metrics, confusion matrices, PCA/LDA/t-SNE/UMAP plots under `jcalm/outputs/multispectral_separability*`. |

Verified outputs:

- `jcalm/outputs/multispectral_separability/sam`
- `jcalm/outputs/multispectral_separability/predicted_pixel`
- `jcalm/outputs/multispectral_separability/predicted_gaussian`
- `jcalm/outputs/multispectral_separability_semantic_test_with_background/predicted_pixel`
- `jcalm/outputs/multispectral_separability_semantic_test_with_background/predicted_gaussian`

### Where core model behavior is implemented

| Behavior | Location | Verified detail |
|---|---|---|
| Color embedding | `jcalm/scene/gaussian_model.py` | Gaussian model optionally stores learned `_color_embedding` with dimension `color_embed_dim`. |
| Color decoder | `jcalm/utils/color_decoder.py` | MLP maps color embedding to `num_channels`, ends with Sigmoid. |
| Number of output channels | `jcalm/arguments/__init__.py`, configs, `jcalm/gaussian_renderer/__init__.py`, CUDA rasterizer config | Default `num_channels=3`; vineyard configs use 10; MMS/Basement-like configs use 9; Spec-NeRF configs use 20. |
| Active-channel loading/inference | `jcalm/scene/dataset_readers.py` | Reads metadata or infers from filename prefixes. |
| Active-channel tensor expansion | `jcalm/utils/camera_utils.py` | Embeds RGB/single/multispectral GT into full output-channel tensor. |
| Partial-channel loss | `jcalm/train.py` | `active_channel_loss_tensors()` selects active channels before L1/SSIM. |
| Object feature rendering | `jcalm/gaussian_renderer/__init__.py`, `jcalm/render.py` | Renders `render_object`; saves PCA visualization and predicted object masks. |
| Object classifier | `jcalm/train.py` | 1x1 Conv2d classifier maps object features to `num_classes`; saved as `classifier.pth`. |
| 2D mask loss | `jcalm/train.py` | Cross-entropy between classifier logits and `viewpoint_cam.objects` when object masks exist. |
| 3D object regularization | `jcalm/train.py`, `jcalm/utils/loss_utils.py` | Uses `loss_cls_3d` at `reg3d_interval` over Gaussian object logits. |
| Rendering all channels | `jcalm/render.py` | Saves full `.npy` renders/GT for multi-channel outputs. |
| Per-band metrics | `jcalm/metrics.py` | Current implementation only per-channel when `single_channel_mode` is true. Older RMSE/SAM per-band script is not found. |

### Hard-coded CUDA constants

Verified in `jcalm/submodules/diff-gaussian-rasterization/cuda_rasterizer/config.h`:

```c
#define NUM_CHANNELS 10 // N for multispectral, 3 for RGB
#define NUM_OBJECTS 16 // Default 16, identity encoding
```

Implication: the CUDA rasterizer must be compiled with constants matching the intended channel/object-feature dimensions. Whether every experiment used a correctly recompiled CUDA extension is `UNKNOWN / NEEDS JOEL CONFIRMATION`.

## 3. Dataset audit table

| Dataset | Path | Source/provider | Purpose | Images/views | Channels | Channel names/order | Per-image channels | COLMAP poses | Masks | Train/test split | Metrics/renders | Status |
|---|---|---|---|---:|---:|---|---|---|---|---|---|---|
| Bear | `UNKNOWN / NEEDS JOEL CONFIRMATION` | User-provided link to Gaussian Grouping Bear dataset; repo docs mention `data/bear` | RGB validation and partial RGB supervision | `UNKNOWN / NEEDS JOEL CONFIRMATION` | 3 | RGB | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | No Bear output found | `UNKNOWN / NEEDS JOEL CONFIRMATION` |
| Basement full | `/home/msiau/data/tmp/jcalm/data/basement` | Arnau Marcos Almansa per user context; not file-verified | Controlled 9-channel full-supervision baseline | 50 | 9 | Folder indices `0..8`; semantic band names `UNKNOWN / NEEDS JOEL CONFIRMATION` | All 9 channels appear available per physical view | Yes: `sparse/0` and `distorted/sparse/0` binary COLMAP files with 50 images | Yes: `object_mask` has 50 files | Possible image-level split via code, but no run found | No Basement output found | Dataset present; experiment status unknown |
| Basement k variants | `UNKNOWN / NEEDS JOEL CONFIRMATION` | Derived from Basement | Partial-channel simulation | `UNKNOWN / NEEDS JOEL CONFIRMATION` | 9 predicted | `UNKNOWN / NEEDS JOEL CONFIRMATION` | Simulated subsets | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | No output found | `UNKNOWN / NEEDS JOEL CONFIRMATION` |
| Basement random variants | `UNKNOWN / NEEDS JOEL CONFIRMATION` | Derived from Basement | Random channel-subset supervision | `UNKNOWN / NEEDS JOEL CONFIRMATION` | 9 predicted | `UNKNOWN / NEEDS JOEL CONFIRMATION` | Random subsets | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | No output found | `UNKNOWN / NEEDS JOEL CONFIRMATION` |
| Basement round-robin variants | `UNKNOWN / NEEDS JOEL CONFIRMATION` | Derived from Basement | Balanced channel assignment simulation | `UNKNOWN / NEEDS JOEL CONFIRMATION` | 9 predicted, possibly 4-channel subset variants | `UNKNOWN / NEEDS JOEL CONFIRMATION` | Round-robin subsets | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | No output found | `UNKNOWN / NEEDS JOEL CONFIRMATION` |
| Basement Bernoulli variants | `UNKNOWN / NEEDS JOEL CONFIRMATION` | Derived from Basement | Bernoulli supervision simulation | `UNKNOWN / NEEDS JOEL CONFIRMATION` | 9 predicted or 4-channel subset | `UNKNOWN / NEEDS JOEL CONFIRMATION` | Bernoulli subset, at least one channel per image per user context but not code-verified | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | No output found | `UNKNOWN / NEEDS JOEL CONFIRMATION` |
| MMS/Birdhouse auxiliary | `/home/msiau/data/tmp/jcalm/data/multi-modal-studio/birdhouse` | Multi-Modal-Studio dataset | Auxiliary 9-channel multispectral experiments | Present, exact count not audited for report | 9 | Index order from `.npy`; semantic names `UNKNOWN / NEEDS JOEL CONFIRMATION` | Full multispectral arrays | Yes | `object_mask` present | Code uses `--eval` | Overnight script intended outputs not found | Auxiliary/unknown |
| Spec-NeRF xjhdesk auxiliary | `/home/msiau/data/tmp/jcalm/data/Spec-NeRF/xjhdesk` or `jcalm/data/Spec-NeRF/xjhdesk` | Spec-NeRF | Auxiliary 20-channel validation | Logs show 7 train / 2 test | 20 | `B0..B19` | Full 20 channels per view | Prepared COLMAP text | `UNKNOWN / NEEDS JOEL CONFIRMATION` | Yes, image-level `--eval` split | Outputs found for `specnerf_xjhdesk_baseline` and `phase4_specnerf_xjhdesk` | Completed auxiliary |
| Real vineyard `vinyes_20260509` | `/home/msiau/data/tmp/jcalm/data/vinyes_20260509` | Felipe Lumbreras per user context; videos under VINIA paths in scripts/logs | Real RGB/MS vineyard | 1327 registered images; 200 RGB masks | 10 predicted; registered data lacks b850 in rendered active groups | RGB + b470,b505,b525,b590,b635,b660; b850 absent/unclear | RGB images supervise `[0,1,2]`; narrowband one channel | Yes | `object_mask` 200, `semantic_mask` 200 | Logs show 1161 train / 166 test | Metrics found in `output/vinyes_20260509_sam3` | Completed but final status unknown |
| Real vineyard `vinyes_20260509_pinhole` | `/home/msiau/data/tmp/jcalm/data/vinyes_20260509_pinhole` | Felipe Lumbreras per user context | Pinhole-camera variant | 1327 registered images; 200 RGB masks | 10 predicted; active groups show no b850 | Same as above | Same as above | Yes | Masks present | Inferred 1161 / 166 from output structure and logs | Metrics found in `output/vinyes_20260509_pinhole_sam3_full` | Completed but final status unknown |
| Real vineyard `vinyes_sam3_vineid_200` | `/home/msiau/workspace/vineyard_posematch/vinyes_sam3_vineid_200` | Felipe Lumbreras + SAM3 masks | SAM3 vine-ID experiment | 1576 images; 200 RGB; registered bands include b850 176 | 10 | RGB/RGBP + b470,b505,b525,b590,b635,b660,b850 | RGB or single narrowband channel per image | Yes | `object_mask`, `semantic_mask`, `sam3_instance_mask` | Output shows 1379 train / 197 test | Metrics found in `output/vinyes_sam3_vineid_200` | Completed; trust/finality unknown |
| Real vineyard `vinyes_sam3_200` | `/home/msiau/workspace/vineyard_posematch/vinyes_sam3_200` | Felipe Lumbreras + SAM3 masks | SAM3 200-frame vineyard experiment | 1576 images; 200 RGB | 10 | RGB/RGBP + seven narrowbands | RGB or single narrowband channel per image | Yes | `object_mask`, `semantic_mask`; instance mask status unclear | `UNKNOWN / NEEDS JOEL CONFIRMATION` | Intended output `output/vinyes_sam3_200`; metrics not confirmed | Partial/unknown |
| Real vineyard `vinyes_partial200` | `/home/msiau/workspace/vineyard_posematch/vinyes_partial200` | Felipe Lumbreras | Posematch partial vineyard | 1576 images | 10 | RGB + seven narrowbands | RGB or single narrowband channel per image | Yes | Object masks present in related outputs | Logs show 1379 / 197 in some runs | Metrics found in `vineyard_posematch/output/vinyes_partial200` and `_2` | Completed older/posematch; finality unknown |
| Real vineyard `vinyes_partial250_active` | `/home/msiau/workspace/vineyard_posematch/vinyes_partial250_active` | Felipe Lumbreras | Larger active-channel vineyard run | 1978 images; 250 RGB inputs; each raw band 250 | 10 | RGB + seven narrowbands | RGB or single narrowband channel per image | Yes | `object_mask` exists | Logs show 1730 train / 248 test | Metrics found in `vineyard_posematch/output/vinyes_partial250_active` | Completed older/posematch; finality unknown |
| Real vineyard `vinyes_20260418` | `/home/msiau/data/tmp/jcalm/data/vinyes_20260418` | Felipe Lumbreras | Real vineyard reconstruction attempt | 1400 images; registered-per-band varies | 10 predicted, but available bands appear RGB + 6 MS | RGB + b470..b660; b850 absent/unclear | RGB or single narrowband | Yes | `object_mask` 200 | `UNKNOWN / NEEDS JOEL CONFIRMATION` | No final model confirmed | Partial/unknown |
| Real vineyard RGB dense turn | `/home/msiau/data/tmp/jcalm/data/vinyes_20260418_rgbdense_turn` | Felipe Lumbreras | RGB-only COLMAP trajectory test | 651 RGB images | 3 | RGB | RGB only | Yes, but quality flagged bad | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | Diagnostics exist | Failed/diagnostic |
| Real vineyard `vinyes2` | `/home/msiau/data/tmp/jcalm/data/vinyes2` | Felipe Lumbreras | Older vineyard experiment | 400 RGB/input, 1800 multispectral files | 10 | RGB + seven narrowbands | RGB or single narrowband | Yes | `object_mask` 1800 | Output exists | Metrics found in `output/vinyes2` | Completed but poor/older |
| Real vineyard `vinyes3` | `/home/msiau/data/tmp/jcalm/data/vinyes3` | Felipe Lumbreras | Older vineyard experiment | 400 RGB/input, 400 multispectral files; dense raw MS 600 per band | 10 | RGB + seven narrowbands | `UNKNOWN / NEEDS JOEL CONFIRMATION` | Yes | `object_mask` 1800 | Output exists | Metrics found in `output/vinyes3` | Completed older |
| Real vineyard `vineyard1` | `/home/msiau/data/tmp/jcalm/output/vineyard1` output only found | Felipe Lumbreras or earlier dataset; exact source unknown | Older vineyard experiment | `UNKNOWN / NEEDS JOEL CONFIRMATION` | Intended 10 | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | Output exists | Metrics show `ms_eval_num_channels=0` | Failed/invalid MS evaluation |

## 4. Experiment configuration table

| Experiment/output | Dataset | Research question | Predicted channels | Active channels | Channel strategy | k | Train/test | Iter/checkpoint | Resolution | Color representation | Embedding/decoder | Object features/classes | Masks/3D reg | Metrics/status |
|---|---|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| Bear RGB validation | Bear | Validate base RGB Gaussian Grouping behavior | 3 | RGB | Full RGB | N/A | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION`; configs exist for both SH/default and color embedding | Configs exist but run unknown | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | No output found |
| Bear R/G/B partial RGB | Bear | Validate active-channel loss with one RGB channel | 3 | One of R/G/B | Single-channel supervision | N/A | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | Configs include `single_channel_mode` | `config/train_single_channel.json`, `config/train_sc_*` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | No output found |
| Basement full 9-channel | Basement | Full-supervision baseline | 9 | All 9 | Full | 1 by professor notation | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION`; MMS 9-channel configs use color embedding + decoder | MMS configs: dim 32, hidden 128, 3 layers | num_objects 16, num_classes 256 in MMS config | Masks present in dataset; whether used in a completed run is unknown; 3D reg in MMS config | Dataset found, output not found |
| Basement k/random/round-robin/Bernoulli variants | Basement | Simulate partial MS acquisition | 9 | Subsets | k/random/round-robin/Bernoulli | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | No output found |
| `specnerf_xjhdesk_baseline` | Spec-NeRF xjhdesk | Auxiliary full 20-channel baseline | 20 | All 20 | Full | N/A | 7 / 2 from log | 40000 | `UNKNOWN / NEEDS JOEL CONFIRMATION` | Color embedding | Exact decoder settings should be read from this output cfg_args; `UNKNOWN / NEEDS JOEL CONFIRMATION` for report use | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | Completed; SSIM 0.9051, PSNR 31.5658 |
| `phase4_specnerf_xjhdesk` | Spec-NeRF xjhdesk | Auxiliary partial/single-channel 20-channel validation | 20 | Single active channel per training iteration or per frame | `single_channel_mode=True` | N/A | `UNKNOWN / NEEDS JOEL CONFIRMATION` | 30000 | 2 from cfg | Color embedding | Config in `cfg_args`; exact architecture in output | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | Completed; macro PSNR 35.7227 |
| Phase3 X-NeRF variants | X-NeRF prepared data | Auxiliary architecture/hyperparameter tests | 10 | Full or metadata-dependent | Full | N/A | 26 / 4 from logs | 30000 | `UNKNOWN / NEEDS JOEL CONFIRMATION` | Color embedding variants | Baseline/conservative/fewer_points/simple_decoder/strong_reg | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | Logs completed; final relevance unknown |
| Phase4 X-NeRF baseline | X-NeRF prepared data | Auxiliary active-channel test | 10 | Single-channel mode | `single_channel_mode=True` | N/A | 26 / 4 from log | 30000 | `UNKNOWN / NEEDS JOEL CONFIRMATION` | Color embedding | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | One log OOM, later train log completed |
| `vinyes_20260509_sam3` | `vinyes_20260509` | Real vineyard 10-channel RGB/MS + SAM3 masks | 10 | RGB `[0,1,2]` or one band | Per-image active metadata | N/A | 1161 / 166 | 40000 | 4 | Color embedding + decoder | `color_embed_dim=32`, hidden 128, 3 layers from config | num_objects 16, num_classes 514 | Masks yes, 3D reg yes | Completed; PSNR 25.2792 |
| `vinyes_20260509_pinhole_sam3_full` | `vinyes_20260509_pinhole` | Pinhole variant of vineyard SAM3 run | 10 | RGB or one band | Per-image active metadata | N/A | Inferred 1161 / 166 | 40000 | 4 | Color embedding + decoder | dim 32, hidden 128, 3 layers | num_objects 16, num_classes 514 | Masks yes, 3D reg yes | Completed; PSNR 28.2312 |
| `vinyes_sam3_vineid_200` | `vinyes_sam3_vineid_200` | Vine-ID SAM3 representation/separability run | 10 | RGB/RGBP or one band | Per-image active metadata | N/A | 1379 / 197 | 30000 | `UNKNOWN / NEEDS JOEL CONFIRMATION` | Color embedding + decoder | Config indicates dim 32/hidden 128/3 layers | num_objects 16, num_classes 362 | Masks yes, 3D reg yes | Completed; PSNR 26.9530 |
| `vinyes_partial100_clean` | Vineyard posematch partial | Older real vineyard run | 10 | Per-image active | Per-image | k appears in old cfg as `1.0`, ignored in logs | `UNKNOWN / NEEDS JOEL CONFIRMATION` | 30000 | `UNKNOWN / NEEDS JOEL CONFIRMATION` | Color embedding | Old cfg exact source unknown | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | Completed; macro PSNR 23.9460 |
| `vinyes_partial200` | Vineyard posematch partial | Older real vineyard run | 10 | Per-image active | Per-image | old cfg `k_value=1.0`; logs say ignored | `UNKNOWN / NEEDS JOEL CONFIRMATION` | 30000 | `UNKNOWN / NEEDS JOEL CONFIRMATION` | Color embedding | Old cfg exact source unknown | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | Completed; macro PSNR 23.0529 |
| `vinyes_partial200_2` | Vineyard posematch partial | Longer older real vineyard run | 10 | Per-image active | Per-image | old cfg `k_value=1.0`; logs say ignored | 1379 / 197 in related log | 60000 | 4 in related log | Color embedding | Old cfg in output | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | Completed; macro PSNR 23.2515 |
| `vinyes_partial250_active` | Vineyard posematch partial | Larger active-channel vineyard run | 10 | Per-image active | Per-image | old cfg `k_value=1.0`; ignored | 1730 / 248 from log | 30000 | `UNKNOWN / NEEDS JOEL CONFIRMATION` | Color embedding | Old cfg in output | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | Completed; macro PSNR 22.7818 |
| `vinyes2` | `data/vinyes2` | Older real vineyard run | 10 | Per-image active | `channel_mode='per_image'` in old cfg | `k_value=1.0` in old cfg | `UNKNOWN / NEEDS JOEL CONFIRMATION` | 30000 | `UNKNOWN / NEEDS JOEL CONFIRMATION` | Color embedding | Old cfg: dim 64, hidden 256, 4 layers | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | Completed but poor; macro PSNR 10.5728 |
| `vinyes3` | `data/vinyes3` | Older real vineyard run | 10 | Per-image active | `channel_mode='per_image'` in old cfg | `k_value=1.0` in old cfg | `UNKNOWN / NEEDS JOEL CONFIRMATION` | 30000 | `UNKNOWN / NEEDS JOEL CONFIRMATION` | Color embedding | Old cfg: dim 64, hidden 256, 4 layers | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | Completed; macro PSNR 20.5727 |
| `vineyard1` | Unknown older vineyard source | Older failed/invalid MS run | 10 intended | Unknown | Unknown | Unknown | Unknown | 30000 | Unknown | Unknown | Unknown | Unknown | Unknown | Metrics show no MS channels evaluated |
| `vinyes_20260418_rgbdense_turn` COLMAP diagnostic | RGB dense turn dataset | Test whether RGB trajectory can be reconstructed robustly | 3 | RGB | RGB-only COLMAP | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | COLMAP registered 651/651 but quality failed |

### Notes per experiment group

#### Bear

Verified from code/files:

- RGB and single-channel configs exist: `jcalm/config/train_base_cap1M.json`, `jcalm/config/train_color_embed.json`, `jcalm/config/train_single_channel.json`, and `jcalm/config/train_sc_*`.
- `jcalm/docs/train.md` references Bear commands such as `prepare_pseudo_label.sh bear 1` and `train.sh bear 1`.

Needs confirmation:

- Actual Bear dataset path, train/test split, commands, completed checkpoints, renders, and metric values are `UNKNOWN / NEEDS JOEL CONFIRMATION`.

#### Basement

Verified from code/files:

- Dataset exists at `/home/msiau/data/tmp/jcalm/data/basement`.
- The dataset contains 50 physical views and 9 channel folders.
- MMS-style 9-channel configs exist: `jcalm/config/gaussian_dataset/train_mms.json` and `jcalm/config/gaussian_dataset/train_mms_msiau.json`.

Needs confirmation:

- No completed Basement output folder was found.
- k/random/round-robin/Bernoulli metadata and output folders were not found.
- Whether Basement was trained with current `images_multispectral` loader, an older loader, or a custom script is `UNKNOWN / NEEDS JOEL CONFIRMATION`.

#### Real vineyard

Verified from code/files:

- Multiple real-vineyard datasets, logs, models, renders, and metrics exist.
- The clearest current-code runs are `vinyes_20260509_sam3`, `vinyes_20260509_pinhole_sam3_full`, and `vinyes_sam3_vineid_200`.
- Older posematch runs exist and include RMSE/SAM metrics not implemented in current tracked `jcalm/metrics.py`.

Needs confirmation:

- Which real vineyard run is final.
- Which failed COLMAP runs should be discussed in Section 5 versus Section 6.

## 5. Train/test and evaluation protocol

### 1. How does the code split train/test views?

Verified from code/files:

- The COLMAP reader sorts camera/image records and, when `--eval` is enabled, assigns test views by index: `idx % llffhold == 0`, with default `llffhold=8`. Evidence: `readColmapSceneInfo()` in `jcalm/scene/dataset_readers.py` and default `llffhold` in `jcalm/arguments/__init__.py`.
- When `--eval` is not enabled, all cameras are train cameras and the test set is empty. Evidence: `jcalm/scene/dataset_readers.py`.
- There is also a `--train_split` path that uses an `images_train/` folder if present. Evidence: `jcalm/scene/dataset_readers.py`.

### 2. Is the split image-level, frame-level, band-level, or grouped by physical multispectral view?

Verified from code/files:

- The current split is image-record-level over sorted COLMAP images. Evidence: `jcalm/scene/dataset_readers.py`.

Inferred from context:

- For real vineyard datasets where each RGB or narrowband frame is a separate COLMAP image, the split is effectively per registered image/band-frame, not explicitly grouped by physical multispectral acquisition.

Needs confirmation:

- Whether any dataset preparation step orders images so that RGB and narrowband frames from the same physical view are systematically grouped before splitting is `UNKNOWN / NEEDS JOEL CONFIRMATION`.

### 3. For Basement, are all channels of the same physical image kept together in train/test?

Verified from code/files:

- If Basement is loaded as one multispectral `.npy` per physical image, the current split would keep all channels of that physical image together because the camera record represents the physical view. Evidence: `jcalm/utils/camera_utils.py` and `jcalm/scene/dataset_readers.py`.

Needs confirmation:

- The visible Basement folder has `images/*.npy` and no `images_multispectral/` folder. The current reader normally detects multispectral arrays through `images_multispectral/`. Therefore, the exact loader/path used for Basement training is `UNKNOWN / NEEDS JOEL CONFIRMATION`.

### 4. For real vineyard, are RGB and narrowband views split independently or synchronized?

Verified from code/files:

- The code performs an index-based split over registered image records. Evidence: `jcalm/scene/dataset_readers.py`.

Inferred from context:

- Since real vineyard RGB and band images are separate image records, the split is not explicitly synchronized by physical multispectral view in the current code.

Needs confirmation:

- Whether metadata or naming order makes the practical split approximately synchronized is `UNKNOWN / NEEDS JOEL CONFIRMATION`.

### 5. Are metrics computed on train views, test views, all views, or selected rendered views?

Verified from code/files:

- Current `jcalm/metrics.py` reads the rendered `test` split folder and computes metrics for `ours_*`. Evidence: `jcalm/metrics.py`.
- `jcalm/render.py` can render train and/or test splits, and can restrict the number of train/test views using `--max_train_views` and `--max_test_views`.

Needs confirmation:

- Older vineyard metric logs/results with RMSE/SAM may use a different metric script. Their exact view selection is `UNKNOWN / NEEDS JOEL CONFIRMATION`.

### 6. For partial-channel experiments

Verified from code/files:

- The model predicts all configured output channels, `num_channels`. Evidence: `jcalm/gaussian_renderer/__init__.py`, `jcalm/utils/color_decoder.py`, and config files.
- During training, photometric loss sees only `viewpoint_cam.active_channels` when active channels are present. Evidence: `active_channel_loss_tensors()` in `jcalm/train.py`.
- If active channels are missing, the loss falls back to all channels in the render/GT tensors. Evidence: `jcalm/train.py`.
- Current metrics select active channels per rendered frame when `frames_index.json` has `active_channels`. Evidence: `jcalm/metrics.py`.

Inferred from context:

- For real vineyard data, unobserved bands are not directly evaluated for a frame because there is no GT for those unobserved channels in that image record.
- For Basement partial simulations, unobserved channels could be evaluated only if full-channel GT is preserved and the metric script is configured to evaluate those channels. Evidence for such an evaluation run was not found.

Needs confirmation:

- Which partial-channel Basement metrics evaluated all 9 GT channels versus only supervised active channels is `UNKNOWN / NEEDS JOEL CONFIRMATION`.

## 6. Metrics audit

### Reconstruction metrics

| Metric | Implemented where | Input | GT or predicted labels? | Split usage | Belongs in |
|---|---|---|---|---|---|
| PSNR | `jcalm/metrics.py` | Rendered image/array and GT image/array | Ground-truth reconstruction target | Current script uses test renders | Section 5 |
| SSIM | `jcalm/metrics.py` | Rendered image/array and GT image/array | Ground-truth reconstruction target | Current script uses test renders | Section 5 |
| LPIPS | `jcalm/metrics.py` | Rendered image/array and GT image/array; VGG LPIPS | Ground-truth reconstruction target | Current script uses test renders | Section 5, with caveat for non-RGB |
| L1 | `jcalm/metrics.py` | Rendered image/array and GT image/array | Ground-truth reconstruction target | Current script uses test renders | Section 5 |
| Per-band PSNR | `jcalm/metrics.py` only when `single_channel_mode` true; older results also contain per-band metrics | Per-channel render/GT arrays | Ground-truth reconstruction target | Test renders in current script | Section 5 |
| Per-band SSIM | Same as above | Per-channel render/GT arrays | Ground-truth reconstruction target | Test renders in current script | Section 5 |
| Per-band LPIPS | Same as above | Single-band expanded to 3 channels for LPIPS in current script | Ground-truth reconstruction target | Test renders in current script | Section 5, with caveat |
| Global/macro averages | `jcalm/metrics.py` for current single-channel mode; older result files have `ms_macro_*` | Aggregated per-frame/per-channel metrics | Ground-truth reconstruction target | Test renders | Section 5 |
| RMSE | Present in older vineyard `results.json`; not in current `jcalm/metrics.py` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | Section 5 only if metric script is confirmed |
| Spectral Angle Mapper | Present in older vineyard `results.json`; not in current `jcalm/metrics.py` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `UNKNOWN / NEEDS JOEL CONFIRMATION` | Section 5 only if metric script is confirmed |
| Vegetation-index metrics | Not found | N/A | N/A | N/A | Not present unless Joel confirms external files |

Important caveat:

- LPIPS is designed for RGB perceptual similarity. Current code expands one-channel inputs or selects three channels for multi-channel inputs. Evidence: `jcalm/metrics.py`. Use cautious wording if reporting LPIPS for multispectral bands.

### Direct segmentation/object metrics

| Metric | Status | Evidence | Belongs in |
|---|---|---|---|
| Pixel accuracy | Not found as a direct segmentation metric against external masks in current evaluation code | `jcalm/metrics.py` lacks it; analysis script uses classifier probe accuracy for separability | Not Section 5 segmentation unless confirmed |
| Macro-F1 | Present in separability analysis, not direct segmentation quality | `jcalm/analysis/multispectral_separability.py` | Representation analysis, not segmentation quality |
| IoU / mIoU | Not found | No implementation found in current metric script | Not present |
| Gaussian-level prediction metrics | Present as classifier separability/probe metrics | `jcalm/analysis/multispectral_separability.py` predicted_gaussian mode | Section 5 representation subsection or appendix |
| Pixel-level prediction metrics | Present as classifier separability/probe metrics | `jcalm/analysis/multispectral_separability.py` predicted_pixel mode | Section 5 representation subsection or appendix |

Important wording:

- Do not describe macro-F1 from `multispectral_separability.py` as segmentation quality unless comparing predictions to external ground-truth masks is explicitly confirmed. In the visible code, these metrics evaluate separability/classifier probes over selected labels/features.

### Feature separability analysis

| Analysis | Implemented where | Input | Labels | Output | Belongs in |
|---|---|---|---|---|---|
| PCA | `jcalm/analysis/multispectral_separability.py` | RGB, MS, RGB+MS channels, and/or color embedding depending mode | SAM masks or predicted labels depending mode | PNG plots and CSVs | Section 5 representation analysis or appendix |
| LDA | Same | Same | Same | Optional plots/metrics | Section 5 or appendix |
| t-SNE | Same | Same | Same | Optional plots | Appendix or Section 5 depending Joel's final figure choice |
| UMAP | Same | Same | Same | Optional plots | Appendix or Section 5 depending Joel's final figure choice |
| Logistic-regression separability | Same | Feature sets with train/test split | SAM/predicted labels | Accuracy, balanced accuracy, macro-F1, confusion | Section 5 representation analysis with careful wording |
| KNN/QDA extras | Same | Same | Same | Extra metrics | Appendix unless central |

Existing results:

- `jcalm/outputs/multispectral_separability/sam/results.csv`
- `jcalm/outputs/multispectral_separability/predicted_pixel/results.csv`
- `jcalm/outputs/multispectral_separability/predicted_gaussian/results.csv`
- Additional plots and confusion matrices in the same folders.

## 7. Basement experiment details

### 1. Exact dataset structure

Verified at `/home/msiau/data/tmp/jcalm/data/basement`:

| Folder/file | Verified content |
|---|---|
| `images/` | 50 `.npy` files |
| `input/` | 50 PNG files |
| `input_2/` | 50 PNG files |
| `object_mask/` | 50 mask files |
| `channels_distorted/0..8/` | 50 PNG files per channel folder |
| `channels_undistorted/0..8/images/` | 50 PNG files per channel folder |
| `sparse/0/` | COLMAP binary files `cameras.bin`, `images.bin`, `points3D.bin`; verified 1 camera and 50 images |
| `distorted/sparse/0/` | COLMAP binary files; verified 1 camera and 50 images |
| `images_multispectral/` | Not found |

### 2. How the 9 channels are loaded

Verified from code:

- Current multispectral loader expects `.npy` files under `images_multispectral/` and loads arrays as float32. Evidence: `jcalm/scene/dataset_readers.py` and `jcalm/utils/camera_utils.py`.
- The current code path treats loaded multispectral `.npy` arrays as `[H, W, C]` and permutes them to `[C, H, W]`. Evidence: `jcalm/utils/camera_utils.py`.
- If resizing is needed, the code uses `scipy.ndimage_zoom` and preserves channel count. Evidence: `jcalm/utils/camera_utils.py`.
- RGB PNGs are converted to tensors from PIL and use the first three channels. Evidence: `jcalm/utils/camera_utils.py`.

Needs confirmation:

- Basement has `images/*.npy`, not `images_multispectral/*.npy`. The exact training loader/path used for Basement is `UNKNOWN / NEEDS JOEL CONFIRMATION`.
- Channel semantic names/order for Basement channels `0..8` are `UNKNOWN / NEEDS JOEL CONFIRMATION`.
- Normalization of Basement `.npy` values is `UNKNOWN / NEEDS JOEL CONFIRMATION`; current camera utility assumes the loaded arrays are already suitable float values.

### 3. Full-supervision baseline

Verified from code/files:

- A plausible 9-channel config family exists in `jcalm/config/gaussian_dataset/train_mms.json` and `jcalm/config/gaussian_dataset/train_mms_msiau.json`.
- These configs set `num_channels=9`, `use_color_embed=true`, `color_embed_dim=32`, decoder hidden dimension 128, decoder hidden layers 3, `num_objects=16`, and `num_classes=256`.

Needs confirmation:

- Basement full-supervision output path: `UNKNOWN / NEEDS JOEL CONFIRMATION`.
- Exact command: `UNKNOWN / NEEDS JOEL CONFIRMATION`.
- Metrics: `UNKNOWN / NEEDS JOEL CONFIRMATION`.
- Completion status: `UNKNOWN / NEEDS JOEL CONFIRMATION`.

### 4. Partial-supervision simulations

| Variant | Verified status |
|---|---|
| k=0 | No output/config metadata found; `UNKNOWN / NEEDS JOEL CONFIRMATION` |
| k=0.25 | No output/config metadata found; `UNKNOWN / NEEDS JOEL CONFIRMATION` |
| k=0.5 | No output/config metadata found; `UNKNOWN / NEEDS JOEL CONFIRMATION` |
| k=1 | Basement full output not found; `UNKNOWN / NEEDS JOEL CONFIRMATION` |
| random | No Basement random output found; `UNKNOWN / NEEDS JOEL CONFIRMATION` |
| round robin | No Basement round-robin output found; `UNKNOWN / NEEDS JOEL CONFIRMATION` |
| Bernoulli | No Basement Bernoulli output found; `UNKNOWN / NEEDS JOEL CONFIRMATION` |
| 4-channel subset | No Basement 4-channel subset output found; `UNKNOWN / NEEDS JOEL CONFIRMATION` |
| 9-channel subset | Full 9-channel dataset exists; experiment output not found |

### 5. Whether k is implemented or only naming convention

Verified from current code:

- The current `jcalm/train.py` and `jcalm/arguments/__init__.py` inspected for the active-channel pipeline do not expose the professor's k formula.
- Older output `cfg_args` files for vineyard runs contain fields such as `channel_mode='per_image'`, `k_value=1.0`, and `channel_keep_prob=0.5`, but current logs say `k_value ignored` for per-image mode.

Conclusion:

- In current tracked code, k appears not to be part of the main active-channel loss implementation. Whether it existed in older code or external scripts is `UNKNOWN / NEEDS JOEL CONFIRMATION`.

### 6. Round robin channel assignment

Needs confirmation:

- Channel assignment algorithm: `UNKNOWN / NEEDS JOEL CONFIRMATION`.
- Whether assignment is balanced: `UNKNOWN / NEEDS JOEL CONFIRMATION`.
- Metadata path: `UNKNOWN / NEEDS JOEL CONFIRMATION`.

### 7. Random/Bernoulli selection

Needs confirmation:

- Seed: `UNKNOWN / NEEDS JOEL CONFIRMATION`.
- Reproducibility: `UNKNOWN / NEEDS JOEL CONFIRMATION`.
- Whether at least one channel is forced: `UNKNOWN / NEEDS JOEL CONFIRMATION`.
- Whether metadata is saved: `UNKNOWN / NEEDS JOEL CONFIRMATION`.

### 8. Which Basement experiments are complete and trustworthy

Verified:

- The Basement dataset itself exists and appears structurally complete.

Needs confirmation:

- No completed Basement experiment output was found in the visible output roots. Therefore, no Basement experiment can be marked complete/trustworthy from repository evidence alone.

### 9. What still needs Joel confirmation

- Actual Basement output paths.
- Commands used.
- Whether current code or older code loaded Basement.
- Which k/random/round-robin/Bernoulli variants completed.
- Which metrics are final.
- Channel semantic names/order.

## 8. Real vineyard experiment details

### 1. Existing vineyard datasets

| Dataset | Path | Verified contents |
|---|---|---|
| `vinyes_20260509` | `/home/msiau/data/tmp/jcalm/data/vinyes_20260509` | 1327 images, 200 RGB images, 200 object masks, 200 semantic masks, active metadata, trained output `vinyes_20260509_sam3`. |
| `vinyes_20260509_pinhole` | `/home/msiau/data/tmp/jcalm/data/vinyes_20260509_pinhole` | Similar to `vinyes_20260509`; trained output `vinyes_20260509_pinhole_sam3_full`. |
| `vinyes_sam3_vineid_200` | `/home/msiau/workspace/vineyard_posematch/vinyes_sam3_vineid_200` | 1576 images, 200 RGB images, object/semantic/instance masks, registered b850 count 176, trained output `vinyes_sam3_vineid_200`. |
| `vinyes_sam3_200` | `/home/msiau/workspace/vineyard_posematch/vinyes_sam3_200` | 1576 images and SAM3-prepared masks; intended training script exists. Metrics not confirmed. |
| `vinyes_partial200` | `/home/msiau/workspace/vineyard_posematch/vinyes_partial200` | 1576 images and active metadata; older outputs exist. |
| `vinyes_partial250_active` | `/home/msiau/workspace/vineyard_posematch/vinyes_partial250_active` | 1978 images, 250 RGB inputs, 250 raw frames per band, older output exists. |
| `vinyes_20260418` | `/home/msiau/data/tmp/jcalm/data/vinyes_20260418` | 1400 images; registered-per-band counts vary; b850 absent/unclear. |
| `vinyes_20260418_rgbdense_turn` | `/home/msiau/data/tmp/jcalm/data/vinyes_20260418_rgbdense_turn` | 651 RGB images; COLMAP trajectory diagnostic flagged bad. |
| `vinyes_fulles_1` | `/home/msiau/data/tmp/jcalm/data/vinyes_fulles_1` | 1600 images, 200 RGB images, 200 frames per RGB/MS band including b850. |
| `vinyes2` | `/home/msiau/data/tmp/jcalm/data/vinyes2` | Older dataset with 400 RGB/input files and 1800 multispectral files. |
| `vinyes3` | `/home/msiau/data/tmp/jcalm/data/vinyes3` | Older dataset with 400 RGB/input files and 400 multispectral files plus dense raw MS folders. |

### 2. Frame extraction

Verified from code:

- Frame extraction is implemented in `jcalm/prepare_vineyard_video_colmap.py`.
- Default real-video channel specification includes one RGB video and seven narrowband videos: `b470`, `b505`, `b525`, `b590`, `b635`, `b660`, `b850`.
- Generated metadata includes `band_info.json`, `frame_info.json`, and `partial_channels_summary.json`.

Verified from datasets:

- Many prepared datasets have `frames_raw` or `frames_raw_partial` folders with 200 or 250 frames per band.
- Naming convention uses band prefixes such as `rgb`, `rgbp`, `b470`, `b505`, etc.; active-channel inference in `jcalm/scene/dataset_readers.py` depends on these prefixes.

Needs confirmation:

- Exact sampling strategy for each final dataset, for example uniform sampling versus selected frame ranges: `UNKNOWN / NEEDS JOEL CONFIRMATION`.

### 3. COLMAP pipeline

Verified from code:

- `jcalm/prepare_vineyard_video_colmap.py` supports stages `extract`, `features`, `match`, `map`, `register`, `finalize`, and `diagnose`.
- Registration modes include `rgb_register`, `direct`, and `rgb_only`.
- The default strategy in code is RGB-first registration mode (`registration_mode=rgb_register`).
- The script can test RGB COLMAP variants such as exhaustive or sequential-loop matching and different camera models.
- Camera model choices include OPENCV, FOV, and PINHOLE variants in logs/scripts.

Verified from logs:

- `jcalm/logs/vinyes_20260418_rgbdense_turn_map_finalize.log` reports RGB sequence-loop PINHOLE reconstruction registered 651/651 images but failed quality diagnostics.
- `/home/msiau/data/tmp/jcalm/data/vinyes_20260418_rgbdense_turn/colmap_rgb_seqloop_pinhole/colmap_quality.json` flags `quality_ok=false` due to folded trajectory.
- `vineyard_posematch/logs/vinyes_partial200_fast.log` reports failed RGB-first registration for 37 target frames.

### 4. Known COLMAP problems

Verified problems:

- Sequential matcher process killed during one RGB-dense run. Evidence: `jcalm/logs/vinyes_20260418_rgbdense_turn.log`.
- A dense RGB reconstruction registered all frames but produced a folded/bad trajectory. Evidence: `jcalm/logs/vinyes_20260418_rgbdense_turn_map_finalize.log` and `colmap_quality.json`.
- RGB-first target-band registration had failed frames in posematch runs. Evidence: `vineyard_posematch/logs/vinyes_partial200_fast.log`.
- Some training/resume attempts encountered memory errors or missing point-cloud paths. Evidence: `vineyard_posematch/logs/vinyes_partial200_loose.log`, `vineyard_posematch/logs/vinyes_partial250_active_resume_gpu1_mapper_retry.log`.

Inferred from context:

- Repeated vineyard rows and cross-band appearance changes are plausible contributors to registration problems, consistent with user context and observed bad trajectories/failed registrations. This causal interpretation is not directly proven by a single log file.

Needs confirmation:

- Which COLMAP failures should be discussed in Section 5 and which should be reserved for Section 6: `UNKNOWN / NEEDS JOEL CONFIRMATION`.

### 5. Mask setup

Verified from code/files:

- SAM3 preparation is implemented in `jcalm/prepare_vinyes_sam3_200.py`.
- Datasets contain mask folders such as `object_mask`, `semantic_mask`, and sometimes `sam3_instance_mask`.
- Vineyard configs include label modes such as `hierarchical_composite` and `instance`. Evidence: `jcalm/config/gaussian_dataset/vinyes_20260509_sam3.json`, `jcalm/config/gaussian_dataset/vinyes_sam3_vineid_200.json`, and related configs.
- Most SAM3-prepared datasets have masks for RGB frames, typically 200 masks matching 200 RGB images.

Needs confirmation:

- Whether each final mask should be described as semantic, instance, vine-ID, hierarchical, or mixed: `UNKNOWN / NEEDS JOEL CONFIRMATION`.
- Whether masks exist only for RGB frames in every final training run: counts suggest this for several datasets, but final statement needs Joel confirmation.

### 6. Trained vineyard models

| Output path | Source dataset | Iteration | Channels | Masks | Metrics | Status |
|---|---|---:|---:|---|---|---|
| `/home/msiau/data/tmp/jcalm/output/vinyes_20260509_sam3` | `/home/msiau/workspace/jcalm/data/vinyes_20260509` | 40000 | 10 | SAM3/object masks present | SSIM 0.6335, PSNR 25.2792, LPIPS 0.5507, L1 0.0528 | Completed; finality unknown |
| `/home/msiau/data/tmp/jcalm/output/vinyes_20260509_pinhole_sam3_full` | `/home/msiau/workspace/jcalm/data/vinyes_20260509_pinhole` | 40000 | 10 | SAM3/object masks present | SSIM 0.7490, PSNR 28.2312, LPIPS 0.4166, L1 0.0348 | Completed; finality unknown |
| `/home/msiau/data/tmp/jcalm/output/vinyes_sam3_vineid_200` | `/home/msiau/workspace/vineyard_posematch/vinyes_sam3_vineid_200` | 30000 | 10 | SAM3 vine-ID/instance-style masks | SSIM 0.8293, PSNR 26.9530, LPIPS 0.2849, L1 0.0341 | Completed; finality unknown |
| `/home/msiau/workspace/vineyard_posematch/output/vinyes_partial100_clean` | Vineyard posematch partial | 30000 | 10 | `UNKNOWN / NEEDS JOEL CONFIRMATION` | ms macro PSNR 23.9460, SAM 0.4104 | Completed older; metric script unknown |
| `/home/msiau/workspace/vineyard_posematch/output/vinyes_partial200` | Vineyard posematch partial | 30000 | 10 | `UNKNOWN / NEEDS JOEL CONFIRMATION` | ms macro PSNR 23.0529, SAM 0.4742 | Completed older; metric script unknown |
| `/home/msiau/workspace/vineyard_posematch/output/vinyes_partial200_2` | Vineyard posematch partial | 60000 | 10 | `UNKNOWN / NEEDS JOEL CONFIRMATION` | ms macro PSNR 23.2515, SAM 0.4413 | Completed older; metric script unknown |
| `/home/msiau/workspace/vineyard_posematch/output/vinyes_partial250_active` | Vineyard posematch partial | 30000 | 10 | `UNKNOWN / NEEDS JOEL CONFIRMATION` | ms macro PSNR 22.7818, SAM 0.5561 | Completed older; metric script unknown |
| `/home/msiau/data/tmp/jcalm/output/vinyes2` | `/home/msiau/workspace/GaussianGroupingMultiSpectralTFG/data/vinyes2` per old cfg | 30000 | 10 | `UNKNOWN / NEEDS JOEL CONFIRMATION` | ms macro PSNR 10.5728; RGB PSNR 8.4492 | Completed but poor/older |
| `/home/msiau/data/tmp/jcalm/output/vinyes3` | `/home/msiau/workspace/GaussianGroupingMultiSpectralTFG/data/vinyes3` per old cfg | 30000 | 10 | `UNKNOWN / NEEDS JOEL CONFIRMATION` | ms macro PSNR 20.5727; RGB PSNR 20.2747 | Completed older |
| `/home/msiau/data/tmp/jcalm/output/vineyard1` | Unknown | 30000 | 10 intended | `UNKNOWN / NEEDS JOEL CONFIRMATION` | `ms_eval_num_channels=0` | Failed/invalid MS evaluation |

## 9. Representation analysis details

### 1. What data is extracted for analysis?

Verified from `jcalm/analysis/multispectral_separability.py`:

- Pixel-level color/channel features from rendered or decoded outputs.
- RGB feature subset: channels `[0,1,2]`.
- MS feature subset: channels `[3,4,5,6,7,8,9]`.
- RGB+MS feature subset: channels `[0..9]`.
- In predicted-Gaussian mode, Gaussian-level features can include the learned `COLOR_EMBEDDING`.
- Labels are collected depending on mode: SAM masks, predicted pixel classes, or predicted Gaussian classes.

### 2. What does predicted_pixel mean?

Verified from code:

- `predicted_pixel` mode uses rendered object features at pixels and the trained object classifier to assign predicted class labels per pixel.
- It is a model-derived label mode, not automatically an external ground-truth segmentation evaluation.

### 3. What does predicted_gaussian mean?

Verified from code:

- `predicted_gaussian` mode applies the trained classifier to Gaussian object features directly, producing predicted labels at Gaussian level.
- It analyzes Gaussian representation/class coherence rather than rendered segmentation against an external mask.

### 4. Which classifier is used for accuracy/F1?

Verified from code:

- The separability probe uses scikit-learn `LogisticRegression` with `StandardScaler` and `class_weight='balanced'`.
- Optional extra classifiers include KNN and QDA.
- Training-time object classifier is a 1x1 Conv2d in `jcalm/train.py`; this is different from the logistic-regression separability probe.

### 5. How is the classifier trained/tested?

Verified from code:

- For pixel modes, the analysis attempts grouped train/test splitting by view to avoid mixing pixels from the same view across splits.
- For Gaussian mode, it uses random stratified splitting where applicable.
- Metrics include accuracy, balanced accuracy, macro-F1, and confusion matrices.

Needs confirmation:

- Which split ratio and random seed should be reported for each final analysis run: `UNKNOWN / NEEDS JOEL CONFIRMATION`.

### 6. Are RGB, MS, and RGB+MS compared using decoder output channels, color embedding, object features, rendered pixel features, or something else?

Verified:

- RGB/MS/RGB+MS comparisons use channel feature subsets from the 10-channel representation/output. Evidence: `jcalm/analysis/multispectral_separability.py`.
- `COLOR_EMBEDDING` is an additional feature set in predicted-Gaussian mode. Evidence: `jcalm/analysis/multispectral_separability.py`.

Needs confirmation:

- Which feature set should be central in the report figures: decoder output channels, rendered pixel features, Gaussian color embedding, or object features: `UNKNOWN / NEEDS JOEL CONFIRMATION`.

### 7. What labels are used?

Verified:

- `sam` mode uses SAM/object masks.
- `predicted_pixel` mode uses model-predicted pixel labels from the rendered object feature classifier.
- `predicted_gaussian` mode uses model-predicted Gaussian labels.

Needs confirmation:

- Whether SAM labels in the final analysis are semantic, vine-ID, instance, background-including, or hierarchical-composite labels: `UNKNOWN / NEEDS JOEL CONFIRMATION`.

### 8. Existing plots/results

Verified output roots:

| Path | Contents |
|---|---|
| `jcalm/outputs/multispectral_separability/sam` | `results.csv`, `class_counts.csv`, `run_summary.json`, PCA plots, confusion matrices, extra metrics/plots. |
| `jcalm/outputs/multispectral_separability/predicted_pixel` | `results.csv`, `class_counts.csv`, `run_summary.json`, PCA/LDA/t-SNE/UMAP-style outputs depending file. |
| `jcalm/outputs/multispectral_separability/predicted_gaussian` | `results.csv`, `class_counts.csv`, `run_summary.json`, PCA plots, classifier metrics, color-embedding comparisons. |
| `jcalm/outputs/multispectral_separability_semantic_test_with_background/predicted_pixel` | Semantic/background variant outputs. |
| `jcalm/outputs/multispectral_separability_semantic_test_with_background/predicted_gaussian` | Semantic/background Gaussian variant outputs. |

Needs confirmation:

- Which plots are final report figures: `UNKNOWN / NEEDS JOEL CONFIRMATION`.
- Whether background-including or background-excluding results should be used: `UNKNOWN / NEEDS JOEL CONFIRMATION`.

### 9. How to describe this without confusing it with direct segmentation evaluation

Verified recommendation based on code:

- Describe this as **feature separability analysis** or **representation analysis**, not as segmentation evaluation.
- Say that classifier accuracy/macro-F1 measure how separable labels are in selected feature spaces under a probe classifier.
- Do not claim these metrics measure final segmentation quality against external ground-truth masks unless Joel confirms that exact evaluation.

## 10. Missing information checklist for Joel

### Critical

- Which experiments are final and should be reported in Section 5?
- Which output folders are obsolete and should not be cited?
- Where are the completed Bear RGB and partial-RGB outputs, if they exist?
- Where are the completed Basement full and partial-supervision outputs, if they exist?
- Which Basement variants completed successfully: k=0, k=0.25, k=0.5, k=1, random, round-robin, Bernoulli, 4-channel, 9-channel?
- Which checkpoint should be reported for each final experiment: 30000, 40000, 60000, or another?
- Which train/test split should be treated as final and defensible?
- For Basement, confirm whether all channels from the same physical view are kept together in train/test.
- For real vineyard, confirm whether RGB and narrowband frames are intentionally synchronized in the split or split independently.
- Confirm the exact Basement channel names/order.
- Confirm the exact real vineyard source/provider wording and video acquisition description.
- Confirm whether vineyard masks are semantic, instance, vine-ID, hierarchical-composite, or mixed for each final run.
- Confirm whether old RMSE/SAM metric values are trustworthy and identify the script that produced them.
- Confirm whether current CUDA constants were recompiled correctly for each reported channel/object-feature setting.

### Useful

- Exact training commands for final Bear, Basement, and vineyard experiments.
- Exact rendering/evaluation commands for final experiments.
- Which config JSON should be cited for each final experiment.
- Whether `single_channel_mode` was used for each partial-channel experiment or whether active-channel metadata alone controlled supervision.
- For random/Bernoulli Basement variants: seed, probability, reproducibility, and whether at least one channel was forced.
- For round-robin Basement variants: assignment rule and whether the assignment is balanced.
- Whether full-channel Basement GT was used to evaluate unobserved channels after partial-supervision training.
- Which COLMAP registration failures should be mentioned in Section 5 and which in Section 6.
- Whether LPIPS should be reported for multispectral bands or only included as auxiliary.
- Which representation-analysis mode is final: `sam`, `predicted_pixel`, `predicted_gaussian`, semantic-with-background, or another.

### Optional / appendix

- Whether auxiliary Spec-NeRF/X-NeRF experiments should be included in appendix.
- Whether overnight MMS/Birdhouse experiments were run elsewhere or deleted.
- Which PCA/LDA/t-SNE/UMAP plots should be included.
- Whether confusion matrices from separability analysis belong in the main text or appendix.
- Whether failed COLMAP trajectories should be visualized as diagnostic figures.
- Whether old `vinyes2`, `vinyes3`, and `vineyard1` results should be used as historical failure cases.
- Whether SAM3 mask-alignment reports should be cited or summarized in an appendix.
