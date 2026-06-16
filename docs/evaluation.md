# Rendering and Evaluation

## Render a Trained Model

```bash
python render.py \
  -m output/vines_20260509_object_rgb_ms \
  --iteration 30000 \
  --skip_train \
  --only_prefix rgb
```

Useful filters are `--only_prefix`, `--max_train_views`, and `--max_test_views`. Multichannel runs save RGB previews, per-channel images, NumPy channel arrays, object predictions, available ground-truth labels, and frame-index metadata.

## Reconstruction Metrics

```bash
python metrics.py -m output/vines_20260509_object_rgb_ms
```

The evaluator reports PSNR, SSIM, and LPIPS. For multispectral data, metrics are evaluated on active channels and can be aggregated per channel.

Evaluate only object predictions without loading LPIPS:

```bash
python metrics.py -m output/vines_20260509_object_rgb_ms --object_only
```

## Corrected-Mask Object Evaluation

The report's RGB/MS comparison evaluates four models on 15 held-out corrected RGB frames. Train the four runs with:

```bash
bash script/train.sh vines_20260509_object_no_color
bash script/train.sh vines_20260509_object_rgb
bash script/train.sh vines_20260509_object_ms
bash script/train.sh vines_20260509_object_rgb_ms
```

After rendering the held-out views, use `tools/eval/evaluate_final15_object_ids.py` for instance-level and class-level mIoU and Dice/F1.

## Plant-Level Measurements

The geometry entry points are:

```bash
python script/scene_scale_metrics.py --help
python script/gaussian_volume_metrics.py --help
python script/gaussian_volume_threshold_sweep.py --help
python script/classifier_leaf_surface_metrics.py --help
```

The workflow is:

1. select Gaussians belonging to a target class or instance;
2. crop the target vine when necessary;
3. convert scene units using the calibrated vineyard scale;
4. calibrate thresholds and correction factors with wooden posts; and
5. estimate trunk volume or one-sided canopy area.

These values are approximate geometric descriptors. Preserve selected labels, crop bounds, scene scale, thresholds, and correction factors with every reported result.
