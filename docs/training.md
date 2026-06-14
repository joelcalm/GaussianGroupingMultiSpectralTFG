# Training

All experiments use `train.py` with a JSON configuration. Command-line arguments define scene/output paths and run length; JSON files define channels, representation capacity, regularization, and experiment variants.

## Bear RGB Validation

```bash
python train.py \
  -s data/bear \
  -m output/bear_rgb \
  --config_file config/train_color_embed.json \
  --iterations 30000 \
  --test_iterations 1000 7000 30000 \
  --save_iterations 7000 30000 \
  --eval
```

The original Gaussian Grouping baseline remains available as `bash script/train.sh bear 1`.

## Basement Multispectral Validation

```bash
python train.py \
  -s data/basement \
  -m output/basement_ms \
  --config_file config/gaussian_dataset/train_mms.json \
  --iterations 30000 \
  --test_iterations 1000 7000 30000 \
  --save_iterations 7000 30000 \
  --eval -r 2
```

Round-robin partial-channel experiments use a config with `single_channel_mode: true`; the model still predicts every configured output channel.

## Vineyard RGB + Multispectral

```bash
python train.py \
  -s data/vinyes_sam3_200 \
  -m output/vinyes_sam3_200 \
  --config_file config/gaussian_dataset/vinyes_sam3_200.json \
  --iterations 30000 \
  --test_iterations 1000 10000 30000 \
  --save_iterations 10000 30000 \
  --resolution 4 \
  --eval --train_split
```

The May 2026 RGB/MS comparison uses:

- `vinyes_20260509_pinhole_no_color.json`
- `vinyes_20260509_pinhole_rgb.json`
- `vinyes_20260509_pinhole_ms.json`
- `vinyes_20260509_pinhole_rgb_ms.json`

The historical four-run wrapper is `script/run_vinyes_20260509_pinhole_hybrid_4experiments.sh`. Review its local paths and environment variables before use.

Those May pinhole ablation configs allocate ten decoder outputs but currently supervise channels `0-8`: RGB plus the six registered narrow bands from 470 to 660 nm. Use `vinyes_sam3_200.json` with a prepared `b850_` group for the full ten-channel example above.

## Important Options

| Option | Purpose |
| --- | --- |
| `-s`, `--source_path` | Prepared scene |
| `-m`, `--model_path` | Run output directory |
| `--config_file` | JSON experiment configuration |
| `--object_path` | Mask directory relative to the scene |
| `--resolution` | Image downsampling factor |
| `--eval` | Enable a train/test split |
| `--train_split` | Use `images_train/` when present |
| `--allow_bad_colmap` | Override a failed trajectory audit after inspection |

The trainer copies label and active-channel metadata into the model directory and saves `cfg_args`, Gaussian checkpoints, the object classifier, and the appearance decoder.
