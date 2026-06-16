# Training

All reported experiments use `train.py` with a JSON configuration. The portable entry point is `script/train.sh`, which selects the matching scene path, output path, and config for each reported run while still allowing extra `train.py` arguments at the end.

## Reported Runs

```bash
bash script/train.sh bear_rgb
bash script/train.sh bear_rgb_round_robin
bash script/train.sh basement_all9
bash script/train.sh basement_round_robin_9
bash script/train.sh vines_20260321_rgb_ms
bash script/train.sh vines_20260418_rgb_ms
bash script/train.sh vines_20260509_rgb_ms
bash script/train.sh vines_20260509_object_no_color
bash script/train.sh vines_20260509_object_rgb
bash script/train.sh vines_20260509_object_ms
bash script/train.sh vines_20260509_object_rgb_ms
```

The wrapper accepts these environment overrides:

```bash
SCENE_DIR=/path/to/scene MODEL_DIR=output/custom ITERATIONS=30000 bash script/train.sh vines_20260509_rgb_ms
```

Additional arguments are passed through to `train.py`:

```bash
bash script/train.sh vines_20260509_rgb_ms --allow_bad_colmap
```

## Config Layout

Configs live directly under `config/gaussian_dataset`:

- `bear_rgb.json`
- `bear_rgb_round_robin.json`
- `basement_all9.json`
- `basement_round_robin_9.json`
- `vines_20260321_rgb_ms.json`
- `vines_20260418_rgb_ms.json`
- `vines_20260509_rgb_ms.json`
- `vines_20260509_object_no_color.json`
- `vines_20260509_object_rgb.json`
- `vines_20260509_object_ms.json`
- `vines_20260509_object_rgb_ms.json`

The May object-comparison configs share the same capacity and regularization settings; only `use_color_embed`, `disable_color`, and `photometric_channels` change between no-color, RGB, MS, and RGB+MS.

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
