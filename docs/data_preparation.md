# Data Preparation

## Expected Scene Layout

Training consumes a COLMAP-style scene:

```text
data/<scene>/
|-- images/                     Registered RGB and/or narrow-band images
|-- sparse/0/                   COLMAP cameras, images, and points3D files
|-- object_mask/                Optional integer instance/class masks
|-- images_train/               Optional explicit training split
|-- metadata/
|   |-- active_channels.json    Channels supervised by each image
|   `-- class_map.json          Optional label metadata
`-- colmap_quality.json         Optional trajectory-quality audit
```

An experiment can select another mask directory with `--object_path`.

## Active Channels

The ten-channel vineyard representation uses:

| Image prefix | Active channels |
| --- | --- |
| `rgb_` | `0, 1, 2` |
| `b470_` | `3` |
| `b505_` | `4` |
| `b525_` | `5` |
| `b590_` | `6` |
| `b635_` | `7` |
| `b660_` | `8` |
| `b850_` | `9` |

Store explicit per-frame mappings in `metadata/active_channels.json` when names do not follow these prefixes.

## Bear

Bear is published by the Gaussian Grouping authors:

<https://huggingface.co/mqye/Gaussian-Grouping/tree/main/data>

```bash
git lfs install
git clone https://huggingface.co/mqye/Gaussian-Grouping /tmp/Gaussian-Grouping-data
cp -a /tmp/Gaussian-Grouping-data/data/bear data/bear
```

The prepared scene should contain `images/`, `object_mask/`, and `sparse/0/`.

## Basement Multispectral Scene

The Basement dataset was provided by **Arnau Marcos Almansa**. It contains complete nine-channel observations at shared camera poses and is used as the controlled multispectral validation scene.

Use `prepare_mms_dataset.py` to convert the source capture into the repository's COLMAP-style representation:

```bash
python prepare_mms_dataset.py --help
```

Place the result at `data/basement` for the documented commands, or substitute its actual path with `-s`.

## Vineyard Scenes

The vineyard RGB and narrow-band videos were captured and provided by **Felipe Lumbreras Ruiz**. The report evaluates acquisitions from March 21, April 18, and May 9, 2026.

Preparation consists of:

1. sampling RGB and narrow-band video frames;
2. building or importing a shared COLMAP reconstruction;
3. exporting an undistorted training scene;
4. generating SAM3 masks on ordered RGB frames;
5. composing class-aware instance masks and active-channel metadata; and
6. auditing the registered trajectory before training.

Relevant entry points:

```bash
python prepare_vineyard_video_colmap.py --help
python script/prepare_undistorted_pinhole_scene.py --help
python compose_hierarchical_vineyard_labels.py --help
```

The `run_vinyes_*.sh` files record specific experimental runs. Prefer the Python entry points and portable commands documented here when adapting the pipeline to another dataset.

## Train/Test Split

Use `--eval` for evaluation mode. With `--train_split`, the loader uses `images_train/` and treats other registered images as test views. Without `images_train/`, it falls back to the loader's index-based split.
