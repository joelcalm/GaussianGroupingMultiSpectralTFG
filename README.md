# Instance-Aware Multispectral 3D Gaussian Splatting for Vineyards

This repository contains the implementation developed for the TFG **Instance-Aware Multispectral Representation of Vineyards with 3D Gaussian Splatting**. It extends [Gaussian Grouping](https://github.com/lkeab/gaussian-grouping) with:

- learned per-Gaussian RGB and multispectral appearance;
- active-channel supervision for images that observe only part of the spectrum;
- class-aware instance features supervised with tracked SAM3 masks; and
- plant-level extraction and approximate trunk-volume and canopy-area measurements.

The complete workflow is:

`dataset preparation -> COLMAP/camera setup -> SAM3 masks -> training -> rendering/evaluation -> plant measurements`

> Plant measurements are feasibility-study estimates. They depend on registration, mask quality, scale calibration, density thresholds, and reconstruction completeness.

## Method

Each Gaussian stores geometry, opacity, a learned appearance embedding, and an object feature. A small decoder maps the appearance embedding to RGB and narrow-band channels. During training, the photometric loss is evaluated only on channels available for the current view, while 2D masks and a 3D neighborhood regularizer supervise object predictions.

<p align="center">
  <img src="docs/assets/vineyard_reconstructions.png" width="92%" alt="Representative vineyard RGB reconstructions">
</p>

The vineyard model predicts ten channels:

`R, G, B, 470, 505, 525, 590, 635, 660, 850 nm`

<p align="center">
  <img src="docs/assets/plant_measurements.png" width="72%" alt="Plant-level Gaussian measurement visualizations">
</p>

## Quick Start

```bash
conda create -n gaussian_grouping python=3.8 -y
conda activate gaussian_grouping
conda install pytorch==1.12.1 torchvision==0.13.1 \
  torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

See [installation](docs/installation.md) for CUDA-extension, COLMAP, and SAM3 notes.

### Bear RGB

Download Bear from the [Gaussian Grouping data repository](https://huggingface.co/mqye/Gaussian-Grouping/tree/main/data), place it at `data/bear`, then run:

```bash
bash script/train.sh bear_rgb
```

### Basement multispectral

After preparing the scene at `data/basement`:

```bash
bash script/train.sh basement_all9
```

The Basement dataset was provided by **Arnau Marcos Almansa**.

### Vineyard RGB + multispectral

```bash
bash script/train.sh vines_20260509_rgb_ms
```

The vineyard scenes were captured and provided by **Felipe Lumbreras Ruiz** in the context of the VINIA project.

### Render and evaluate

```bash
python render.py -m output/vines_20260509_rgb_ms \
  --iteration 30000 --skip_train --only_prefix rgb

python metrics.py -m output/vines_20260509_rgb_ms
```

Models, renders, and metrics are written below `output/<run-name>`. Both `data/` and `output/` are intentionally ignored by Git.

## Documentation

- [Installation](docs/installation.md)
- [Data preparation](docs/data_preparation.md)
- [Training](docs/training.md)
- [SAM3 masks](docs/sam3_masks.md)
- [Rendering and evaluation](docs/evaluation.md)

## Repository Structure

```text
.
|-- train.py, render.py, metrics.py     Core entry points
|-- scene/                              Cameras, COLMAP readers, Gaussian model
|-- gaussian_renderer/                  Differentiable renderer integration
|-- arguments/ and utils/               Configuration, losses, shared utilities
|-- config/                             Training, SAM3, experiment configurations
|-- script/                             Reproducible and experiment wrappers
|-- tools/                              Evaluation and mask/temporal utilities
|-- docs/                               User documentation and research notes
|-- submodules/                         CUDA rasterizer and nearest-neighbor extensions
|-- data/                               Local datasets (ignored)
|-- output/                             Models and results (ignored)
`-- outputs/                            Report sources/build products (ignored)
```

The final report is maintained locally under `outputs/`; it is not required to run the code and remains excluded from Git.

## Acknowledgements

This implementation builds on [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [Gaussian Grouping](https://github.com/lkeab/gaussian-grouping). Their notices and licenses are retained in the corresponding source and third-party directories.

This work was developed in the VINIA project, funded by the Generalitat de Catalunya, Departament d'Agricultura, Ramaderia, Pesca i Alimentacio, Activitats de Demostracio (`ACC_2023_EXP_SIA002_40_0001658`).
