# Installation

## Requirements

- Linux with an NVIDIA GPU
- A CUDA-compatible compiler toolchain
- Conda or Mamba
- COLMAP on `PATH` for camera reconstruction
- Git LFS for datasets and checkpoints hosted on Hugging Face

The project was developed with an NVIDIA RTX 3090, Python 3.8, PyTorch 1.12.1, and CUDA 11.3.

## Core Environment

```bash
conda create -n gaussian_grouping python=3.8 -y
conda activate gaussian_grouping

conda install pytorch==1.12.1 torchvision==0.13.1 \
  torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

pip install -r requirements.txt

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

The CUDA extensions must be compiled for the Python, PyTorch, CUDA, and GPU combination used on the target machine. Do not rely on a precompiled `.so` copied from another system.

Verify the core entry points:

```bash
python train.py --help
python render.py --help
python metrics.py --help
```

## COLMAP

Install a CUDA-enabled COLMAP build when possible and verify it with:

```bash
colmap -h
```

Prepared scenes must contain a valid sparse reconstruction under `sparse/0`; see [data preparation](data_preparation.md).

## SAM3

SAM3 preprocessing is separate from Gaussian training. The repository expects a compatible `sam3_vine_video.py` environment and a checkpoint such as `weights/sam3.pt`; model weights are intentionally ignored by Git.

Verify the local script interface before processing a full video:

```bash
python sam3_vine_video.py --help
```

Keep SAM3 in a separate environment if its PyTorch/CUDA requirements conflict with the Python 3.8 Gaussian-training environment.

## Optional Components

The main vineyard experiments use SAM3 masks and do not require the older DEVA tracking or LAMA editing components inherited from Gaussian Grouping.

Poisson canopy reconstruction additionally requires `open3d`; install a version compatible with the active Python environment only when that estimator is needed.
