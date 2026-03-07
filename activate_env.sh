#!/bin/bash
# Activate the gaussian_grouping environment with all needed env vars
# Usage: source activate_env.sh

conda activate gaussian_grouping
CONDA_ENV_PATH=$(conda info --base)/envs/gaussian_grouping
export CPATH=/usr/include:$CPATH
TORCH_LIB=${CONDA_ENV_PATH}/lib/python3.8/site-packages/torch/lib
WSL_LIB=/usr/lib/wsl/lib
export LD_LIBRARY_PATH=${TORCH_LIB}:${LD_LIBRARY_PATH}
# Add WSL CUDA shim only if present (local machine)
[ -d "$WSL_LIB" ] && export LD_LIBRARY_PATH=${WSL_LIB}:${LD_LIBRARY_PATH}

# Auto-detect CUDA_HOME from nvcc location
if command -v nvcc &>/dev/null; then
    export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
elif [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME=/usr/local/cuda
else
    export CUDA_HOME=${CONDA_ENV_PATH}
fi
export PATH=${CUDA_HOME}/bin:$PATH
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Force g++/gcc 10 for CUDA 11.3 compatibility (max supported: g++ 10)
if command -v g++-10 &>/dev/null; then
    export CXX=g++-10
    export CC=gcc-10
fi

echo "Gaussian Grouping environment activated!"
echo "  Python: $(python --version)"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  CXX: ${CXX:-$(which g++)}"
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
