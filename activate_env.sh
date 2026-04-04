#!/bin/bash
# Activate the gaussian_grouping environment with all needed env vars
# Usage: source activate_env.sh

CONDA_ENV_NAME=gaussian_grouping

if ! command -v conda >/dev/null 2>&1; then
    echo "conda command not found in PATH"
    return 1 2>/dev/null || exit 1
fi

CONDA_BASE=$(conda info --base 2>/dev/null)
if ! declare -F conda >/dev/null 2>&1; then
    if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        . "$CONDA_BASE/etc/profile.d/conda.sh"
    else
        echo "Could not find conda initialization script at $CONDA_BASE/etc/profile.d/conda.sh"
        return 1 2>/dev/null || exit 1
    fi
fi

# Resolve env path, including user-local env dirs such as ~/.conda/envs.
CONDA_ENV_PATH=$(conda info --envs 2>/dev/null | awk -v env_name="$CONDA_ENV_NAME" '$NF ~ ("/" env_name "$") { print $NF; exit }')
if [ -z "$CONDA_ENV_PATH" ] && [ -n "$CONDA_BASE" ] && [ -d "$CONDA_BASE/envs/$CONDA_ENV_NAME" ]; then
    CONDA_ENV_PATH="$CONDA_BASE/envs/$CONDA_ENV_NAME"
fi

if [ -n "$CONDA_ENV_PATH" ]; then
    conda activate "$CONDA_ENV_PATH"
else
    conda activate "$CONDA_ENV_NAME"
fi

if [ "${CONDA_DEFAULT_ENV}" != "$CONDA_ENV_NAME" ] && [ "$(basename "${CONDA_PREFIX:-}")" != "$CONDA_ENV_NAME" ]; then
    echo "Failed to activate conda env '$CONDA_ENV_NAME'"
    return 1 2>/dev/null || exit 1
fi

CONDA_ENV_PATH=${CONDA_PREFIX:-$CONDA_ENV_PATH}

# Avoid forcing /usr/include into CPATH; it can break conda CUDA extension builds.
if [ -n "${CPATH:-}" ]; then
    CLEANED_CPATH=""
    OLD_IFS="$IFS"
    IFS=':'
    for p in $CPATH; do
        if [ -n "$p" ] && [ "$p" != "/usr/include" ]; then
            CLEANED_CPATH="${CLEANED_CPATH:+$CLEANED_CPATH:}$p"
        fi
    done
    IFS="$OLD_IFS"
    export CPATH="$CLEANED_CPATH"
fi

PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)
TORCH_LIB=${CONDA_ENV_PATH}/lib/python${PYTHON_VERSION}/site-packages/torch/lib
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
