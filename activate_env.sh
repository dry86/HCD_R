#!/bin/bash

# 检查是否提供了环境名称参数
if [ $# -eq 0 ]; then
    echo "Usage: source activate_env.sh <environment_name>"
    echo "Example: source activate_env.sh WWS_R1_V"
    return 1
fi

ENV_NAME=$1
CONDA_BASE="/newdisk/public/miniconda3"

# 检查环境是否存在
if ! conda env list | grep -q "$ENV_NAME"; then
    echo "Error: Environment '$ENV_NAME' does not exist"
    return 1
fi

# 设置 CUDA_HOME 并激活环境
export CUDA_HOME="$CONDA_BASE/envs/$ENV_NAME"
conda activate "$ENV_NAME"

echo "Activated environment: $ENV_NAME"
echo "CUDA_HOME set to: $CUDA_HOME" 