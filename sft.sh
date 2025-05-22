#!/bin/bash

# 设置环境变量
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"
export PATH="/usr/local/cuda-11.8/bin:$PATH"
export CUDA_VISIBLE_DEVICES="2"

# 获取Python解释器路径
PYTHON_PATH=$(which python)
if [ -z "$PYTHON_PATH" ]; then
    echo "Python interpreter not found. Please ensure Python is installed and in your PATH."
    exit 1
fi

# 执行训练命令
"$PYTHON_PATH" -m llamafactory.cli train examples/train_lora/qwen2vl_lora_sft_test.yaml