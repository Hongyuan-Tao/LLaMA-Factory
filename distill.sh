#!/bin/bash

# 设置环境变量
export WANDB_DISABLED=true
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
#NPROC_PER_NODE=4
#NNODES=1
#RANK=0
#MASTER_ADDR=10.34.8.75
#MASTER_PORT=29400
python -m llamafactory.cli train examples/train_lora/qwen2vl_dt_test.yaml