#!/bin/bash

set -e

start=`date +%s`

START_DATE=$(date '+%Y-%m-%d')

PORT=$((9000 + RANDOM % 1000))
GPU=6
NB_GPU=3



# CUDA_VISIBLE_DEVICES=${GPU} python -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} main.py --model 'vgg11'
CUDA_VISIBLE_DEVICES=${GPU} python trades_main.py \
    --model 'resnet34' --lr 0.05 \
    --batch_size 64 --epochs 30 
    # --checkpoint ${checkpoints_model} \
