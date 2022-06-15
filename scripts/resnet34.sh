#!/bin/bash

set -e

start=`date +%s`

START_DATE=$(date '+%Y-%m-%d')

PORT=$((9000 + RANDOM % 1000))
GPU=4,5
NB_GPU=3

checkpoints_model='checkpoints_model/pgd_model/resnet34_parameter.pkl'

# CUDA_VISIBLE_DEVICES=${GPU} python -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} main.py --model 'vgg11'
CUDA_VISIBLE_DEVICES=${GPU} python main.py \
    --model 'resnet34' --lr 0.001 \
    --batch_size 512 --epochs 20 \
    --checkpoint ${checkpoints_model} 
