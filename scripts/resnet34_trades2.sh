#!/bin/bash

set -e

start=`date +%s`

START_DATE=$(date '+%Y-%m-%d')

PORT=$((9000 + RANDOM % 1000))
GPU=3,5
NB_GPU=3
model_name='trade_model/beta_6'
beta=6

checkpoints_model='checkpoints_model/trade_model/beta_6/ImageNet_resnet34_parameter.pkl'

# CUDA_VISIBLE_DEVICES=${GPU} python -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} main.py --model 'vgg11'
CUDA_VISIBLE_DEVICES=${GPU} python trades_main.py \
    --model 'resnet34' --lr 0.0005 \
    --batch_size 32 --epochs 100 \
    --model_name ${model_name} --beta ${beta}   \
    --checkpoint ${checkpoints_model} 
