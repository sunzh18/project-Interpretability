#!/bin/bash

set -e

start=`date +%s`

START_DATE=$(date '+%Y-%m-%d')

PORT=$((9000 + RANDOM % 1000))
GPU=2,3
NB_GPU=3
model_name='trade_model/beta_1'

checkpoints_model='checkpoints_model/trade_model/resnet34_best_parameter.pkl'

# CUDA_VISIBLE_DEVICES=${GPU} python -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} main.py --model 'vgg11'
CUDA_VISIBLE_DEVICES=${GPU} python test_main.py \
    --model 'resnet34' --lr 0.007 \
    --batch_size 128 --epochs 30 \
    --checkpoint ${checkpoints_model} \
    --model_name ${model_name}

CUDA_VISIBLE_DEVICES=${GPU} python test_main.py \
    --model 'resnet34' --lr 0.007 \
    --batch_size 128 --epochs 30 \
    --checkpoint 'checkpoints_model/trade_model/beta_6/resnet34_parameter.pkl' \
    --model_name 'trade_model/beta_6'