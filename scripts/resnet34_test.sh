#!/bin/bash

set -e

start=`date +%s`

START_DATE=$(date '+%Y-%m-%d')

PORT=$((9000 + RANDOM % 1000))
GPU=0
NB_GPU=3
model_name='pgd_model'

checkpoints_model='checkpoints_model/pgd_model/random_init/resnet34_parameter.pkl'

# CUDA_VISIBLE_DEVICES=${GPU} python -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} main.py --model 'vgg11'
CUDA_VISIBLE_DEVICES=${GPU} python test_main.py \
    --model 'resnet34' --lr 0.007 \
    --batch_size 128 --epochs 30 \
    --checkpoint ${checkpoints_model} \
    --model_name ${model_name}
