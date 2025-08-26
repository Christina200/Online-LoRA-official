#!/bin/bash

python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --use_env main.py \
        sketch_online_lora \
        --model vit_base_patch16_224 \
        --batch-size 64 \
        --MAS-weight 2000 \
        --lr 0.0003 \
        --num_tasks 20 \
        --loss-window-variance-threshold 0.04 \
        --loss-window-mean-threshold 5.6 \
        --data-path ./local_datasets \
        --output_dir ./output \