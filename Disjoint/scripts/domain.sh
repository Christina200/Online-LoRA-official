#!/bin/bash

python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --use_env main.py \
        core50_online_lora \
        --model vit_small_patch16_224 \
        --batch-size 64 \
        --lr 0.0005 \
        --data-path ./local_datasets \
        --output_dir ./output 