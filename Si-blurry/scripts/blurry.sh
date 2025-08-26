#!/bin/bash

NOTE="olora_imnR_bs64" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)

MODE="olora"
DATASET="imagenet-r" # cifar10, cifar100, tinyimagenet, imagenet
N_TASKS=5
N=50
M=10
GPU_TRANSFORM="--gpu_transform"
USE_AMP="--use_amp"
SEEDS=$SLURM_ARRAY_TASK_ID

if [ "$DATASET" == "cifar100" ]; then
    MEM_SIZE=0 ONLINE_ITER=1
    MODEL_NAME="olora" EVAL_PERIOD=1000
    BATCHSIZE=64; LR=2e-4 OPT_NAME="adam" SCHED_NAME="default"

elif [ "$DATASET" == "tinyimagenet" ]; then
    MEM_SIZE=0 ONLINE_ITER=1
    MODEL_NAME="olora" EVAL_PERIOD=1000
    BATCHSIZE=64; LR=2e-4 OPT_NAME="adam" SCHED_NAME="default"

elif [ "$DATASET" == "imagenet-r" ]; then
    MEM_SIZE=0 ONLINE_ITER=1
    MODEL_NAME="olora" EVAL_PERIOD=1000
    BATCHSIZE=64; LR=2e-4 OPT_NAME="adam" SCHED_NAME="default"
else
    echo "Undefined setting"
    exit 1
fi

for seed in 2 3 4
do
    python main.py --mode $MODE \
    --dataset $DATASET \
    --n_tasks $N_TASKS --m $M --n $N \
    --rnd_seed $seed \
    --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
    --lr $LR --batchsize $BATCHSIZE \
    --loss-window-mean-threshold 5.2 --loss-window-variance-threshold 0.02 \
    --memory_size $MEM_SIZE $GPU_TRANSFORM --transforms --online_iter $ONLINE_ITER --data_dir ./local_datasets \
    --note $NOTE --eval_period $EVAL_PERIOD --n_worker 4 --rnd_NM
done