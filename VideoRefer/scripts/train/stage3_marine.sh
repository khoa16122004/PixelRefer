#!/bin/bash

# Environment Variables
ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=1
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16666
ARG_RANK=${3:-0}

# Multiple conditions
if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi

echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"

# Training Arguments
GLOBAL_BATCH_SIZE=4
LOCAL_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=$[$GLOBAL_BATCH_SIZE/($WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE)]

# Log Arguments
# export TRANSFORMERS_OFFLINE=1
RUN_NAME=videorefer
OUTP_DIR=./work_dirs


python videorefer/train.py \
    --deepspeed scripts/zero2.json \
    --model_type videorefer_qwen2 \
    --model_path Qwen/Qwen2-0.5B-Instruct \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_projector_type stc_connector_v35 \
    --data_path /kaggle/working/data/processed/marine.json \
    --data_folder /kaggle/working/data/MSC-small/ \
    --image_aspect_ratio square \
    --mm_vision_select_layer -2 \
    --mm_region_encoder_type pooling \
    --num_frames 16 \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --output_dir ./work_dirs/videorefer_stage3 \
    --num_train_epochs 100 \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 99 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to tensorboard \
    --run_name $RUN_NAME \

