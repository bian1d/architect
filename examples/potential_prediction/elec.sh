#!/usr/bin/env bash

module load anaconda/2022.10
module load cuda/11.8
source activate ms

export PYTHONUNBUFFERED=1

CUDA_VISIBLE_DEVICES=0,1 fairseq-train --user-dir ../../graphormer  \
   --user-data-dir customized_dataset \
   --dataset-name streaming_elec_dataset \
   --num-workers 0 --ddp-backend=c10d \
   --best-checkpoint-metric loss \
   --task graph_prediction --criterion l1_loss --num-classes 1 --arch graphormer3d_base \
   --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 --clip-norm 5.0 \
   --batch-size 64 \
   --lr-scheduler polynomial_decay --lr 3e-4 --warmup-updates 3000 --total-num-update 50000 \
   --dropout 0.0 --attention-dropout 0.0 --weight-decay 0.001 --update-freq 1 --seed 1 \
   --tensorboard-logdir /root/tf-logs/ \
   --embed-dim 768 --ffn-embed-dim 1 --attention-heads 48 \
   --max-update 50000 --log-interval 10000 --log-format simple \
   --save-interval-updates 10000 --validate-interval-updates 10000 --keep-interval-updates 2   \
   --save-dir /root/autodl-tmp/force_32 --layers 12 --blocks 4 --required-batch-size-multiple 1  --node-loss-weight 15 \
#   --fp16 --fp16-init-scale 4 --fp16-scale-window 256
   
 