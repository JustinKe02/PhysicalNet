#!/bin/bash
# Line 3: Combined Target — 2 experiments
# WS2_combined: train = old WS2 train + supp WS2 train, val/test = old WS2 only
set -e
cd /root/autodl-tmp/PhysicalNet

export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

PRETRAINED="output/seed_test/seed_42/repela_small_20260324_080734/best_model.pth"

echo "================================================================"
echo "  Line 3: Combined Target Experiments"
echo "  $(date)"
echo "================================================================"

echo ""
echo ">>> [1/2] WS2_combined Scratch"
python transfer/finetune.py \
    --data_root supplementary_prepared/WS2_combined --name ws2combined_scratch \
    --pretrained none \
    --num_classes 4 --epochs 300 --lr 6e-4 \
    --encoder_lr_scale 1.0 --freeze_encoder_epochs 0 \
    --warmup_epochs 5 --class_map none --seed 42 \
    --batch_size 4 --early_stop_patience 30

echo ""
echo ">>> [2/2] WS2_combined FT+reset_head"
python transfer/finetune.py \
    --data_root supplementary_prepared/WS2_combined --name ws2combined_ft_resethead \
    --pretrained "$PRETRAINED" \
    --num_classes 4 --epochs 300 --lr 3e-4 \
    --encoder_lr_scale 0.25 --freeze_encoder_epochs 5 \
    --warmup_epochs 5 --class_map none --reset_head --seed 42 \
    --batch_size 4 --early_stop_patience 30

echo ""
echo "================================================================"
echo "  Line 3 complete!"
echo "  $(date)"
echo "================================================================"
