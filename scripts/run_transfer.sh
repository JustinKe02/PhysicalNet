#!/bin/bash
# Transfer Learning Experiments — Round 1
# 2 datasets × 2 strategies = 4 core experiments
# Pretrained: MoS2 seed=42 (selected by best val mIoU=0.8456)
# class_map=none to isolate pure transfer benefit

set -e
cd /root/autodl-tmp/PhysicalNet

export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

PRETRAINED="output/seed_test/seed_42/repela_small_20260324_080734/best_model.pth"

echo "================================================================"
echo "  Transfer Learning — Round 1 (Scratch vs Finetune-Full)"
echo "  Source checkpoint: val-best seed=42"
echo "  $(date)"
echo "================================================================"

# ─── Graphene (3 classes, train=30, val=14) ───────────────────────

echo ""
echo ">>> [1/4] Graphene - Scratch"
python transfer/finetune.py \
    --data_root "other data/graphene" --name graphene_scratch \
    --pretrained none --num_classes 3 --epochs 300 --lr 6e-4 \
    --encoder_lr_scale 1.0 --class_map none --seed 42 \
    --batch_size 4 --early_stop_patience 30

echo ""
echo ">>> [2/4] Graphene - Finetune-Full"
python transfer/finetune.py \
    --data_root "other data/graphene" --name graphene_finetune \
    --pretrained "$PRETRAINED" \
    --num_classes 3 --epochs 200 --lr 1e-4 \
    --encoder_lr_scale 0.1 --class_map none --seed 42 \
    --batch_size 4 --early_stop_patience 30

# ─── WS2 (4 classes, train=25, val=6, test=16) ───────────────────

echo ""
echo ">>> [3/4] WS2 - Scratch"
python transfer/finetune.py \
    --data_root "other data/WS2_data" --name ws2_scratch \
    --pretrained none --num_classes 4 --epochs 300 --lr 6e-4 \
    --encoder_lr_scale 1.0 --val_split val --class_map none --seed 42 \
    --batch_size 4 --early_stop_patience 30

echo ""
echo ">>> [4/4] WS2 - Finetune-Full"
python transfer/finetune.py \
    --data_root "other data/WS2_data" --name ws2_finetune \
    --pretrained "$PRETRAINED" \
    --num_classes 4 --epochs 200 --lr 1e-4 \
    --encoder_lr_scale 0.1 --val_split val --class_map none --seed 42 \
    --batch_size 4 --early_stop_patience 30

echo ""
echo "================================================================"
echo "  Round 1 complete!"
echo "  $(date)"
echo "================================================================"
