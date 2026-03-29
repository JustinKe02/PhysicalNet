#!/bin/bash
# Transfer Learning — Round 2
# Fixed LR + proper head comparison
# Pretrained: MoS2 seed=42 (selected by val mIoU, not test)

set -e
cd /root/autodl-tmp/PhysicalNet

export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

PRETRAINED="output/seed_test/seed_42/repela_small_20260324_080734/best_model.pth"

echo "================================================================"
echo "  Transfer Learning — Round 2"
echo "  $(date)"
echo "================================================================"

# ─── Graphene (3 classes) ─────────────────────────────────────────

echo ""
echo ">>> [1/4] Graphene Finetune — class_map=none (random head)"
python transfer/finetune.py \
    --data_root "other data/graphene" --name graphene_r2_nomap \
    --pretrained "$PRETRAINED" \
    --num_classes 3 --epochs 200 --lr 3e-4 \
    --encoder_lr_scale 0.25 --freeze_encoder_epochs 5 \
    --warmup_epochs 10 --class_map none --seed 42 \
    --batch_size 4 --early_stop_patience 30

echo ""
echo ">>> [2/4] Graphene Finetune — class_map=graphene (remapped head)"
python transfer/finetune.py \
    --data_root "other data/graphene" --name graphene_r2_mapped \
    --pretrained "$PRETRAINED" \
    --num_classes 3 --epochs 200 --lr 3e-4 \
    --encoder_lr_scale 0.25 --freeze_encoder_epochs 5 \
    --warmup_epochs 10 --class_map graphene --seed 42 \
    --batch_size 4 --early_stop_patience 30

# ─── WS2 (4 classes) ─────────────────────────────────────────────

echo ""
echo ">>> [3/4] WS2 Finetune — reset_head (random head, encoder-only transfer)"
python transfer/finetune.py \
    --data_root "other data/WS2_data" --name ws2_r2_resethead \
    --pretrained "$PRETRAINED" \
    --num_classes 4 --epochs 200 --lr 3e-4 \
    --encoder_lr_scale 0.25 --freeze_encoder_epochs 5 \
    --warmup_epochs 10 --class_map none --reset_head --seed 42 \
    --batch_size 4 --early_stop_patience 30 --val_split val

echo ""
echo ">>> [4/4] WS2 Finetune — keep source head"
python transfer/finetune.py \
    --data_root "other data/WS2_data" --name ws2_r2_keephead \
    --pretrained "$PRETRAINED" \
    --num_classes 4 --epochs 200 --lr 3e-4 \
    --encoder_lr_scale 0.25 --freeze_encoder_epochs 5 \
    --warmup_epochs 10 --class_map none --seed 42 \
    --batch_size 4 --early_stop_patience 30 --val_split val

echo ""
echo "================================================================"
echo "  Round 2 complete!"
echo "  $(date)"
echo "================================================================"
