#!/bin/bash
# Line 2: Internal Supplementary — 6 experiments
set -e
cd /root/autodl-tmp/PhysicalNet

export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

PRETRAINED="output/seed_test/seed_42/repela_small_20260324_080734/best_model.pth"

echo "================================================================"
echo "  Line 2: Internal Supplementary Experiments"
echo "  $(date)"
echo "================================================================"

# ── WS2_supp (4 classes) ──────────────────────────────────────────

echo ""
echo ">>> [1/6] WS2_supp Scratch"
python transfer/finetune.py \
    --data_root supplementary_prepared/WS2_supp --name ws2supp_scratch \
    --pretrained none \
    --num_classes 4 --epochs 300 --lr 6e-4 \
    --encoder_lr_scale 1.0 --freeze_encoder_epochs 0 \
    --warmup_epochs 5 --class_map none --seed 42 \
    --batch_size 4 --early_stop_patience 30

echo ""
echo ">>> [2/6] WS2_supp FT+reset_head"
python transfer/finetune.py \
    --data_root supplementary_prepared/WS2_supp --name ws2supp_ft_resethead \
    --pretrained "$PRETRAINED" \
    --num_classes 4 --epochs 300 --lr 3e-4 \
    --encoder_lr_scale 0.25 --freeze_encoder_epochs 5 \
    --warmup_epochs 5 --class_map none --reset_head --seed 42 \
    --batch_size 4 --early_stop_patience 30

echo ""
echo ">>> [3/6] WS2_supp FT+keep_head"
python transfer/finetune.py \
    --data_root supplementary_prepared/WS2_supp --name ws2supp_ft_keephead \
    --pretrained "$PRETRAINED" \
    --num_classes 4 --epochs 300 --lr 3e-4 \
    --encoder_lr_scale 0.25 --freeze_encoder_epochs 5 \
    --warmup_epochs 5 --class_map none --seed 42 \
    --batch_size 4 --early_stop_patience 30

# ── Gr_supp (3 classes) ──────────────────────────────────────────

echo ""
echo ">>> [4/6] Gr_supp Scratch"
python transfer/finetune.py \
    --data_root supplementary_prepared/Gr_supp --name grsupp_scratch \
    --pretrained none \
    --num_classes 3 --epochs 300 --lr 6e-4 \
    --encoder_lr_scale 1.0 --freeze_encoder_epochs 0 \
    --warmup_epochs 5 --class_map none --seed 42 \
    --batch_size 4 --early_stop_patience 30

echo ""
echo ">>> [5/6] Gr_supp FT+reset_head"
python transfer/finetune.py \
    --data_root supplementary_prepared/Gr_supp --name grsupp_ft_resethead \
    --pretrained "$PRETRAINED" \
    --num_classes 3 --epochs 300 --lr 3e-4 \
    --encoder_lr_scale 0.25 --freeze_encoder_epochs 5 \
    --warmup_epochs 5 --class_map none --reset_head --seed 42 \
    --batch_size 4 --early_stop_patience 30

echo ""
echo ">>> [6/6] Gr_supp FT+keep_head"
python transfer/finetune.py \
    --data_root supplementary_prepared/Gr_supp --name grsupp_ft_keephead \
    --pretrained "$PRETRAINED" \
    --num_classes 3 --epochs 300 --lr 3e-4 \
    --encoder_lr_scale 0.25 --freeze_encoder_epochs 5 \
    --warmup_epochs 5 --class_map none --seed 42 \
    --batch_size 4 --early_stop_patience 30

echo ""
echo "================================================================"
echo "  Line 2 complete!"
echo "  $(date)"
echo "================================================================"
