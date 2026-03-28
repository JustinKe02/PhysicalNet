#!/bin/bash
# Phase B: Line 2 rerun with filtered supplementary data
set -e
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /root/autodl-tmp/PhysicalNet

SRC_CKPT="output/seed_test/seed_42/repela_small_20260324_080734/best_model.pth"

echo "================================================================"
echo "  Phase B: Line 2 Rerun (filtered supplementary data)"
echo "================================================================"

# ── B1: WS2_supp Scratch ──
echo ""
echo ">>> B1: WS2_supp Scratch"
python transfer/finetune.py \
    --data_root supplementary_prepared/WS2_supp \
    --name ws2supp_v2_scratch \
    --num_classes 4 --epochs 300 --lr 6e-4 \
    --encoder_lr_scale 1.0 --freeze_encoder_epochs 0 \
    --warmup_epochs 5 --class_map none \
    --seed 42 --batch_size 4 --early_stop_patience 30

# ── B2: WS2_supp FT+reset_head ──
echo ""
echo ">>> B2: WS2_supp FT+reset_head"
python transfer/finetune.py \
    --data_root supplementary_prepared/WS2_supp \
    --name ws2supp_v2_ft_resethead \
    --pretrained $SRC_CKPT \
    --num_classes 4 --epochs 300 --lr 3e-4 \
    --encoder_lr_scale 0.25 --freeze_encoder_epochs 5 \
    --warmup_epochs 5 --class_map none --reset_head \
    --seed 42 --batch_size 4 --early_stop_patience 30

# ── B3: Gr_supp Scratch ──
echo ""
echo ">>> B3: Gr_supp Scratch"
python transfer/finetune.py \
    --data_root supplementary_prepared/Gr_supp \
    --name grsupp_v2_scratch \
    --num_classes 3 --epochs 300 --lr 6e-4 \
    --encoder_lr_scale 1.0 --freeze_encoder_epochs 0 \
    --warmup_epochs 5 --class_map none \
    --seed 42 --batch_size 4 --early_stop_patience 30

# ── B4: Gr_supp FT+reset_head ──
echo ""
echo ">>> B4: Gr_supp FT+reset_head"
python transfer/finetune.py \
    --data_root supplementary_prepared/Gr_supp \
    --name grsupp_v2_ft_resethead \
    --pretrained $SRC_CKPT \
    --num_classes 3 --epochs 300 --lr 3e-4 \
    --encoder_lr_scale 0.25 --freeze_encoder_epochs 5 \
    --warmup_epochs 5 --class_map none --reset_head \
    --seed 42 --batch_size 4 --early_stop_patience 30

echo ""
echo "================================================================"
echo "  Phase B complete!"
echo "================================================================"
echo "  B1: $(cat output/ws2supp_v2_scratch/results.txt 2>/dev/null | grep mIoU || echo 'no results')"
echo "  B2: $(cat output/ws2supp_v2_ft_resethead/results.txt 2>/dev/null | grep mIoU || echo 'no results')"
echo "  B3: $(cat output/grsupp_v2_scratch/results.txt 2>/dev/null | grep mIoU || echo 'no results')"
echo "  B4: $(cat output/grsupp_v2_ft_resethead/results.txt 2>/dev/null | grep mIoU || echo 'no results')"
echo ""
echo "  Previous baselines: WS2_supp FT+reset=64.82, Gr_supp FT+reset=66.44"
