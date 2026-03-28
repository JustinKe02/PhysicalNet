#!/bin/bash
# Phase C: Line 3 rerun with filtered supplementary data
# Auto-triggered after Phase B completes
set -e
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /root/autodl-tmp/PhysicalNet

SRC_CKPT="output/seed_test/seed_42/repela_small_20260324_080734/best_model.pth"

echo "================================================================"
echo "  Phase C: Line 3 Rerun (filtered combined target)"
echo "  WS2_combined: old train(25) + filtered supp train(22) = 47"
echo "================================================================"

# ── C1: WS2_combined Scratch ──
echo ""
echo ">>> C1: WS2_combined Scratch"
python transfer/finetune.py \
    --data_root supplementary_prepared/WS2_combined \
    --name ws2combined_v2_scratch \
    --num_classes 4 --epochs 300 --lr 6e-4 \
    --encoder_lr_scale 1.0 --freeze_encoder_epochs 0 \
    --warmup_epochs 5 --class_map none \
    --seed 42 --batch_size 4 --early_stop_patience 30

# ── C2: WS2_combined FT+reset_head ──
echo ""
echo ">>> C2: WS2_combined FT+reset_head"
python transfer/finetune.py \
    --data_root supplementary_prepared/WS2_combined \
    --name ws2combined_v2_ft_resethead \
    --pretrained $SRC_CKPT \
    --num_classes 4 --epochs 300 --lr 3e-4 \
    --encoder_lr_scale 0.25 --freeze_encoder_epochs 5 \
    --warmup_epochs 5 --class_map none --reset_head \
    --seed 42 --batch_size 4 --early_stop_patience 30

echo ""
echo "================================================================"
echo "  Phase C complete!"
echo "================================================================"
echo "  C1: $(cat output/finetune_ws2combined_v2_scratch/results.txt 2>/dev/null | grep mIoU || echo 'no results')"
echo "  C2: $(cat output/finetune_ws2combined_v2_ft_resethead/results.txt 2>/dev/null | grep mIoU || echo 'no results')"
echo ""
echo "  Previous baselines: WS2_combined Scratch=86.06, FT+resethead=89.69"
