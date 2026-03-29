#!/bin/bash
# Other Data V2 (second-literature MoS2) transfer experiments
# 3 strategies: Scratch, FT+reset_head, FT+keep_head
set -e
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /root/autodl-tmp/PhysicalNet

SRC_CKPT="output/seed_test/seed_42/repela_small_20260324_080734/best_model.pth"
DATA_ROOT="other_datav2_prepared"

echo "================================================================"
echo "  Other Data V2: MoS2 Literature #2 Transfer Experiments"
echo "  Data: train=22, val=6, test=7, 4-class"
echo "================================================================"

# ── 1. Scratch ──
echo ""
echo ">>> [1/3] MoS2v2 Scratch"
python transfer/finetune.py \
    --data_root $DATA_ROOT \
    --name mos2v2_scratch \
    --num_classes 4 --epochs 300 --lr 6e-4 \
    --encoder_lr_scale 1.0 --freeze_encoder_epochs 0 \
    --warmup_epochs 5 --class_map none \
    --seed 42 --batch_size 4 --early_stop_patience 30

# ── 2. FT+reset_head ──
echo ""
echo ">>> [2/3] MoS2v2 FT+reset_head"
python transfer/finetune.py \
    --data_root $DATA_ROOT \
    --name mos2v2_ft_resethead \
    --pretrained $SRC_CKPT \
    --num_classes 4 --epochs 300 --lr 3e-4 \
    --encoder_lr_scale 0.25 --freeze_encoder_epochs 5 \
    --warmup_epochs 5 --class_map none --reset_head \
    --seed 42 --batch_size 4 --early_stop_patience 30

# ── 3. FT+keep_head ──
echo ""
echo ">>> [3/3] MoS2v2 FT+keep_head"
python transfer/finetune.py \
    --data_root $DATA_ROOT \
    --name mos2v2_ft_keephead \
    --pretrained $SRC_CKPT \
    --num_classes 4 --epochs 300 --lr 3e-4 \
    --encoder_lr_scale 0.25 --freeze_encoder_epochs 5 \
    --warmup_epochs 5 --class_map none \
    --seed 42 --batch_size 4 --early_stop_patience 30

echo ""
echo "================================================================"
echo "  All 3 experiments complete!"
echo "================================================================"
echo "  [1] Scratch:       $(grep mIoU output/finetune_mos2v2_scratch/results.txt 2>/dev/null || echo 'no results')"
echo "  [2] FT+reset_head: $(grep mIoU output/finetune_mos2v2_ft_resethead/results.txt 2>/dev/null || echo 'no results')"
echo "  [3] FT+keep_head:  $(grep mIoU output/finetune_mos2v2_ft_keephead/results.txt 2>/dev/null || echo 'no results')"
echo ""
echo "  Compare: L1 WS2 Scratch=89.68, FT+reset=91.09"
