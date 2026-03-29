#!/bin/bash
# MoS2v2 3-Fold Cross Validation: Scratch vs FT+reset_head
set -e
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /root/autodl-tmp/PhysicalNet

SRC_CKPT="output/seed_test/seed_42/repela_small_20260324_080734/best_model.pth"

echo "================================================================"
echo "  MoS2v2 3-Fold Cross Validation"
echo "  6 experiments: 3 folds × {Scratch, FT+reset_head}"
echo "================================================================"

for FOLD in 0 1 2; do
    DATA="other_datav2_cv/fold${FOLD}"
    echo ""
    echo "════════════════════════════════════════"
    echo "  Fold ${FOLD}"
    echo "════════════════════════════════════════"

    # Scratch
    echo ""
    echo ">>> Fold${FOLD} Scratch"
    python transfer/finetune.py \
        --data_root $DATA \
        --name mos2v2_cv_fold${FOLD}_scratch \
        --num_classes 4 --epochs 300 --lr 6e-4 \
        --encoder_lr_scale 1.0 --freeze_encoder_epochs 0 \
        --warmup_epochs 5 --class_map none \
        --seed 42 --batch_size 4 --early_stop_patience 30

    # FT+reset_head
    echo ""
    echo ">>> Fold${FOLD} FT+reset_head"
    python transfer/finetune.py \
        --data_root $DATA \
        --name mos2v2_cv_fold${FOLD}_ft_resethead \
        --pretrained $SRC_CKPT \
        --num_classes 4 --epochs 300 --lr 3e-4 \
        --encoder_lr_scale 0.25 --freeze_encoder_epochs 5 \
        --warmup_epochs 5 --class_map none --reset_head \
        --seed 42 --batch_size 4 --early_stop_patience 30
done

echo ""
echo "================================================================"
echo "  3-Fold CV Complete! Summary:"
echo "================================================================"
for FOLD in 0 1 2; do
    echo "  Fold${FOLD} Scratch:       $(grep mIoU output/finetune_mos2v2_cv_fold${FOLD}_scratch/results.txt 2>/dev/null || echo 'N/A')"
    echo "  Fold${FOLD} FT+reset_head: $(grep mIoU output/finetune_mos2v2_cv_fold${FOLD}_ft_resethead/results.txt 2>/dev/null || echo 'N/A')"
done
