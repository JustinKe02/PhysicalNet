#!/bin/bash
# Graphene Round 3 — Fix negative transfer
# 3 strategies targeting the root cause: MoS2 encoder features too domain-specific
set -e
cd /root/autodl-tmp/PhysicalNet

export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

PRETRAINED="output/seed_test/seed_42/repela_small_20260324_080734/best_model.pth"

echo "================================================================"
echo "  Graphene Round 3 — Fix negative transfer"
echo "  $(date)"
echo "================================================================"

# Strategy 1: Full-LR transfer (same LR as scratch, but start from pretrained)
# Rationale: give encoder full freedom to adapt while benefiting from init
echo ""
echo ">>> [1/3] Full-LR transfer (lr=6e-4, enc_scale=1.0, class_map=graphene)"
python transfer/finetune.py \
    --data_root "other data/graphene" --name graphene_r3_fullLR \
    --pretrained "$PRETRAINED" \
    --num_classes 3 --epochs 300 --lr 6e-4 \
    --encoder_lr_scale 1.0 --freeze_encoder_epochs 0 \
    --warmup_epochs 5 --class_map graphene --seed 42 \
    --batch_size 4 --early_stop_patience 30

# Strategy 2: Partial transfer (only stage1+2, low-level features only)
# Rationale: low-level features (edges, textures) are more universal
echo ""
echo ">>> [2/3] Partial transfer (stage1+2 only, class_map=graphene)"
python transfer/finetune.py \
    --data_root "other data/graphene" --name graphene_r3_partial \
    --pretrained "$PRETRAINED" \
    --num_classes 3 --epochs 300 --lr 6e-4 \
    --encoder_lr_scale 0.5 --freeze_encoder_epochs 0 \
    --warmup_epochs 5 --class_map graphene --seed 42 \
    --transfer_stages 1,2 \
    --batch_size 4 --early_stop_patience 30

# Strategy 3: Full-LR + reset_head (encoder-only, no head help)
# Rationale: isolate whether encoder init itself helps (without class_map)
echo ""
echo ">>> [3/3] Full-LR + reset_head (lr=6e-4, enc_scale=1.0, no map)"
python transfer/finetune.py \
    --data_root "other data/graphene" --name graphene_r3_fullLR_resethead \
    --pretrained "$PRETRAINED" \
    --num_classes 3 --epochs 300 --lr 6e-4 \
    --encoder_lr_scale 1.0 --freeze_encoder_epochs 0 \
    --warmup_epochs 5 --class_map none --reset_head --seed 42 \
    --batch_size 4 --early_stop_patience 30

echo ""
echo "================================================================"
echo "  Graphene Round 3 complete!"
echo "  $(date)"
echo "================================================================"
