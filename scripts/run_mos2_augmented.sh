#!/bin/bash
# MoS2 Augmented Source — 1 seed trial
set -e
cd /root/autodl-tmp/PhysicalNet

export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "================================================================"
echo "  MoS2 Augmented Source Training (seed=42)"
echo "  $(date)"
echo "================================================================"

python tools/train_oldcfg.py \
    --data_root Mos2_data_augmented \
    --split_dir Mos2_data_augmented/splits \
    --model repela_small \
    --epochs 300 \
    --batch_size 4 \
    --lr 6e-4 \
    --seed 42 \
    --output_dir output/mos2_augmented_seed42

echo ""
echo "================================================================"
echo "  MoS2 Augmented Training Complete!"
echo "  $(date)"
echo "================================================================"
