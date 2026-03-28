#!/bin/bash
# Decoder Comparison Experiments
# Fixed RepELA-Small encoder (w/o CSE), deterministic seed, 200 epochs, patience=30
# Trains 6 decoders sequentially

set -e

cd /root/autodl-tmp/PhysicalNet

export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

DECODERS=("unet" "fpn" "aspp" "segformer" "ppm" "hamburger")

for dec in "${DECODERS[@]}"; do
    echo ""
    echo "================================================================"
    echo "  Training decoder: $dec"
    echo "  $(date)"
    echo "================================================================"
    echo ""
    python tools/train_decoder_compare.py \
        --decoder "$dec" \
        --data-root Mos2_data \
        --split-dir splits \
        --output-dir output/decoder_compare \
        --epochs 200 \
        --batch-size 8 \
        --lr 6e-4 \
        --crop-size 512 \
        --seed 42 \
        --early-stop-patience 30
done

echo ""
echo "All decoder comparison experiments complete!"
echo "$(date)"
