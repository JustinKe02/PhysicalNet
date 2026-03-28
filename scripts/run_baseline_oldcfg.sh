#!/bin/bash
# ============================================================
# RepELA-Net 训练计划脚本
# 日期: 2026-03-23
# 目标: 用 train_oldcfg.py 复刻旧 0.8333 baseline + 自动评估
# 预计总耗时: ~45min
# ============================================================

set -e

echo "=============================================="
echo "训练计划开始 $(date)"
echo "=============================================="

# ============================================================
# Step 1: 复刻旧 baseline
# ============================================================
# 使用 train_oldcfg.py（复用 train_ablation.py 的简化训练循环）
# 固定配置:
#   RepELA-Small w/o CSE, DS=False, EMA=False, CopyPaste=False
#   无梯度裁剪, 无 early stopping, 跑满 200 epochs
#   Focal(alpha=[0.15,3.6,4.56,0.57], gamma=2.0) + Dice
#   lr=6e-4, min_lr=1e-6, weight_decay=0.01, warmup=10, seed=42
# 预计耗时: ~40min
# ============================================================
echo ""
echo "====== Step 1/3: 复刻旧 baseline (train_oldcfg.py) ======"
echo "开始: $(date)"

python tools/train_oldcfg.py \
  --output_dir ./output/baseline_oldcfg

echo "Step 1 完成: $(date)"

# ============================================================
# Step 2: Test 评估
# ============================================================
echo ""
echo "====== Step 2/3: Test 评估 ======"
echo "开始: $(date)"

# 取最新目录的 checkpoint
LATEST_DIR=$(ls -td output/baseline_oldcfg/repela_small_*/ | head -1)
CKPT="${LATEST_DIR}best_model.pth"
echo "Checkpoint: $CKPT"

python tools/eval.py \
  --model repela_small \
  --checkpoint "$CKPT" \
  --split test \
  --output output/eval_results/repela_oldcfg

echo "Step 2 完成: $(date)"

# ============================================================
# Step 3: 结果汇总
# ============================================================
echo ""
echo "====== Step 3/3: 结果汇总 ======"

TRAIN_LOG="${LATEST_DIR}train.log"
VAL_MIOU=$(grep "Done." "$TRAIN_LOG" | grep -oP "mIoU: \K[0-9.]+")
echo "Val mIoU: $VAL_MIOU (目标: ≈0.8333)"

echo ""
echo "Test 结果:"
grep -E "mIoU:|background|monolayer|fewlayer|multilayer" \
  output/eval_results/repela_oldcfg/test_metrics.txt

echo ""
echo "=============================================="
echo "全部完成 $(date)"
echo "=============================================="
echo ""
echo "下一步:"
echo "  1. 确认 Val mIoU ≈ 0.8333"
echo "  2. 更新 experiment_results.md 和 ablation_results.md"
