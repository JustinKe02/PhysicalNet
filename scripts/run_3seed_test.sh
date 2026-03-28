#!/bin/bash
# ============================================================
# 3-Seed Reproducibility Test
# 目标: 判断旧 0.8333 是方差结果还是旧代码差异
# 总耗时: ~2.5hr (3 × ~45min + 3 × ~2min eval)
# ============================================================

set -e

# 严格 CUDA 复现环境变量
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

echo "=============================================="
echo "3-Seed Reproducibility Test"
echo "CUBLAS_WORKSPACE_CONFIG=$CUBLAS_WORKSPACE_CONFIG"
echo "PYTHONHASHSEED=$PYTHONHASHSEED"
echo "Start: $(date)"
echo "=============================================="

SEEDS=(42 123 2026)

for SEED in "${SEEDS[@]}"; do
  echo ""
  echo "====== Seed=$SEED: Training ======"
  echo "Start: $(date)"

  python tools/train_oldcfg.py \
    --seed $SEED \
    --output_dir ./output/seed_test/seed_${SEED}

  echo "Seed=$SEED training done: $(date)"

  # Test eval
  CKPT=$(ls -t output/seed_test/seed_${SEED}/repela_small_*/best_model.pth | head -1)
  echo "Checkpoint: $CKPT"

  python tools/eval.py \
    --model repela_small \
    --checkpoint "$CKPT" \
    --split test \
    --output output/eval_results/seed_${SEED}

  echo "Seed=$SEED eval done: $(date)"
done

# ============================================================
# Summary
# ============================================================
echo ""
echo "=============================================="
echo "SUMMARY"
echo "=============================================="
echo ""
echo "Seed | Val mIoU  | Test mIoU"
echo "---- | --------- | ---------"

for SEED in "${SEEDS[@]}"; do
  LOG=$(ls -t output/seed_test/seed_${SEED}/repela_small_*/train.log | head -1)
  VAL=$(grep "Done." "$LOG" | grep -oP "mIoU: \K[0-9.]+")
  TEST=$(grep "mIoU:" output/eval_results/seed_${SEED}/test_metrics.txt 2>/dev/null | head -1 | awk '{print $2}')
  echo "$SEED | $VAL     | $TEST"
done

echo ""
echo "Target: 旧 run Val=0.8333"
echo "=============================================="
echo "ALL DONE: $(date)"
echo "=============================================="
