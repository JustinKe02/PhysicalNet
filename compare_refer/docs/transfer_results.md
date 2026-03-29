# Transfer Learning Results — Three-Line Summary

> Date: 2026-03-28 (updated with v2 filtered supplementary results)
> Source checkpoint used in transfer experiments: MoS2 RepELA-Small (seed 42, val mIoU = 0.8456, test mIoU = 0.8096)

---

## Line 1: External Benchmark (Public Data) — ✅ 正迁移

**WS2** (old WS2 val, 6 images)

| Strategy | Train | Val mIoU | BG | 1L | FL | ML |
|----------|-------|----------|------|------|------|------|
| Scratch | 24 | 89.68 | 0.989 | 0.914 | 0.861 | 0.823 |
| **FT+reset_head** | 24 | **91.09** | 0.989 | 0.917 | 0.880 | 0.857 |
| FT+keep_head | 24 | 89.26 | 0.990 | 0.916 | 0.857 | 0.809 |

> Source: `output/finetune_ws2_scratch/results.txt`, `output/finetune_ws2_r2_resethead/results.txt`, `output/finetune_ws2_r2_keephead/results.txt`

**Graphene** (old Graphene val, 14 images)

| Strategy | Train | Val mIoU | BG | 1L | >1L |
|----------|-------|----------|------|------|------|
| Scratch | 30 | 60.67 | 0.943 | 0.196 | 0.681 |
| **FT partial(1+2)** | 30 | **68.43** | 0.963 | 0.595 | 0.495 |
| FT full-LR + reset_head | 30 | 66.79 | 0.947 | 0.321 | 0.736 |

> Source: `output/finetune_graphene_scratch/results.txt`, `output/finetune_graphene_r3_partial/results.txt`, `output/finetune_graphene_r3_fullLR_resethead/results.txt`

**结论**: 同族 TMD 材料 (WS2) 迁移有效 (+1.41 pts)。异族材料 (Graphene) 通过部分迁移也可实现正向 (+7.76 pts)。

---

## Line 2: Internal Supplementary

### v1 — 未筛选（负迁移 ❌）

**WS2_supp v1** (56 train, 12 val)

| Strategy | Train | Val mIoU | BG | 1L | FL | ML |
|----------|-------|----------|------|------|------|------|
| **Scratch** | 56 | **70.87** | 0.977 | 0.353 | 0.561 | 0.945 |
| FT+reset_head | 56 | 64.82 | 0.909 | 0.144 | 0.602 | 0.939 |
| FT+keep_head | 56 | 62.92 | 0.825 | 0.090 | 0.651 | 0.951 |

**Gr_supp v1** (53 train, 11 val, 3 classes)

| Strategy | Train | Val mIoU | BG | FL | ML |
|----------|-------|----------|------|------|------|
| **Scratch** | 53 | **71.96** | 0.907 | 0.318 | 0.934 |
| FT+reset_head | 53 | 66.44 | 0.937 | 0.341 | 0.715 |

### v2 — 数据筛选后（负迁移→正迁移 ✅）

> 删除与原始数据基底颜色差异过大的样本后重新训练。

**WS2_supp v2** (22 train, 4 val)

| Strategy | Train | Val mIoU | BG | 1L | FL | ML |
|----------|-------|----------|------|------|------|------|
| Scratch | 22 | 75.07 | 0.949 | 0.265 | 0.817 | 0.972 |
| **FT+reset_head** | 22 | **81.21** | 0.971 | 0.442 | 0.866 | 0.970 |

**Gr_supp v2** (28 train, 6 val, 3 classes)

| Strategy | Train | Val mIoU | BG | FL | ML |
|----------|-------|----------|------|------|------|
| Scratch | 28 | 77.79 | 0.913 | 0.497 | 0.925 |
| **FT+reset_head** | 28 | **77.85** | 0.890 | 0.563 | 0.883 |

> Source: `output/finetune_ws2supp_v2_scratch/results.txt`, `output/finetune_ws2supp_v2_ft_resethead/results.txt`, `output/finetune_grsupp_v2_scratch/results.txt`, `output/finetune_grsupp_v2_ft_resethead/results.txt`

**结论**: 数据筛选后的结果与“域外噪声”假设一致。WS2_supp FT 从 −6.05 变为 **+6.14 pts**；Gr_supp 从 −5.52 改善到近似持平。

---

## Line 3: Combined Target — ✅ FT 仍优于 Scratch

**WS2_combined v1** (80 = 24 old + 56 supp, val=6 old)

| Strategy | Train | Val mIoU | BG | 1L | FL | ML |
|----------|-------|----------|------|------|------|------|
| Scratch | 80 | 86.06 | 0.977 | 0.844 | 0.834 | 0.788 |
| **FT+reset_head** | 80 | **89.69** | 0.987 | 0.901 | 0.860 | 0.840 |

**WS2_combined v2** (46 = 24 old + 22 filtered supp, val=6 old)

| Strategy | Train | Val mIoU | BG | 1L | FL | ML |
|----------|-------|----------|------|------|------|------|
| Scratch | 46 | 88.04 | 0.988 | 0.902 | 0.848 | 0.783 |
| **FT+reset_head** | 46 | **89.65** | 0.985 | 0.881 | 0.833 | 0.888 |

> Source: `output/finetune_ws2combined_v2_scratch/results.txt`, `output/finetune_ws2combined_v2_ft_resethead/results.txt`

**结论**: 筛选后 Combined Scratch 提升 (86.06→88.04)，FT 持平 (89.69≈89.65)。Combined 仍未超过 L1 old FT (91.09)。

---

## Source Augmentation — ❌ 当前未显示收益

| Source Model | Train | Verified Val mIoU | Verified Test mIoU |
|-------------|-------|-------------------|--------------------|
| **Old seed42 baseline** | 120 | **0.8456** | **0.8096** |
| MoS2 Augmented | 225 | 0.7917 | 0.4971 |

> Source: `output/seed_test/seed_42/repela_small_20260324_080734/train.log`, `output/eval_results/seed_42/test_metrics.txt`, `output/mos2_augmented_seed42/repela_small_20260327_141546/train.log`, `output/mos2_augmented_seed42/repela_small_20260327_141546/test_metrics.txt`

**结论**: 在当前设置下，Supplementary MoS2 全量并入 train 导致 val 下降 (84.56 → 79.17) 和 test 显著下降 (80.96 → 49.71)，不适合作为直接的 source augmentation 方案。

---

## 核心结论

1. **WS2 迁移有效** — FT+reset_head (91.09) > Scratch (89.68)，+1.41 pts
2. **Graphene 部分迁移有效** — FT partial(1+2) (68.43) > Scratch (60.67)，+7.76 pts
3. **自建数据负迁移在筛选后得到明显缓解** — WS2_supp 从 −6.05 改善为 **+6.14 pts**
4. **Combined 迁移仍有效** — FT+reset_head (89.65~89.69) > Scratch (86.06~88.04)
5. **Source augmentation 不可行** — 全量混入 val/test 均显著下降
6. **关键发现：在当前设置下，数据质量比数据数量更关键** — 筛选后的子集结果优于未筛选的补充集
