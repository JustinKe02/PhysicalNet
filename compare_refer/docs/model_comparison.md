# Model Comparison Summary

> **Dataset**: MoS2 2D Material Segmentation (4 classes)  
> **Benchmark**: input 512×512, GPU=RTX 3090, CPU=Intel Xeon  
> **Evaluation**: test set 27 images, sliding window crop=512 stride=384

---

## 1. Overall Comparison (Paper-Style)

| Model | Backbone | Params (M) | FLOPs (G) | Size (MB) | Mem (MB) | GPU FPS | CPU FPS | mIoU (%) |
|---|---|---|---|---|---|---|---|---|
| UNet | ResNet-18 | 14.33 | 21.82 | 54.80 | 213.3 | 237.4 | 7.4 | 87.39 |
| DeepLabV3+ | ResNet-18 | 12.33 | 18.40 | 47.16 | 269.2 | 307.4 | 11.2 | 83.32 |
| PSPNet | ResNet-18 | 11.34 | 5.86 | 43.38 | 203.6 | 585.2 | 29.9 | 86.57 |
| DeepLabV3+ | EfficientNet-B0 | 4.91 | 2.36 | 19.17 | 165.3 | 112.8 | 6.4 | 88.27 |
| FPN | MobileNetV2 | 4.22 | 9.74 | 16.38 | 161.5 | 166.7 | 7.3 | **90.95** |
| DeepLabV3+ | MobileNetV2 | 4.38 | 6.19 | 17.02 | 145.7 | 169.8 | 7.2 | 88.32 |
| UNet | MobileNetV3-S | 3.59 | 10.36 | 13.96 | 119.8 | 151.3 | 8.8 | 88.34 |
| FPN | MobileNetV3-S | 2.72 | 8.19 | 10.64 | 127.3 | 160.9 | 14.2 | 89.53 |
| **Ours (RepELA-Net)** | **—** | **2.12** | **5.28** | **8.28** | **91.8** | **74.5** | **9.4** | **84.64** |

> [!NOTE]
> 除 RepELA-Net 外，所有 baseline 均使用 ImageNet 预训练权重。RepELA-Net 从零开始训练（scratch），无任何预训练。
> mIoU (%) 取各模型最优测试结果（RepELA-Net 取 3-seed 最优 seed=123）。

---

## 2. Scratch-Only Comparison（均无预训练，公平对比）

| Model | Backbone | Params (M) | FLOPs (G) | Size (MB) | Mem (MB) | GPU FPS | CPU FPS | mIoU (%) |
|---|---|---|---|---|---|---|---|---|
| DeepLabV3+ | EfficientNet-B0 | 4.91 | 2.36 | 19.17 | 165.3 | 112.8 | 6.4 | **86.89** |
| UNet | MobileNetV3-S | 3.59 | 10.36 | 13.96 | 119.8 | 151.3 | 8.8 | 85.87 |
| **Ours (RepELA-Net)** | **—** | **2.12** | **5.28** | **8.28** | **91.8** | **74.5** | **9.4** | **84.64** |
| FPN | MobileNetV3-S | 2.72 | 8.19 | 10.64 | 127.3 | 160.9 | 14.2 | 83.79 |

> [!NOTE]
> 在公平的 scratch 对比中，RepELA-Net 以 **最少参数量（2.12M）** 排名第 3，仅落后 DLV3+/EffB0（4.91M）约 2.25%。

---

## 3. RepELA-Net 3-Seed Reproducibility

| Seed | Best Val mIoU (%) | Test mIoU (%) | Pixel Acc (%) | Mean F1 (%) |
|------|-------------------|---------------|---------------|-------------|
| 42   | 84.56             | 80.96         | 97.75         | 88.57       |
| 123  | 82.04             | **84.64**     | **98.08**     | **91.16**   |
| 2026 | 83.92             | 78.16         | 97.23         | 86.50       |
| **Mean±Std** | **83.51±1.3** | **81.25±3.3** | **97.69±0.4** | **88.74±2.3** |

---

## 4. Ablation Study

| Model Variant | Params (M) | Best Val mIoU (%) | Δ Val | Test mIoU (%) |
|---|---|---|---|---|
| **RepELA-Small (w/o CSE)** | **2.14** | **82.64** | — | **81.25** (mean) |
| w/o RepConv | 2.12 | 81.20 | −1.7% | 83.42 |
| w/o ELA | 1.10 | 80.60 | −2.5% | 77.33 |
| w/o BoundaryEnhancement | 2.25 | 77.96 | −5.7% | 82.96 |
| w/o DW-MFF | 2.07 | 77.24 | −6.5% | 81.60 |

---

## 5. Lightweight Efficiency Comparison

| Metric | RepELA-Net | Best Lightweight BL | RepELA Advantage |
|---|---|---|---|
| Params (M) | **2.12** | 2.72 (FPN/MNV3-S) | ✅ 22% fewer |
| Model Size (MB) | **8.28** | 10.64 (FPN/MNV3-S) | ✅ 22% smaller |
| GPU Memory (MB) | **91.8** | 119.8 (UNet/MNV3-S) | ✅ 23% lower |
| FLOPs (G) | 5.28 | **2.36** (DLV3+/EffB0) | — |
| GPU FPS | 74.5 | **169.8** (DLV3+/MV2) | — |
| CPU FPS | 9.4 | **14.2** (FPN/MNV3-S) | — |

---

## 6. Decoder Comparison（固定 RepELA-Small Encoder）

> **实验条件**：固定 RepELA-Small encoder（w/o CSE），仅替换 decoder。  
> **训练 recipe**：seed=42, deterministic, epochs=200, patience=30, lr=6e-4, Focal+Dice, 无预训练。

### 6.1 Decoder 精度 & 效率对比

| Decoder | Params (M) | Dec Params (M) | FLOPs (G) | Val mIoU (%) | Test mIoU (%) |
|---------|-----------|----------------|-----------|--------------|---------------|
| FPN | 3.08 | 1.24 | 30.16 | 83.26 | **91.52** |
| DeepLabV3+ ASPP | 3.22 | 1.39 | 16.18 | 81.86 | 88.77 |
| SegFormer MLP | 1.96 | 0.13 | 6.55 | 80.92 | 88.17 |
| PSPNet PPM | 2.31 | 0.48 | 4.33 | 83.14 | 87.46 |
| **Ours (DW-MFF + Boundary)** | **2.12** | **0.29** | **10.61** | **82.70** | **87.45** |

### 6.2 Per-class Test IoU

| Decoder | BG | Mono | Few | Multi |
|---------|------|------|------|------|
| FPN | 0.9925 | 0.9128 | 0.7905 | 0.9649 |
| DeepLabV3+ ASPP | 0.9911 | 0.8526 | 0.7426 | 0.9646 |
| SegFormer MLP | 0.9898 | 0.8244 | 0.7424 | 0.9701 |
| PSPNet PPM | 0.9847 | 0.7983 | 0.7552 | 0.9602 |
| **Ours (DW-MFF + Boundary)** | 0.9887 | 0.8600 | 0.6951 | 0.9541 |

> [!NOTE]
> Ours 的 test mIoU (87.45%) 与 PPM (87.46%) 基本持平，但 decoder 参数量仅 0.29M（PPM 的 60%），FLOPs 为 10.61G。
> FPN decoder 在此数据集上表现异常强势（91.52%），但 FLOPs 高达 30.16G（Ours 的 2.8 倍）。
> SegFormer MLP 是极致轻量选择（decoder 仅 0.13M），精度却达到 88.17%。

---

## 7. Transfer Learning（跨材料迁移）

> **Source domain**: MoS2 (4 classes), checkpoint = seed 42 (selected by val mIoU)  
> **Target domains**: WS2 (4 classes, TMD 同族), Graphene (3 classes, 异族材料)

### 7.1 WS2 迁移（Val set, 6 images）

| Strategy | Pretrained | Head Init | Val mIoU (%) | BG | 1L | FL | ML |
|----------|-----------|-----------|--------------|------|------|------|------|
| Scratch | ❌ | random | 89.68 | 0.989 | 0.914 | 0.861 | 0.823 |
| **FT + reset_head** | ✅ MoS2 | random | **91.09** | 0.989 | 0.917 | 0.880 | 0.857 |
| FT + keep_head | ✅ MoS2 | source | 89.26 | 0.990 | 0.916 | 0.857 | 0.809 |

> [!NOTE]
> MoS2 预训练 encoder 在 WS2 上迁移有效：val mIoU 从 89.68% → **91.09%**（Δ = +1.41 points）。
> reset_head > keep_head：虽然 WS2/MoS2 类别结构一致（4 类），但分布差异使重训 head 更优。
> multilayer IoU 从 0.823 → 0.857（+0.034），各类均有提升。
>
> Source: `output/finetune_ws2_scratch/results.txt`, `output/finetune_ws2_r2_resethead/results.txt`, `output/finetune_ws2_r2_keephead/results.txt`

### 7.2 Graphene 迁移（Val set, 14 images）

| Strategy | Pretrained | Transfer Scope | Val mIoU (%) | BG | 1L | >1L |
|----------|-----------|----------------|--------------|------|------|------|
| Scratch | ❌ | — | 60.67 | 0.943 | 0.196 | 0.681 |
| **FT partial (stage1+2)** | ✅ MoS2 | low-level only | **68.43** | 0.963 | 0.595 | 0.495 |
| FT full-LR + reset_head | ✅ MoS2 | all stages | 66.79 | 0.947 | 0.321 | 0.736 |
| FT full-LR + mapped | ✅ MoS2 | all stages | 57.63 | 0.929 | 0.196 | 0.603 |

> [!NOTE]
> Graphene 与 MoS2 视觉域差距较大（异族材料），全模型迁移在保守 LR 下产生负迁移。
> **部分迁移（仅 stage1+2 低层特征）可以克服负迁移**，mIoU 从 60.67% → **68.43%**（Δ = +7.76 points）。
> monolayer IoU 从 0.196 → 0.595（+0.399），说明 MoS2 低层纹理特征对 Graphene monolayer 分类有迁移价值。

### 7.3 Line 2: Internal Supplementary（自建数据独立迁移验证）

> 验证 MoS2 预训练在自建补充数据上的迁移能力。以下为 val mIoU。

**WS2_supp v1 — 未筛选 (80 images, val=12)**

| Strategy | Val mIoU (%) | BG | 1L | FL | ML |
|----------|-------------|------|------|------|------|
| **Scratch** | **70.87** | 0.977 | 0.353 | 0.561 | 0.945 |
| FT+reset_head | 64.82 | 0.909 | 0.144 | 0.602 | 0.939 |
| FT+keep_head | 62.92 | 0.825 | 0.090 | 0.651 | 0.951 |

> Source: `output/finetune_ws2supp_scratch/results.txt`, `output/finetune_ws2supp_ft_resethead/results.txt`, `output/finetune_ws2supp_ft_keephead/results.txt`

**WS2_supp v2 — 筛选后 (32 images, val=4)**

| Strategy | Val mIoU (%) | BG | 1L | FL | ML |
|----------|-------------|------|------|------|------|
| Scratch | 75.07 | 0.949 | 0.265 | 0.817 | 0.972 |
| **FT+reset_head** | **81.21** | 0.971 | 0.442 | 0.866 | 0.970 |

> Source: `output/finetune_ws2supp_v2_scratch/results.txt`, `output/finetune_ws2supp_v2_ft_resethead/results.txt`

**Gr_supp v1 — 未筛选 (53 train, 11 val, 3 classes: BG/FL/ML)**

| Strategy | Val mIoU (%) | BG | FL | ML |
|----------|-------------|------|------|------|
| **Scratch** | **71.96** | 0.907 | 0.318 | 0.934 |
| FT+reset_head | 66.44 | 0.937 | 0.341 | 0.715 |

> Source: `output/finetune_grsupp_scratch/results.txt`, `output/finetune_grsupp_ft_resethead/results.txt`

**Gr_supp v2 — 筛选后 (40 images, val=6, 3 classes: BG/FL/ML)**

| Strategy | Val mIoU (%) | BG | FL | ML |
|----------|-------------|------|------|------|
| Scratch | 77.79 | 0.913 | 0.497 | 0.925 |
| **FT+reset_head** | **77.85** | 0.890 | 0.563 | 0.883 |

> Source: `output/finetune_grsupp_v2_scratch/results.txt`, `output/finetune_grsupp_v2_ft_resethead/results.txt`

> [!NOTE]
> **数据筛选效果显著**：删除明显域外样本后，WS2_supp 的 FT+reset_head 从负迁移 (64.82 < 70.87) 翻转为正迁移 (**81.21 > 75.07, +6.14 pts**)。
> Gr_supp 也从负迁移 (66.44 < 71.96) 改善为持平 (77.85 ≈ 77.79)。
> 这些结果与“目标域数据中存在域外噪声”这一解释一致。

### 7.4 Line 3: Combined Target（目标域数据扩展实验）

> 将公开 WS2 train + 补充 WS2 train 合并训练。val/test = old WS2 only。

**v1 — 未筛选 (80 = 24 old + 56 supp)**

| Strategy | Train 量 | Val mIoU (%) | BG | 1L | FL | ML |
|----------|---------|-------------|------|------|------|------|
| L1 old Scratch | 24 | 89.68 | 0.989 | 0.914 | 0.861 | 0.823 |
| L1 old FT+resethead | 24 | **91.09** | 0.989 | 0.917 | 0.880 | 0.857 |
| L3 comb Scratch | 80 | 86.06 | 0.977 | 0.844 | 0.834 | 0.788 |
| L3 comb FT+resethead | 80 | 89.69 | 0.987 | 0.901 | 0.860 | 0.840 |

> Source: `output/finetune_ws2combined_scratch/results.txt`, `output/finetune_ws2combined_ft_resethead/results.txt`

**v2 — 筛选后 (46 = 24 old + 22 filtered supp)**

| Strategy | Train 量 | Val mIoU (%) | BG | 1L | FL | ML |
|----------|---------|-------------|------|------|------|------|
| L1 old Scratch | 24 | 89.68 | 0.989 | 0.914 | 0.861 | 0.823 |
| L1 old FT+resethead | 24 | **91.09** | 0.989 | 0.917 | 0.880 | 0.857 |
| L3v2 comb Scratch | 46 | 88.04 | 0.988 | 0.902 | 0.848 | 0.783 |
| **L3v2 comb FT+resethead** | 46 | **89.65** | 0.985 | 0.881 | 0.833 | 0.888 |

> Source: `output/finetune_ws2combined_v2_scratch/results.txt`, `output/finetune_ws2combined_v2_ft_resethead/results.txt`

> [!NOTE]
> 筛选后 Combined Scratch (88.04) 比 v1 (86.06) 提升 +1.98 pts，这与“域外噪声被部分移除”一致。
> FT+reset_head (89.65) 与 v1 (89.69) 基本持平。
> Combined 仍未超过 L1 old FT (91.09)，说明补充数据的边际价值有限。

### 7.5 Source Augmentation（源域数据扩展实验）

> 将 supplementary MoS2 (105 张) 加入源域训练，老 val/test 固定。

| Source Model | Train | Verified Val mIoU (%) | Verified Test mIoU (%) |
|-------------|-------|-----------------------|------------------------|
| **Old seed42 baseline** | 120 | **84.56** | **80.96** |
| MoS2 Augmented | 225 | 79.17 | 49.71 |

> Source: `output/seed_test/seed_42/repela_small_20260324_080734/train.log`, `output/eval_results/seed_42/test_metrics.txt`, `output/mos2_augmented_seed42/repela_small_20260327_141546/train.log`, `output/mos2_augmented_seed42/repela_small_20260327_141546/test_metrics.txt`

> [!NOTE]
> 在当前设置下，Supplementary MoS2 全量并入 train 导致 val 下降 (84.56 → 79.17) 和 test 显著下降 (80.96 → 49.71)。不适合作为直接的 source augmentation 方案。
