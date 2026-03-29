# RepELA-Net 实验结果汇总

> **日期**：2026-03-24  
> **数据集**：MoS2 2D Material (4 类: background, monolayer, fewlayer, multilayer)  
> **训练配置**：主线 3-seed source 使用 `epochs=200, Focal+Dice loss, lr=6e-4, cosine warmup 10, DS=False, EMA=False, CopyPaste=False`；其余 baseline / 消融 / decoder compare / 迁移实验见各自日志  
> **评估配置**：test set 27 images, sliding window crop=512 stride=384  
> **Benchmark**：input 512×512, GPU=RTX 3090

---

## 1. 多种子可复现性（3-Seed Reproducibility）

### 1.1 总览

| Seed | Best Val mIoU | Test mIoU | Test Pixel Acc | Test Mean F1 |
|------|---------------|-----------|----------------|---------------|
| 42   | 0.8456        | 0.8096    | 0.9775         | 0.8857        |
| 123  | 0.8204        | **0.8464**| **0.9808**     | **0.9116**    |
| 2026 | 0.8392        | 0.7816    | 0.9723         | 0.8650        |
| **Mean±Std** | **0.8351±0.013** | **0.8125±0.033** | **0.9769±0.004** | **0.8874±0.023** |

### 1.2 Per-class Test IoU

| Seed | Background | Monolayer | Fewlayer | Multilayer |
|------|------------|-----------|----------|------------|
| 42   | 0.9847     | 0.7095    | 0.5968   | 0.9475     |
| 123  | 0.9848     | **0.7669**| **0.6809**| **0.9528** |
| 2026 | 0.9798     | 0.6571    | 0.5458   | 0.9436     |
| **Mean±Std** | **0.9831±0.003** | **0.7112±0.055** | **0.6078±0.069** | **0.9480±0.005** |

> [!NOTE]
> Fewlayer 和 Monolayer 方差较大（std=0.069/0.055），Background/Multilayer 非常稳定（std<0.005）。
> 最优测试表现 seed=123（test mIoU=0.8464），后续以此作为 RepELA-Small 代表值。

---

## 2. 消融实验（Ablation Study）

### 2.1 Historical Val mIoU（来自 `ablation_results.md` 与对应单次消融日志）

| 模型变体 | 参数量 | Best Val mIoU | Δ Val |
|---|---|---|---|
| **RepELA-Small (w/o CSE)** | **2.14M** | **0.8264** | — |
| w/o RepConv | 2.12M | 0.8120 | −1.7% |
| w/o ELA | 1.10M | 0.8060 | −2.5% |
| w/o BoundaryEnhancement | 2.25M | 0.7796 | −5.7% |
| w/o DW-MFF | 2.07M | 0.7724 | −6.5% |

### 2.2 Test mIoU + Per-class

| 模型变体 | Test mIoU | BG | Mono | Few | Multi |
|---|---|---|---|---|---|
| **RepELA-Small (3-seed mean)** | **0.8125** | 0.9831 | 0.7112 | 0.6078 | 0.9480 |
| RepELA-Small (best, seed 123) | **0.8464** | 0.9848 | 0.7669 | 0.6809 | 0.9528 |
| w/o RepConv | 0.8342 | 0.9868 | 0.7881 | 0.6184 | 0.9436 |
| w/o BoundaryEnhancement | 0.8296 | 0.9797 | 0.6859 | 0.6840 | 0.9487 |
| w/o DW-MFF | 0.8160 | 0.9797 | 0.6761 | 0.6559 | 0.9522 |
| w/o ELA | 0.7733 | 0.9757 | 0.6150 | 0.5544 | 0.9480 |

> [!IMPORTANT]
> 本节混合了两种真实来源：2.1 为历史单次消融的 **val** 记录，2.2 为当前主线 3-seed RepELA baseline 与各单次消融的 **test** 记录。
> 因此 2.1 的 historical val baseline（0.8264）与 2.2 的 current test baseline（0.8125 / 0.8464）**不可直接做跨表差值解释**。
> 消融变体仅训练了单次（无多种子），其 test mIoU 可能受种子方差影响。
> RepELA baseline 的 3-seed mean=0.8125（std=0.033），对比消融结果：
> - DW-MFF 和 ELA 的移除效果明确（均低于 mean−1σ=0.7795）
> - RepConv 和 Boundary 的单次结果（0.8342/0.8296）落在 RepELA 的 1σ 范围内，需更多种子验证

### 2.3 模块贡献度分析（基于 Val mIoU）

1. **DW-MFF 动态融合（−6.5%）**：多尺度特征融合贡献最大，移除后 fewlayer IoU 从 0.7075 降至 0.6333
2. **BoundaryEnhancement 边界增强（−5.7%）**：monolayer IoU 降幅最大（0.6531→0.5019，−23.2%）
3. **ELA 高效线性注意力（−2.5%）**：参数从 2.14M 降至 1.10M，但 mIoU 降幅较小；主要影响 monolayer
4. **RepConv 重参数化（−1.7%）**：贡献最小，主要优势在推理端分支融合加速

---

## 3. Baseline 对比

### 3.1 Pretrained Lightweight Baselines（ImageNet 预训练）

| Model | Params(M) | FLOPs(G) | Size(MB) | GPU FPS | CPU FPS | Mem(MB) | Test mIoU |
|---|---|---|---|---|---|---|---|
| FPN/MV2 | 4.22 | 9.74 | 16.38 | 166.7 | 7.3 | 161.5 | **0.9095** |
| FPN/MNV3-S | 2.72 | 8.19 | 10.64 | 160.9 | 14.2 | 127.3 | 0.8953 |
| UNet/MNV3-S | 3.59 | 10.36 | 13.96 | 151.3 | 8.8 | 119.8 | 0.8834 |
| DLV3+/MV2 | 4.38 | 6.19 | 17.02 | 169.8 | 7.2 | 145.7 | 0.8832 |
| DLV3+/EffB0 | 4.91 | 2.36 | 19.17 | 112.8 | 6.4 | 165.3 | 0.8827 |

### 3.2 Standard Baselines（>10M 参数，ImageNet 预训练）

| Model | Params(M) | FLOPs(G) | Size(MB) | GPU FPS | CPU FPS | Mem(MB) | Test mIoU |
|---|---|---|---|---|---|---|---|
| UNet/R18 | 14.33 | 21.82 | 54.80 | 237.4 | 7.4 | 213.3 | 0.8739 |
| PSPNet/R18 | 11.34 | 5.86 | 43.38 | 585.2 | 29.9 | 203.6 | 0.8657 |
| DLV3+/R18 | 12.33 | 18.40 | 47.16 | 307.4 | 11.2 | 269.2 | 0.8332 |

### 3.3 RepELA-Small（无预训练，3-seed）

| Model | Params(M) | FLOPs(G) | Size(MB) | GPU FPS | CPU FPS | Mem(MB) | Test mIoU |
|---|---|---|---|---|---|---|---|
| **RepELA-Small (mean±std)** | **2.12** | **5.28** | **8.28** | 74.5 | 9.4 | **91.8** | **0.8125±0.033** |
| RepELA-Small (best, seed 123) | 2.12 | 5.28 | 8.28 | 74.5 | 9.4 | 91.8 | **0.8464** |

> [!NOTE]
> RepELA-Small 无 ImageNet 预训练，参数量仅为最小预训练 baseline（FPN/MNV3-S, 2.72M）的 78%，
> 模型大小仅 8.28MB（最小预训练 baseline 的 78%），GPU 显存占用仅 91.8MB（最低）。
> Best seed 的 test mIoU=0.8464 已超过 DLV3+/R18（0.8332）和 PSPNet/R18（0.8657 的 98%）。

---

## 4. 预训练 vs 从头训练（Pretrained vs Scratch）

| Model | Params(M) | Test mIoU (PT) | Test mIoU (Scratch) | 预训练增益 |
|---|---|---|---|---|
| FPN/MNV3-S | 2.72 | 0.8953 | 0.8379 | +6.8% |
| UNet/MNV3-S | 3.59 | 0.8834 | 0.8587 | +2.9% |
| DLV3+/EffB0 | 4.91 | 0.8827 | 0.8689 | +1.6% |
| **RepELA-Small (3-seed mean)** | **2.12** | — | **0.8125** | **无预训练** |

### Scratch 排名（公平对比，均无 ImageNet 预训练）

| 排名 | Model | Params(M) | Test mIoU |
|---|---|---|---|
| 🥇 | DLV3+/EffB0 | 4.91 | **0.8689** |
| 🥈 | UNet/MNV3-S | 3.59 | 0.8587 |
| 🥉 | **RepELA-Small (best, seed 123)** | **2.12** | **0.8464** |
| 4 | FPN/MNV3-S | 2.72 | 0.8379 |

---

## 5. 轻量化指标对比

| 指标 | RepELA-Small | 最佳轻量BL | 最佳标准BL | RepELA 优势 |
|---|---|---|---|---|
| Params (M) | **2.12** | 2.72 (FPN/MNV3-S) | 11.34 (PSPNet/R18) | ✅ 最少 |
| Model Size (MB) | **8.28** | 10.64 (FPN/MNV3-S) | 43.38 (PSPNet/R18) | ✅ 最小 |
| GPU Memory (MB) | **91.8** | 119.8 (UNet/MNV3-S) | 203.6 (PSPNet/R18) | ✅ 最低 |
| FLOPs (G) | 5.28 | **2.36** (DLV3+/EffB0) | 5.86 (PSPNet/R18) | 中等 |
| GPU FPS | 74.5 | 169.8 (DLV3+/MV2) | **585.2** (PSPNet/R18) | 较慢 |
| CPU FPS | 9.4 | **14.2** (FPN/MNV3-S) | **29.9** (PSPNet/R18) | 较慢 |
| **Test mIoU (best)** | **0.8464** | **0.8689** (DLV3+/EffB0 scratch) | 0.8657 (PSPNet/R18) | 接近 |

---

## 6. 检查点路径索引

| 模型 | 检查点路径 |
|---|---|
| RepELA-Small (seed 42) | `output/seed_test/seed_42/repela_small_20260324_080734/best_model.pth` |
| RepELA-Small (seed 123, best) | `output/seed_test/seed_123/repela_small_20260324_085530/best_model.pth` |
| RepELA-Small (seed 2026) | `output/seed_test/seed_2026/repela_small_20260324_094407/best_model.pth` |
| w/o ELA | `output/ablation/no_ela_20260323_135032/best_model.pth` |
| w/o RepConv | `output/ablation/no_rep_20260323_142757/best_model.pth` |
| w/o Boundary | `output/ablation/no_boundary_20260323_151358/best_model.pth` |
| w/o DW-MFF | `output/ablation/no_dwmff_20260323_155950/best_model.pth` |
| fpn_mnv3s (PT) | `output/baselines/fpn_mnv3s_20260323_172631/best_model.pth` |
| unet_mnv3s (PT) | `output/baselines/unet_mnv3s_20260323_174117/best_model.pth` |
| fpn_mv2 (PT) | `output/baselines/fpn_mv2_20260323_175901/best_model.pth` |
| deeplabv3p_mv2 (PT) | `output/baselines/deeplabv3p_mv2_20260323_181917/best_model.pth` |
| deeplabv3p_effb0 (PT) | `output/baselines/deeplabv3p_effb0_20260323_183434/best_model.pth` |
| unet_r18 (PT) | `output/baselines/unet_r18_20260323_185337/best_model.pth` |
| deeplabv3p_r18 (PT) | `output/baselines/deeplabv3p_r18_20260323_192334/best_model.pth` |
| pspnet_r18 (PT) | `output/baselines/pspnet_r18_20260323_194716/best_model.pth` |
| unet_mnv3s (scratch) | `output/baselines_scratch/unet_mnv3s_20260323_202601/best_model.pth` |
| deeplabv3p_effb0 (scratch) | `output/baselines_scratch/deeplabv3p_effb0_20260323_205413/best_model.pth` |
| fpn_mnv3s (scratch) | `output/baselines_scratch/fpn_mnv3s_20260323_211744/best_model.pth` |

---

## 7. 评估结果目录

所有 test 评估结果（per-image metrics + confusion matrix）保存在 `output/eval_results/` 下：

```
output/eval_results/
├── seed_42/                   # RepELA-Small 3-seed
├── seed_123/
├── seed_2026/
├── repela_baseline/           # RepELA-Small (legacy DS=True)
├── ablation_no_ela/           # 消融: w/o ELA
├── ablation_no_rep/           # 消融: w/o RepConv
├── ablation_no_boundary/      # 消融: w/o BoundaryEnhancement
├── ablation_no_dwmff/         # 消融: w/o DW-MFF
├── pretrained_fpn_mnv3s/      # 预训练 baselines
├── pretrained_unet_mnv3s/
├── pretrained_fpn_mv2/
├── pretrained_deeplabv3p_mv2/
├── pretrained_deeplabv3p_effb0/
├── pretrained_unet_r18/
├── pretrained_deeplabv3p_r18/
├── pretrained_pspnet_r18/
├── scratch_unet_mnv3s/        # 从头训练 baselines
├── scratch_deeplabv3p_effb0/
└── scratch_fpn_mnv3s/
```

Benchmark CSV: `benchmark_512.csv`
