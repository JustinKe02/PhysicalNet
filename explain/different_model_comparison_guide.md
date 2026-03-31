# 不同模型对比整理清单

> 用途：给自己看，后续写论文时快速确认“哪些对比该放正文，哪些放附录”，以及每条对比对应的数据来源和图。

---

## 1. 总体结构

当前项目里的“不同模型对比”建议分成 4 条线：

1. 主模型 vs 常见分割 baseline
2. 固定 encoder 的 decoder 对比
3. 主模型内部消融对比
4. 迁移学习策略对比

这 4 条线不要混写。

---

## 2. 主模型 vs baseline

### 2.1 这条线回答什么

- RepELA-Net 相比常见分割网络，整体精度如何
- 参数量、FLOPs、显存、速度是否有优势
- 在 scratch-only 条件下是否仍有竞争力

### 2.2 正文建议使用的口径

- 主模型代表结果：`seed_123`
- 主模型数值：
  - `mIoU = 84.64`
  - 来源：`output/eval_results/seed_123/test_metrics.txt`

### 2.3 对比对象

- UNet / ResNet-18
- DeepLabV3+ / ResNet-18
- PSPNet / ResNet-18
- DeepLabV3+ / EfficientNet-B0
- FPN / MobileNetV2
- DeepLabV3+ / MobileNetV2
- UNet / MobileNetV3-S
- FPN / MobileNetV3-S
- Ours (RepELA-Net)

### 2.4 主要文档

- `output/model_comparison.md`
- `output/experiment_results.md`

### 2.5 适合正文的图和表

- 表：`output/model_comparison.md` 第 1 节、第 2 节
- 训练曲线：`output/paper_figures/curve_mos2_baseline.png`
- 混淆矩阵：`output/paper_figures/cm_mos2_baseline.png`
- 主模型定性图：`output/feature_visualization/seed_123_results/inference_grid.png`

### 2.6 写作建议

- 正文主模型统一用 `seed_123`
- 3-seed 结果放正文或附录都可以，但不能和 `seed_42` 混成一个主口径

---

## 3. Decoder 对比

### 3.1 这条线回答什么

- 在固定 RepELA-Small encoder 的前提下，不同 decoder 的精度与效率差异
- Ours 的 decoder 是否在轻量化和精度之间取得平衡

### 3.2 当前正式 test 结果

- FPN = `91.52`
- ASPP = `88.77`
- SegFormer = `88.17`
- PPM = `87.46`
- Ours = `87.45`
- Hamburger = `87.13`
- UNet = `86.61`

来源：

- `output/eval_results/decoder_unet/test_metrics.txt`
- `output/eval_results/decoder_fpn/test_metrics.txt`
- `output/eval_results/decoder_aspp/test_metrics.txt`
- `output/eval_results/decoder_segformer/test_metrics.txt`
- `output/eval_results/decoder_ppm/test_metrics.txt`
- `output/eval_results/decoder_hamburger/test_metrics.txt`
- `output/eval_results/decoder_ours/test_metrics.txt`

### 3.3 适合正文的图

- 指标柱状图：`output/paper_figures/decoder_metrics_test.png`
- 效率散点图：`output/paper_figures/decoder_params_vs_miou.png`
- 定性图：`output/transfer_vis/decoder_qualitative.png`

### 3.4 适合附录的图

- 混淆矩阵：
  - `output/paper_figures/cm_decoder_unet.png`
  - `output/paper_figures/cm_decoder_fpn.png`
  - `output/paper_figures/cm_decoder_aspp.png`
  - `output/paper_figures/cm_decoder_segformer.png`
  - `output/paper_figures/cm_decoder_ppm.png`
  - `output/paper_figures/cm_decoder_hamburger.png`
  - `output/paper_figures/cm_decoder_ours.png`

- 训练曲线：
  - `output/paper_figures/curve_decoder_unet.png`
  - `output/paper_figures/curve_decoder_fpn.png`
  - `output/paper_figures/curve_decoder_aspp.png`
  - `output/paper_figures/curve_decoder_segformer.png`
  - `output/paper_figures/curve_decoder_ppm.png`
  - `output/paper_figures/curve_decoder_hamburger.png`
  - `output/paper_figures/curve_decoder_ours.png`

### 3.5 写作建议

- 正文强调：
  - FPN 精度最高，但计算量最大
  - SegFormer 最轻
  - Ours 与 PPM 基本持平，但 decoder 参数量更小

- 不建议在正文里把 7 个 decoder 的定性图全部展开
- 正文只保留代表性 4 个：`FPN / PPM / SegFormer / Ours`

---

## 4. 消融对比

### 4.1 这条线回答什么

- RepConv / ELA / DW-MFF / BoundaryEnhancement 分别带来了什么贡献

### 4.2 当前正文统一口径

- Ours 使用 `seed_123`
- 消融使用 official `test`

当前结果：

- Ours (`seed_123`) = `84.64`
- w/o RepConv = `83.42`
- w/o BoundaryEnhancement = `82.96`
- w/o DW-MFF = `81.60`
- w/o ELA = `77.33`

来源：

- `output/eval_results/seed_123/test_metrics.txt`
- `output/eval_results/ablation_no_rep/test_metrics.txt`
- `output/eval_results/ablation_no_boundary/test_metrics.txt`
- `output/eval_results/ablation_no_dwmff/test_metrics.txt`
- `output/eval_results/ablation_no_ela/test_metrics.txt`

### 4.3 正文建议使用的图

- 指标图：`output/paper_figures/ablation_metrics_seed123.png`
- 定性图：`output/transfer_vis/ablation_qualitative.png`

### 4.4 附录图

- 混淆矩阵：
  - `output/paper_figures/cm_ablation_no_rep.png`
  - `output/paper_figures/cm_ablation_no_boundary.png`
  - `output/paper_figures/cm_ablation_no_dwmff.png`
  - `output/paper_figures/cm_ablation_no_ela.png`

- 训练曲线：
  - `output/paper_figures/curve_ablation_no_rep.png`
  - `output/paper_figures/curve_ablation_no_boundary.png`
  - `output/paper_figures/curve_ablation_no_dwmff.png`
  - `output/paper_figures/curve_ablation_no_ela.png`

### 4.5 写作建议

- 只写“完整模型在整体 test mIoU 上优于各消融版本”
- 不要写成“所有单类指标都全面领先”
- 历史 `val` 消融记录仅作内部备查，不再作为正文依据

---

## 5. 迁移学习对比

### 5.1 这条线回答什么

- 源域 MoS2 训练好的 encoder 对不同目标域是否有迁移收益
- 不同目标域的最佳迁移策略是否一致

### 5.2 当前主线结果

- WS2 (public):
  - Scratch = `69.28` (official test)
  - FT+reset_head = `90.17` (official test)

- Graphene (public):
  - Scratch = `60.67`
  - FT partial(stage1+2) = `68.43`

- other_datav2 (public MoS2):
  - Scratch = `60.22`
  - FT+reset_head = `77.35`

### 5.3 当前统一口径

- 所有迁移实验固定 source checkpoint = `seed_42`
- 这是为了不同目标域间保持公平比较

### 5.4 正文建议使用的图

- `output/transfer_vis/WS2_inference_grid.png`
- `output/transfer_vis/GRAPHENE_inference_grid.png`
- `output/transfer_vis/MOS2V2_inference_grid.png`

### 5.5 附录建议

- WS2 / Graphene / MoS2v2 的混淆矩阵与训练曲线
- supplementary 与 filtering 结果
- source augmentation 的负结果

---

## 6. 最终论文排法建议

### 正文

1. 主模型 vs baseline 总表
2. 主模型训练曲线 + 混淆矩阵 + 定性图
3. 消融指标图 + 消融定性图
4. decoder 指标图 + decoder 效率散点图 + decoder 定性图
5. 迁移学习三条主线结果

### 附录

1. decoder 全部混淆矩阵
2. decoder 全部训练曲线
3. 消融全部混淆矩阵
4. 消融全部训练曲线
5. supplementary / filtered / combined 结果
6. MoS2v2 CV 稳定性检查

---

## 7. 当前最关键的统一原则

1. 主模型正文统一用 `seed_123`
2. 消融正文统一用 `seed_123 + official test`
3. 迁移学习继续固定 `seed_42` 作为 source checkpoint
4. decoder 对比是独立训练，不属于 seed_42/123 迁移链

只要守住这 4 条，论文口径就不会乱。
