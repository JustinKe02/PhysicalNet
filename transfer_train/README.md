# transfer_train 说明

> 用途：集中保存**需要放进论文主文或主附录的迁移学习主线材料**，方便后续写作和查阅。  
> 当前只收纳公开/主线迁移，不收纳 supplementary、自建筛选、source augmentation、CV 稳定性检查。

---

## 1. 当前收纳范围

只保留这 3 条主线：

1. `WS2 public`
2. `Graphene public`
3. `other_datav2`（同材料外部对照）

每条主线都尽量保留：
- 训练日志
- 结果文件
- 正文会用到的关键图

---

## 2. 目录结构

### `ws2_public/`

保存 `MoS2 -> WS2` 主线结果：

- `results/`
  - `scratch_results.txt`
  - `scratch_test_results.txt`
  - `scratch_finetune.log`
  - `ft_resethead_results.txt`
  - `ft_resethead_test_results.txt`
  - `ft_resethead_finetune.log`
  - `ft_keephead_results.txt`
  - `ft_keephead_test_results.txt`
  - `ft_keephead_finetune.log`

- `figures/`
  - `WS2_inference_grid.png`
  - `WS2_confusion_matrices.png`
  - `curve_transfer_ws2_scratch.png`
  - `curve_transfer_ws2_ft_resethead.png`

说明：
- 正文主结论现在以 official `test` 为准，主要看 `Scratch` 和 `FT+reset_head`
- `keep_head` 保留作补充比较

---

### `graphene_public/`

保存 `MoS2 -> Graphene` 主线结果：

- `results/`
  - `scratch_results.txt`
  - `scratch_finetune.log`
  - `ft_partial_results.txt`
  - `ft_partial_finetune.log`
  - `ft_fullLR_resethead_results.txt`
  - `ft_fullLR_resethead_finetune.log`
  - `ft_fullLR_mapped_results.txt`
  - `ft_fullLR_mapped_finetune.log`

- `figures/`
  - `GRAPHENE_inference_grid.png`
  - `GRAPHENE_inference_grid_full.png`
  - `GRAPHENE_confusion_matrices.png`
  - `curve_transfer_graphene_scratch.png`
  - `curve_transfer_graphene_ft_partial.png`

说明：
- 正文主结论主要看 `Scratch` 与 `FT partial(stage1+2)`
- 其余策略保留作对照

---

### `other_datav2/`

保存 `MoS2 -> other_datav2` 主线结果：

- `results/`
  - `scratch_val_results.txt`
  - `scratch_test_results.txt`
  - `scratch_finetune.log`
  - `ft_resethead_val_results.txt`
  - `ft_resethead_test_results.txt`
  - `ft_resethead_finetune.log`
  - `ft_keephead_val_results.txt`
  - `ft_keephead_test_results.txt`
  - `ft_keephead_finetune.log`

- `figures/`
  - `MOS2V2_inference_grid.png`
  - `MOS2V2_confusion_matrices.png`
  - `curve_transfer_mos2v2_scratch.png`
  - `curve_transfer_mos2v2_ft_resethead.png`

- `splits/`
  - `train.txt`
  - `val.txt`
  - `test.txt`

说明：
- 正文主要看 official `test=7`
- `val_results` 一并保留，便于后续核对训练过程

---

## 3. 当前明确不收纳的内容

以下内容**不放进这个目录**：

1. `WS2_supp / Gr_supp / WS2_combined`
2. supplementary 数据筛选前后 v1/v2
3. `MoS2` source augmentation
4. `other_datav2` 的 3-fold CV
5. supplementary 数据筛选、删除样本、分布漂移分析细节

这些内容如果后面要写附录或 discussion，再单独从原始目录取。

---

## 4. 口径说明

### 迁移实验 source checkpoint

迁移实验统一使用：

- `output/seed_test/seed_42/repela_small_20260324_080734/best_model.pth`

原因：
- 所有迁移实验固定使用同一个 source checkpoint，保证不同目标域间可比

### 正文推荐写法

- `WS2 public`: `Scratch` vs `FT+reset_head`
- `Graphene public`: `Scratch` vs `FT partial(stage1+2)`
- `other_datav2`: `Scratch` vs `FT+reset_head`

---

## 5. 使用建议

如果后面只是写论文迁移学习部分，优先直接看：

1. `ws2_public/results/`
2. `graphene_public/results/`
3. `other_datav2/results/`
4. 各子目录下的 `figures/`

这样可以避免再去主 `output/` 目录里到处找文件。
