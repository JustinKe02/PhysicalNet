# 实验结果来源关系说明

这个文档用来说明当前项目里各条实验线的来源关系，重点是：

- 哪些实验是从头训练
- 哪些实验加载了已有 checkpoint
- `seed_42`、`seed_123` 分别在什么地方被使用

方便后面写论文、查图、核对口径时不混乱。

## 1. 总览表

| 实验线 | 目的 | 入口脚本 | 是否加载已有 checkpoint | 实际来源 | 主要输出目录 |
|---|---|---|---|---|---|
| 主模型 3-seed | MoS2 主 benchmark | `seed_test` 相关训练脚本 | 否 | 3 条独立从头训练 | `output/seed_test/` |
| `baseline_unified` | 统一训练框架下的对照基线 | `tools/train.py` | 否 | 独立从头训练 | `output/baseline_unified/` |
| `baseline_oldcfg` | 旧配置复刻基线 | `tools/train_oldcfg.py` | 否 | 独立从头训练 | 历史目录，现已清理 |
| 消融实验 | 去掉不同模块看影响 | `tools/train_ablation.py` | 否 | 每个消融版本都独立从头训练 | `output/ablation/` |
| decoder compare | 比较不同 decoder | `tools/train_decoder_compare.py` | 否 | 每个 decoder 都独立从头训练 | `output/decoder_compare/` |
| 迁移学习 | `MoS2 -> WS2 / Graphene / MoS2v2 / supplementary / combined` | `scripts/run_transfer*.sh`、`scripts/run_phase*.sh`、`scripts/run_mos2v2*.sh` | 是 | 固定加载 `seed_42` 作为 source checkpoint | 各 `output/finetune_*` |
| source augmentation | `Mos2_data + supplementary MoS2` | 对应 augmented 训练脚本 | 否 | 在增广后的 source 数据上独立从头训练 | `output/mos2_augmented_seed42/` |

## 2. 主模型 3-seed

当前主模型正式 3-seed 在这里：

- [seed_42](/root/autodl-tmp/PhysicalNet/output/seed_test/seed_42/repela_small_20260324_080734)
- [seed_123](/root/autodl-tmp/PhysicalNet/output/seed_test/seed_123/repela_small_20260324_085530)
- [seed_2026](/root/autodl-tmp/PhysicalNet/output/seed_test/seed_2026/repela_small_20260324_094407)

推荐理解方式：

- `seed_123`：主模型单次最优 test 结果
- `3-seed mean ± std`：主模型稳定性结果

这里要特别注意：

- `seed_42` 的 **val 更高**
- `seed_123` 的 **test 更高**

所以：

- 如果论文主模型只报单次最好结果，应该优先用 `seed_123`
- 如果做迁移 source，`seed_42` 也能解释得通，因为它更像按 source validation 选出的 canonical checkpoint

## 3. `baseline_unified`

目录：

- [baseline_unified](/root/autodl-tmp/PhysicalNet/output/baseline_unified/repela_small_20260323_222221)

作用：

- 这是统一训练框架下的 baseline
- 主要是为了和统一框架下的消融、decoder compare 做公平对比

特点：

- 从头训练
- **不是**从 `seed_42` 或 `seed_123` 微调出来的

## 4. `baseline_oldcfg`

说明：

- 这条分支原本用于复刻旧配置
- 当前目录已清理，不再作为主线结果来源保留

作用：

- 尽量复刻之前旧实验配置
- 用来和历史结果接轨

特点：

- 从头训练
- **不是**从 `seed_42` 或 `seed_123` 微调出来的

## 5. 消融实验

训练目录：

- [no_rep](/root/autodl-tmp/PhysicalNet/output/ablation/no_rep_20260323_142757)
- [no_ela](/root/autodl-tmp/PhysicalNet/output/ablation/no_ela_20260323_135032)
- [no_boundary](/root/autodl-tmp/PhysicalNet/output/ablation/no_boundary_20260323_151358)
- [no_dwmff](/root/autodl-tmp/PhysicalNet/output/ablation/no_dwmff_20260323_155950)

正式评估目录：

- [ablation_no_rep](/root/autodl-tmp/PhysicalNet/output/eval_results/ablation_no_rep)
- [ablation_no_ela](/root/autodl-tmp/PhysicalNet/output/eval_results/ablation_no_ela)
- [ablation_no_boundary](/root/autodl-tmp/PhysicalNet/output/eval_results/ablation_no_boundary)
- [ablation_no_dwmff](/root/autodl-tmp/PhysicalNet/output/eval_results/ablation_no_dwmff)

关键结论：

- 消融实验**不是**从 `seed_42` 或 `seed_123` 继续训练出来的
- 每一条 `w/o XXX` 都是独立从头训练

当前正文可视化口径：

- 消融定性图里的 `Ours` 已经切到 `seed_123`
- 消融列则来自各自的正式评估结果 `output/eval_results/ablation_*`

也就是说：

- 定性图是为了和论文最终主模型口径统一
- 数值表仍然来自消融各自独立训练的正式 test 结果

## 6. decoder compare

目录：

- [ours](/root/autodl-tmp/PhysicalNet/output/decoder_compare/ours_20260325_072211)
- [fpn](/root/autodl-tmp/PhysicalNet/output/decoder_compare/fpn_20260324_234528)
- [ppm](/root/autodl-tmp/PhysicalNet/output/decoder_compare/ppm_20260325_012110)
- [segformer](/root/autodl-tmp/PhysicalNet/output/decoder_compare/segformer_20260325_005945)

关键结论：

- decoder compare **不属于迁移**
- 也**没有**加载 `seed_42` 或 `seed_123`
- 每个 decoder 版本都是独立从头训练

所以 decoder compare 里的：

- `Ours`

指的是：

- `output/decoder_compare/ours_*`

不是：

- `seed_42`
- `seed_123`

## 7. 迁移学习

当前所有迁移脚本都固定用了同一个 source checkpoint：

- [seed_42 best_model.pth](/root/autodl-tmp/PhysicalNet/output/seed_test/seed_42/repela_small_20260324_080734/best_model.pth)

涉及脚本包括：

- [run_transfer.sh](/root/autodl-tmp/PhysicalNet/scripts/run_transfer.sh)
- [run_transfer_r2.sh](/root/autodl-tmp/PhysicalNet/scripts/run_transfer_r2.sh)
- [run_transfer_r3.sh](/root/autodl-tmp/PhysicalNet/scripts/run_transfer_r3.sh)
- [run_transfer_line2.sh](/root/autodl-tmp/PhysicalNet/scripts/run_transfer_line2.sh)
- [run_transfer_line3.sh](/root/autodl-tmp/PhysicalNet/scripts/run_transfer_line3.sh)
- [run_phase_b.sh](/root/autodl-tmp/PhysicalNet/scripts/run_phase_b.sh)
- [run_phase_c.sh](/root/autodl-tmp/PhysicalNet/scripts/run_phase_c.sh)
- [run_mos2v2_transfer.sh](/root/autodl-tmp/PhysicalNet/scripts/run_mos2v2_transfer.sh)
- [run_mos2v2_cv.sh](/root/autodl-tmp/PhysicalNet/scripts/run_mos2v2_cv.sh)

这意味着：

- `WS2`
- `Graphene`
- `supplementary`
- `combined`
- `MoS2v2`

这些迁移结果彼此可比，因为 source 是同一个。

但要注意：

- 这里的 source 不是全局 test 最优的 `seed_123`
- 而是固定选定的 `seed_42`

所以论文里更稳的表述是：

- 迁移实验统一使用固定 source checkpoint（`seed_42`），以保证不同目标域间的公平比较

## 8. source augmentation

目录：

- [mos2_augmented_seed42](/root/autodl-tmp/PhysicalNet/output/mos2_augmented_seed42/repela_small_20260327_141546)

作用：

- 测试 supplementary MoS2 直接并入 source train 是否有帮助

特点：

- 也是独立从头训练
- 不是从 `seed_42` 或 `seed_123` 继续训出来的

## 9. 现在写论文时最稳的口径

### 主模型

- 单次最优结果：`seed_123`
- 稳定性：`3-seed mean ± std`

### 迁移学习

- 统一说明：全部 target domain 都使用固定 `seed_42` source checkpoint

### 消融

- 数值来自各自独立训练
- 正文定性图里 `Ours` 可以用 `seed_123` 对齐主模型口径

### decoder compare

- 明确说明是独立训练的 decoder 对比
- 不属于迁移，也不是从 seed checkpoint 微调

## 10. 一句话总结

- `seed_123`：主模型最终最适合展示的单次结果
- `seed_42`：迁移学习统一固定使用的 source checkpoint
- 消融和 decoder compare：都是各自独立从头训练，不是从这两个 seed 继续训练出来的
