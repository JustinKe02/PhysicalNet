# Transfer Learning Three-Line Design

> Date: 2026-03-27  
> Project: RepELA-Net  
> Scope: external public benchmark, internal supplementary benchmark, and combined-target benchmark

---

## 1. Goal

在补充数据加入后，迁移学习实验不应再只有一条线，而应拆成三条互相独立的实验线：

1. `External Benchmark`
2. `Internal Supplementary`
3. `Combined Target`

这样做的目的不是增加实验数量，而是保证结论可解释：

- 公开数据集的迁移效果如何
- 自有补充数据上的迁移效果如何
- 当目标域数据增多时，迁移是否进一步改善

---

## 2. Why Three Lines

当前项目里目标域数据分成两类：

- [other data](/root/autodl-tmp/PhysicalNet/other%20data)：来自公开文献或开源数据
- [supplementary_data](/root/autodl-tmp/PhysicalNet/supplementary_data)：来自自建补充数据

这两类数据在以下方面可能不同：

- 成像条件
- 样本筛选标准
- 标注风格
- 数据清洗标准
- 域分布难度

如果一开始就把它们混在一起做迁移训练，后面很难判断：

- 提升来自更多样本，还是来自更接近目标域的自有数据
- 迁移成功是公开数据驱动，还是内部补充数据驱动
- 负迁移是模型问题，还是数据混合导致的口径污染

因此，三条线必须分开。

---

## 3. Available Data

### 3.1 Public target datasets

| Dataset | Path | Current Split | Notes |
|---|---|---|---|
| Graphene | [other data/graphene](/root/autodl-tmp/PhysicalNet/other%20data/graphene) | train=30, val=14 | Labels: `BG / 1L / >1L` |
| WS2 | [other data/WS2_data](/root/autodl-tmp/PhysicalNet/other%20data/WS2_data) | train=24, val=6, test=15 | Labels: `BG / 1L / FL / ML` |

### 3.2 Internal supplementary target datasets

| Dataset | Path | Count | Labels |
|---|---|---:|---|
| Supplementary Gr | [supplementary_data/Gr](/root/autodl-tmp/PhysicalNet/supplementary_data/Gr) | 77 | `BG / FL / ML` |
| Supplementary MoS2 | [supplementary_data/MoS2](/root/autodl-tmp/PhysicalNet/supplementary_data/MoS2) | 105 | `BG / 1L / FL / ML` |
| Supplementary WS2 | [supplementary_data/WS2](/root/autodl-tmp/PhysicalNet/supplementary_data/WS2) | 80 | `BG / 1L / FL / ML` |

### 3.3 Important label-space note

旧 Graphene 与 supplementary Gr 不能直接合并：

- old Graphene: `BG / 1L / >1L`
- supplementary Gr: `BG / FL / ML`

其中：

- `1L` 不是 `FL`
- `>1L` 也不是纯 `ML`

因此，Graphene 必须拆成两条不同目标域，不能直接视为同一数据集。

---

## 4. Line 1: External Benchmark

### 4.1 Purpose

这条线用于论文主迁移结果，保证与公开文献口径尽可能接近。

### 4.2 Datasets

- [other data/graphene](/root/autodl-tmp/PhysicalNet/other%20data/graphene)
- [other data/WS2_data](/root/autodl-tmp/PhysicalNet/other%20data/WS2_data)

### 4.3 Recommended experiments

| Dataset | Strategy A | Strategy B | Strategy C |
|---|---|---|---|
| Graphene | Scratch | Finetune-ResetHead | Finetune-PartialClassMap |
| WS2 | Scratch | Finetune-ResetHead | Finetune-KeepHead |

### 4.4 Main question

在公开目标域上，MoS2 预训练 encoder 是否具有迁移价值？

### 4.5 Recommended use in paper

- 这条线作为迁移学习主结果
- WS2 适合作为正迁移案例
- Graphene 适合作为负迁移案例

---

## 5. Line 2: Internal Supplementary

### 5.1 Purpose

这条线用于验证模型在自建补充数据上的迁移能力，回答：

- 自有数据分布是否更适合当前模型
- 与公开数据相比，迁移是否更容易成功
- supplement 数据是否值得长期纳入目标域实验

### 5.2 Datasets

- [supplementary_data/WS2](/root/autodl-tmp/PhysicalNet/supplementary_data/WS2)
- [supplementary_data/Gr](/root/autodl-tmp/PhysicalNet/supplementary_data/Gr)

### 5.3 Directory recommendation

建议整理成：

```text
supplementary_prepared/
├── WS2_supp/
│   ├── img_dir/{train,val,test}
│   └── ann_dir/{train,val,test}
└── Gr_supp/
    ├── img_dir/{train,val,test}
    └── ann_dir/{train,val,test}
```

图像建议从 `ori_jpg/` 复制，mask 从 `mask/*.png` 复制。

### 5.4 Recommended experiments

| Dataset | Strategy A | Strategy B | Strategy C |
|---|---|---|---|
| WS2_supp | Scratch | Finetune-ResetHead | Finetune-KeepHead |
| Gr_supp | Scratch | Finetune-ResetHead | Finetune-PartialClassMap |

### 5.5 Main question

在自有补充数据上，迁移是否比公开数据更容易成功？

### 5.6 Recommended use in paper

- 可作为补充实验
- 若 WS2_supp 成功，可增强“迁移有效”的证据
- 若 Gr_supp 仍困难，可增强“域差主导负迁移”的论证

---

## 6. Line 3: Combined Target

### 6.1 Purpose

这条线用于回答一个更实际的问题：

> 当目标域数据总量增大时，迁移效果是否进一步提升？

这条线不是主结果，而是扩展实验。

### 6.2 Allowed combined datasets

| Combined Dataset | Allowed? | Reason |
|---|---|---|
| `WS2_data + supplementary WS2` | Yes | 标签空间一致，材料域一致 |
| `graphene + supplementary Gr` | No | 标签空间不一致，不能直接混 |

### 6.3 Recommended experiments

| Dataset | Strategy A | Strategy B |
|---|---|---|
| WS2_combined | Scratch | Finetune-ResetHead |

### 6.4 Main question

在标签空间一致的前提下，更多目标域数据是否进一步提高迁移性能？

### 6.5 Recommended use in paper

- 可放补充材料
- 适合作为“target data scaling”结论
- 不建议在第一轮就把它作为主结果

---

## 7. Source Model Note

三条线虽然都属于迁移学习，但共享同一个源模型问题：

- 是否继续使用当前最佳 source checkpoint
- 是否用 supplementary MoS2 重训 source model

建议把 source augmentation 作为独立前置实验，不直接混在三条 target-line 里。

### Recommended source strategy

1. 先保留现有 source baseline 不动
2. 再单独建立 `MoS2_augmented`
3. 只在 source 明显变强时，才把新 source 用到三条线中

这能避免“source 变了、target 也变了”导致结论混乱。

---

## 8. Recommended Execution Order

### Stage 1

先跑 `External Benchmark`：

- 公开数据最适合做论文主结果
- 先确认正迁移和负迁移的公开口径

### Stage 2

再跑 `Internal Supplementary`：

- 检查自有数据是否更贴近模型分布
- 判断 supplementary 数据的独立价值

### Stage 3

最后再跑 `Combined Target`：

- 只作为扩展实验
- 不作为第一批主结论

---

## 9. Concrete Experiment Table

| Line | Dataset | Strategy | Recommended Priority |
|---|---|---|---|
| External | old WS2 | Scratch | High |
| External | old WS2 | Finetune-ResetHead | High |
| External | old WS2 | Finetune-KeepHead | Medium |
| External | old Graphene | Scratch | High |
| External | old Graphene | Finetune-ResetHead | High |
| External | old Graphene | Finetune-PartialClassMap | Medium |
| Internal | WS2_supp | Scratch | Medium |
| Internal | WS2_supp | Finetune-ResetHead | Medium |
| Internal | WS2_supp | Finetune-KeepHead | Low |
| Internal | Gr_supp | Scratch | Medium |
| Internal | Gr_supp | Finetune-ResetHead | Medium |
| Internal | Gr_supp | Finetune-PartialClassMap | Medium |
| Combined | WS2_combined | Scratch | Low |
| Combined | WS2_combined | Finetune-ResetHead | Low |

---

## 10. What Not To Do

以下做法当前不建议：

1. 不要直接把 supplementary 数据混入现有公开目标域后立即重跑全部实验
2. 不要把 old Graphene 和 supplementary Gr 直接合并
3. 不要在三条线还没跑清楚时就重跑所有 decoder comparison
4. 不要用 combined 结果替代 external benchmark 主结果

---

## 11. Final Recommendation

### Main line for the paper

- `External Benchmark`

### Strong supporting line

- `Internal Supplementary`

### Optional extension

- `Combined Target`

### Practical summary

- 先用公开数据给出主迁移结论
- 再用自有数据验证迁移是否具有工程可用性
- 最后用 combined 数据回答“更多目标域样本是否进一步改善”

---

## 12. One-Sentence Summary

三条线的核心作用分别是：

- `External Benchmark`：论文主迁移结果
- `Internal Supplementary`：自有数据上的独立迁移验证
- `Combined Target`：目标域数据规模扩展实验

