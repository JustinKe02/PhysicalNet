"""
消融实验柱状图 — SCI 论文配色
Baseline = w/o CSE (no_color)，对比去掉各个模块的效果。
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

# ── SCI 配色方案 (来自 Nature / IEEE 常用色板) ─────────────────────────────────
# 柔和、高对比、灰度打印友好
SCI_BLUE    = '#4472C4'   # 主色
SCI_ORANGE  = '#ED7D31'
SCI_GREEN   = '#70AD47'
SCI_PURPLE  = '#7030A0'
SCI_GOLD    = '#FFC000'
SCI_GRAY    = '#A5A5A5'
SCI_RED     = '#C00000'

# Per-class colors (偏柔和的学术风格)
CLS_BG      = '#8DB4E2'   # 浅蓝
CLS_MONO    = '#E6855E'   # 砖红
CLS_FEW     = '#76B387'   # 草绿
CLS_MULTI   = '#9B8EC0'   # 薰衣草紫

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.linewidth': 1.0,
    'axes.edgecolor': '#333333',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

# ── 数据 ──────────────────────────────────────────────────────────────────────
ablations = [
    'Baseline\n(w/o CSE)',
    'w/o ELA',
    'w/o RepConv',
    'w/o DWMFF',
    'w/o Boundary',
]

miou = [0.9161, 0.8724, 0.8952, 0.8613, 0.8841]
is_real = [True, False, False, False, False]

bg_iou   = [0.9915, 0.9880, 0.9901, 0.9865, 0.9895]
mono_iou = [0.9002, 0.8410, 0.8820, 0.8350, 0.8690]
few_iou  = [0.8070, 0.7100, 0.7620, 0.6700, 0.7350]
multi_iou= [0.9658, 0.9505, 0.9468, 0.9538, 0.9430]

colors_bar = [SCI_BLUE, SCI_ORANGE, SCI_GREEN, SCI_PURPLE, SCI_GOLD]

# ── Figure 1: mIoU 柱状图 ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))

x = np.arange(len(ablations))
bars = ax.bar(x, miou, width=0.52, color=colors_bar, edgecolor='#333333',
              linewidth=0.8, zorder=3)

# 数值标签
for i, (bar, val, real) in enumerate(zip(bars, miou, is_real)):
    label = f'{val:.4f}'
    if not real:
        label += '*'
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            label, ha='center', va='bottom', fontsize=11, fontweight='bold',
            color='#333333')

# Baseline 参考线
ax.axhline(y=miou[0], color=SCI_BLUE, linestyle='--', linewidth=1.0, alpha=0.5,
           label=f'Baseline mIoU = {miou[0]:.4f}')

# 差值标注
for i in range(1, len(ablations)):
    delta = miou[i] - miou[0]
    ax.annotate(f'{delta:+.4f}', xy=(i, miou[i] - 0.004),
                ha='center', va='top', fontsize=10, color=SCI_RED, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(ablations, fontsize=11, fontweight='bold')
ax.set_ylabel('Test mIoU', fontsize=13, fontweight='bold')
ax.set_title('Ablation Study on RepELA-Net-Small', fontsize=14, fontweight='bold', pad=10)
ax.set_ylim(0.83, 0.935)
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
ax.grid(axis='y', alpha=0.25, linewidth=0.6, linestyle='-', color='#999')
ax.legend(fontsize=10, loc='lower right', framealpha=0.9, edgecolor='#ccc')

ax.text(0.98, 0.02, '*Placeholder, pending experiments', transform=ax.transAxes,
        ha='right', va='bottom', fontsize=9, color='#999', style='italic')

plt.tight_layout()
plt.savefig('outputv4_plots/ablation_miou_bar.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: outputv4_plots/ablation_miou_bar.png')


# ── Figure 2: Per-class IoU 分组柱状图 ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5.5))

n_groups = len(ablations)
n_classes = 4
bar_width = 0.17
class_names = ['Background', 'Monolayer', 'Fewlayer', 'Multilayer']
class_colors = [CLS_BG, CLS_MONO, CLS_FEW, CLS_MULTI]
class_data = [bg_iou, mono_iou, few_iou, multi_iou]
hatches = ['', '///', '...', 'xxx']  # 灰度打印区分

x = np.arange(n_groups)

for j, (cls_name, cls_color, cls_vals, hatch) in enumerate(
        zip(class_names, class_colors, class_data, hatches)):
    offset = (j - 1.5) * bar_width
    bars = ax.bar(x + offset, cls_vals, bar_width, label=cls_name,
                  color=cls_color, edgecolor='#444444', linewidth=0.6,
                  zorder=3, hatch=hatch)
    for bar, val in zip(bars, cls_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                f'{val:.2f}', ha='center', va='bottom', fontsize=7,
                color='#333', fontweight='bold', rotation=0)

ax.set_xticks(x)
ax.set_xticklabels(ablations, fontsize=11, fontweight='bold')
ax.set_ylabel('Test IoU', fontsize=13, fontweight='bold')
ax.set_title('Per-class IoU Comparison — Ablation Study', fontsize=14, fontweight='bold', pad=10)
ax.set_ylim(0.62, 1.06)
ax.grid(axis='y', alpha=0.25, linewidth=0.6, linestyle='-', color='#999')
ax.legend(fontsize=10, loc='lower left', ncol=4, framealpha=0.9, edgecolor='#ccc')

ax.text(0.98, 0.02, '*Except Baseline, all values are placeholders', transform=ax.transAxes,
        ha='right', va='bottom', fontsize=8, color='#999', style='italic')

plt.tight_layout()
plt.savefig('outputv4_plots/ablation_perclass_bar.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: outputv4_plots/ablation_perclass_bar.png')

print('\n Done')
