"""
Three-Line Transfer Learning Visualization.
Generates: training curves, bar chart comparison, and confusion matrices.
"""
import os, sys, re, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

OUT_DIR = 'output/transfer_vis'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Helpers ──────────────────────────────────────────────
def parse_finetune_log(log_path):
    """Parse finetune.log for epochwise metrics."""
    epochs, val_mious, train_losses = [], [], []
    with open(log_path) as f:
        for line in f:
            m = re.search(r'Epoch \[(\d+)/\d+\]', line)
            if m: cur_ep = int(m.group(1))
            m = re.search(r'Train Loss:\s*([\d.]+)', line)
            if m: train_losses.append((cur_ep, float(m.group(1))))
            m = re.search(r'Val mIoU:\s*([\d.]+)', line)
            if m:
                epochs.append(cur_ep)
                val_mious.append(float(m.group(1)))
    return epochs, val_mious, train_losses

def parse_train_log(log_path):
    """Parse train.log for the source model."""
    epochs, val_mious = [], []
    with open(log_path) as f:
        for line in f:
            m = re.search(r'Epoch \[(\d+)/\d+\]', line)
            if m: cur_ep = int(m.group(1))
            m = re.search(r'Val.*mIoU=([\d.]+)', line)
            if m:
                epochs.append(cur_ep)
                val_mious.append(float(m.group(1)))
    return epochs, val_mious

# ── 1. Training Curves ──────────────────────────────────
def plot_training_curves():
    """Plot val mIoU curves for all three lines."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Line 1: External
    ax = axes[0]
    ax.set_title('Line 1: External Benchmark', fontweight='bold', fontsize=12)
    for name, label, color, ls in [
        ('finetune_ws2_scratch', 'WS2 Scratch', '#2196F3', '-'),
        ('finetune_ws2_r2_resethead', 'WS2 FT+reset', '#F44336', '-'),
        ('finetune_ws2_r2_keephead', 'WS2 FT+keep', '#FF9800', '--'),
        ('finetune_graphene_scratch', 'Gr Scratch', '#4CAF50', '-'),
        ('finetune_graphene_r3_partial', 'Gr FT partial', '#9C27B0', '-'),
    ]:
        log = f'output/{name}/finetune.log'
        if not os.path.exists(log): continue
        ep, miou, _ = parse_finetune_log(log)
        if ep: ax.plot(ep, miou, color=color, linestyle=ls, label=label, linewidth=1.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Val mIoU')
    ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Line 2: Internal Supplementary
    ax = axes[1]
    ax.set_title('Line 2: Internal Supplementary', fontweight='bold', fontsize=12)
    for name, label, color, ls in [
        ('finetune_ws2supp_scratch', 'WS2_supp Scratch', '#2196F3', '-'),
        ('finetune_ws2supp_ft_resethead', 'WS2_supp FT+reset', '#F44336', '-'),
        ('finetune_ws2supp_ft_keephead', 'WS2_supp FT+keep', '#FF9800', '--'),
        ('finetune_grsupp_scratch', 'Gr_supp Scratch', '#4CAF50', '-'),
        ('finetune_grsupp_ft_resethead', 'Gr_supp FT+reset', '#9C27B0', '-'),
    ]:
        log = f'output/{name}/finetune.log'
        if not os.path.exists(log): continue
        ep, miou, _ = parse_finetune_log(log)
        if ep: ax.plot(ep, miou, color=color, linestyle=ls, label=label, linewidth=1.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Val mIoU')
    ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Line 3: Combined
    ax = axes[2]
    ax.set_title('Line 3: Combined Target', fontweight='bold', fontsize=12)
    for name, label, color, ls in [
        ('finetune_ws2combined_scratch', 'Combined Scratch', '#2196F3', '-'),
        ('finetune_ws2combined_ft_resethead', 'Combined FT+reset', '#F44336', '-'),
    ]:
        log = f'output/{name}/finetune.log'
        if not os.path.exists(log): continue
        ep, miou, _ = parse_finetune_log(log)
        if ep: ax.plot(ep, miou, color=color, linestyle=ls, label=label, linewidth=1.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Val mIoU')
    ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.suptitle('Transfer Learning — Training Curves (Three Lines)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = f'{OUT_DIR}/transfer_training_curves.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# ── 2. Bar Chart Comparison ──────────────────────────────
def plot_bar_comparison():
    """Bar chart comparing val mIoU across all experiments."""
    # Read results from results.txt files
    experiments = []
    configs = [
        ('L1\nWS2\nScratch', 'finetune_ws2_scratch', '#2196F3'),
        ('L1\nWS2\nFT+reset', 'finetune_ws2_r2_resethead', '#F44336'),
        ('L1\nWS2\nFT+keep', 'finetune_ws2_r2_keephead', '#FF9800'),
        ('L1\nGr\nScratch', 'finetune_graphene_scratch', '#2196F3'),
        ('L1\nGr\nFT partial', 'finetune_graphene_r3_partial', '#F44336'),
        ('L2\nWS2s\nScratch', 'finetune_ws2supp_scratch', '#2196F3'),
        ('L2\nWS2s\nFT+reset', 'finetune_ws2supp_ft_resethead', '#F44336'),
        ('L2\nGrs\nScratch', 'finetune_grsupp_scratch', '#2196F3'),
        ('L2\nGrs\nFT+reset', 'finetune_grsupp_ft_resethead', '#F44336'),
        ('L3\nComb\nScratch', 'finetune_ws2combined_scratch', '#2196F3'),
        ('L3\nComb\nFT+reset', 'finetune_ws2combined_ft_resethead', '#F44336'),
    ]

    labels, values, colors = [], [], []
    for label, name, color in configs:
        rfile = f'output/{name}/results.txt'
        if not os.path.exists(rfile):
            continue
        with open(rfile) as f:
            for line in f:
                m = re.search(r'mIoU:\s*([\d.]+)', line)
                if m:
                    labels.append(label)
                    values.append(float(m.group(1)) * 100)
                    colors.append(color)
                    break

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, width=0.7, edgecolor='white', linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Line separators
    for sep in [3, 5, 8]:
        if sep < len(labels):
            ax.axvline(x=sep - 0.5, color='gray', linestyle='--', alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, ha='center')
    ax.set_ylabel('Val mIoU (%)')
    ax.set_ylim(50, 100)
    ax.set_title('Transfer Learning — Val mIoU Comparison (All Experiments)', fontweight='bold')

    legend = [Patch(facecolor='#2196F3', label='Scratch'),
              Patch(facecolor='#F44336', label='FT+reset_head'),
              Patch(facecolor='#FF9800', label='FT+keep_head')]
    ax.legend(handles=legend, loc='lower right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = f'{OUT_DIR}/transfer_bar_comparison.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# ── 3. Per-class IoU Grouped Bar Chart ──────────────────
def plot_perclass_comparison():
    """Per-class IoU comparison for WS2 across lines."""
    CLS = ['BG', '1L', 'FL', 'ML']
    data = {}
    configs = [
        ('L1 Scratch', 'finetune_ws2_scratch'),
        ('L1 FT+reset', 'finetune_ws2_r2_resethead'),
        ('L2 supp Scratch', 'finetune_ws2supp_scratch'),
        ('L3 comb Scratch', 'finetune_ws2combined_scratch'),
        ('L3 comb FT+reset', 'finetune_ws2combined_ft_resethead'),
    ]

    for label, name in configs:
        rfile = f'output/{name}/results.txt'
        if not os.path.exists(rfile): continue
        ious = []
        with open(rfile) as f:
            for line in f:
                m = re.search(r'IoU=([\d.]+)', line)
                if m: ious.append(float(m.group(1)))
        if len(ious) >= 4:
            data[label] = ious[:4]

    fig, ax = plt.subplots(figsize=(12, 5))
    n = len(data)
    x = np.arange(len(CLS))
    w = 0.8 / n
    colors_list = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0']

    for i, (label, ious) in enumerate(data.items()):
        offset = (i - n/2 + 0.5) * w
        bars = ax.bar(x + offset, [v*100 for v in ious], w, label=label,
                      color=colors_list[i % len(colors_list)], edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(CLS, fontsize=11, fontweight='bold')
    ax.set_ylabel('IoU (%)')
    ax.set_ylim(0, 105)
    ax.set_title('WS2 Per-class IoU — Cross-Line Comparison', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = f'{OUT_DIR}/transfer_perclass_ws2.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


if __name__ == '__main__':
    plot_training_curves()
    plot_bar_comparison()
    plot_perclass_comparison()
    print('\n✅ All visualizations complete!')
