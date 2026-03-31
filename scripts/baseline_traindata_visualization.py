"""
Main baseline training data visualization — SCI paper style.

Generates 5 figures from train.log:
  1. Train Loss + Val mIoU (dual axis)
  2. Learning Rate schedule
  3. Per-class IoU over epochs
  4. Val Loss components (CE + Dice)
  5. Train vs Val mIoU

Output: output/paper_figures/
Usage:
  python scripts/baseline_traindata_visualization.py
  python scripts/baseline_traindata_visualization.py --seed-tag seed_42
"""
import argparse
import glob
import re, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# ═══════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════
OUT = 'output/paper_figures'
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.linewidth': 0.8,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.03,
})


def smooth(values, window=7):
    """Uniform filter smoothing."""
    if len(values) < window:
        return values
    return uniform_filter1d(np.array(values, dtype=float),
                            size=window, mode='nearest').tolist()


# ═══════════════════════════════════════════════════════════
# Parse train.log
# ═══════════════════════════════════════════════════════════
def parse_log(path):
    lr, train_loss, train_miou = [], [], []
    val_loss, val_ce, val_dice, val_miou, val_f1 = [], [], [], [], []
    iou_bg, iou_mono, iou_few, iou_multi = [], [], [], []

    with open(path) as f:
        for line in f:
            m = re.search(r'Epoch \[\d+/\d+\] LR=([\d.e-]+)', line)
            if m:
                lr.append(float(m.group(1)))
                continue

            m = re.search(r'Train\s+Loss=([\d.]+)\s+mIoU=([\d.]+)', line)
            if m:
                train_loss.append(float(m.group(1)))
                train_miou.append(float(m.group(2)))
                continue

            m = re.search(
                r'Val\s+Loss=([\d.]+)\s+\(CE=([\d.]+)\s+Dice=([\d.]+)\)'
                r'\s+mIoU=([\d.]+)\s+F1=([\d.]+)', line)
            if m:
                val_loss.append(float(m.group(1)))
                val_ce.append(float(m.group(2)))
                val_dice.append(float(m.group(3)))
                val_miou.append(float(m.group(4)))
                val_f1.append(float(m.group(5)))
                continue

            m = re.search(
                r'IoU: background: ([\d.]+) \| monolayer: ([\d.]+) '
                r'\| fewlayer: ([\d.]+) \| multilayer: ([\d.]+)', line)
            if m:
                iou_bg.append(float(m.group(1)))
                iou_mono.append(float(m.group(2)))
                iou_few.append(float(m.group(3)))
                iou_multi.append(float(m.group(4)))

    n = min(len(train_loss), len(val_miou))
    return {
        'n': n,
        'ep': list(range(1, n + 1)),
        'lr': lr[:n],
        'train_loss': train_loss[:n],
        'train_miou': train_miou[:n],
        'val_loss': val_loss[:n],
        'val_ce': val_ce[:n],
        'val_dice': val_dice[:n],
        'val_miou': val_miou[:n],
        'val_f1': val_f1[:n],
        'iou_bg': iou_bg[:n],
        'iou_mono': iou_mono[:n],
        'iou_few': iou_few[:n],
        'iou_multi': iou_multi[:n],
    }


# ═══════════════════════════════════════════════════════════
# Plotting functions
# ═══════════════════════════════════════════════════════════

def plot_loss_miou(d):
    """Fig 1: Train Loss + Val mIoU (dual axis)."""
    fig, ax1 = plt.subplots(figsize=(5.5, 3.5))
    c_loss, c_miou = '#1f77b4', '#d62728'

    ax1.plot(d['ep'], d['train_loss'], c_loss, alpha=0.15, lw=0.5)
    ax1.plot(d['ep'], smooth(d['train_loss']), c_loss, lw=1.8, label='Train Loss')
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Train Loss', color=c_loss, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=c_loss)
    ax1.set_xlim(1, d['n'])
    ax1.grid(True, alpha=0.2, lw=0.5)

    ax2 = ax1.twinx()
    ax2.plot(d['ep'], [v*100 for v in d['val_miou']], c_miou, alpha=0.15, lw=0.5)
    ax2.plot(d['ep'], [v*100 for v in smooth(d['val_miou'])], c_miou, lw=1.8,
             label='Val mIoU')
    ax2.set_ylabel('Val mIoU (%)', color=c_miou, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=c_miou)

    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1+l2, lb1+lb2, loc='center right',
               frameon=True, fancybox=False, edgecolor='gray', framealpha=0.9)
    plt.tight_layout()
    path = f'{OUT}/curve_mos2_baseline.png'
    fig.savefig(path); plt.close(fig)
    print(f'  1. {path}')


def plot_lr_schedule(d):
    """Fig 2: Learning Rate schedule."""
    fig, ax = plt.subplots(figsize=(5.5, 2.5))
    ax.plot(d['ep'], [l*1000 for l in d['lr']], '#2ca02c', lw=1.8)
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Learning Rate (×10⁻³)', fontweight='bold')
    ax.set_xlim(1, d['n'])
    ax.grid(True, alpha=0.2, lw=0.5)
    ax.ticklabel_format(useOffset=False)
    plt.tight_layout()
    path = f'{OUT}/curve_lr_schedule.png'
    fig.savefig(path); plt.close(fig)
    print(f'  2. {path}')


def plot_perclass_iou(d):
    """Fig 3: Per-class IoU over epochs."""
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    series = [
        (d['iou_bg'],    '#1f77b4', 'Background'),
        (d['iou_mono'],  '#ff7f0e', 'Monolayer'),
        (d['iou_few'],   '#2ca02c', 'Few-layer'),
        (d['iou_multi'], '#d62728', 'Multilayer'),
    ]
    for vals, c, lb in series:
        ep = d['ep'][:len(vals)]
        ax.plot(ep, [v*100 for v in vals], c, alpha=0.15, lw=0.5)
        ax.plot(ep, [v*100 for v in smooth(vals)], c, lw=1.8, label=lb)

    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('IoU (%)', fontweight='bold')
    ax.set_xlim(1, d['n'])
    ax.grid(True, alpha=0.2, lw=0.5)
    ax.legend(loc='lower right', frameon=True, fancybox=False,
              edgecolor='gray', framealpha=0.9, ncol=2)
    plt.tight_layout()
    path = f'{OUT}/curve_perclass_iou.png'
    fig.savefig(path); plt.close(fig)
    print(f'  3. {path}')


def plot_val_loss_components(d):
    """Fig 4: Val Loss components (CE + Dice)."""
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.plot(d['ep'], smooth(d['val_ce']), '#1f77b4', lw=1.8, label='CE Loss')
    ax.plot(d['ep'], smooth(d['val_dice']), '#ff7f0e', lw=1.8, label='Dice Loss')
    ax.plot(d['ep'], smooth(d['val_loss']), '#333333', lw=1.5, ls='--',
            label='Total Val Loss')
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Loss', fontweight='bold')
    ax.set_xlim(1, d['n'])
    ax.grid(True, alpha=0.2, lw=0.5)
    ax.legend(loc='upper right', frameon=True, fancybox=False,
              edgecolor='gray', framealpha=0.9)
    plt.tight_layout()
    path = f'{OUT}/curve_val_loss_components.png'
    fig.savefig(path); plt.close(fig)
    print(f'  4. {path}')


def plot_train_val_miou(d):
    """Fig 5: Train vs Val mIoU."""
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.plot(d['ep'], [v*100 for v in smooth(d['train_miou'])], '#1f77b4',
            lw=1.8, label='Train mIoU')
    ax.plot(d['ep'], [v*100 for v in smooth(d['val_miou'])], '#d62728',
            lw=1.8, label='Val mIoU')
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('mIoU (%)', fontweight='bold')
    ax.set_xlim(1, d['n'])
    ax.grid(True, alpha=0.2, lw=0.5)
    ax.legend(loc='lower right', frameon=True, fancybox=False,
              edgecolor='gray', framealpha=0.9)
    plt.tight_layout()
    path = f'{OUT}/curve_train_val_miou.png'
    fig.savefig(path); plt.close(fig)
    print(f'  5. {path}')


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot SCI-style baseline training curves')
    parser.add_argument('--seed-tag', default='seed_123',
                        help='Seed directory under output/seed_test, e.g. seed_123')
    parser.add_argument('--log-path', default=None,
                        help='Optional explicit train.log path')
    args = parser.parse_args()

    if args.log_path:
        log_path = args.log_path
    else:
        matches = glob.glob(f'output/seed_test/{args.seed_tag}/*/train.log')
        if not matches:
            raise FileNotFoundError(f'No train.log found for {args.seed_tag}')
        log_path = matches[0]

    print(f'Parsing {log_path} ...')
    data = parse_log(log_path)
    print(f'  {data["n"]} epochs parsed\n')

    print('Generating SCI-style training visualizations:')
    plot_loss_miou(data)
    plot_lr_schedule(data)
    plot_perclass_iou(data)
    plot_val_loss_components(data)
    plot_train_val_miou(data)

    print(f'\n✅ All figures saved to {OUT}/')
