"""
SCI-style paper figures:
1. Confusion matrices — no title, no axis labels, bold white text, blue colormap
2. Smoothed training curves — clean, publication-ready

Main MoS2 baseline now defaults to seed_123, which is the representative
single-run model in the paper.

Output: output/paper_figures/
"""
import sys, os, re, glob
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, '/root/autodl-tmp/PhysicalNet')
os.chdir('/root/autodl-tmp/PhysicalNet')

import torch, torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms as T
from models.repela_net import RepELANet

OUT = 'output/paper_figures'
os.makedirs(OUT, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ═══════════════════════════════════════════════════════════
# SCI Style Config
# ═══════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.linewidth': 0.8,
    'axes.labelsize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
BASELINE_SEED = 'seed_123'

# ═══════════════════════════════════════════════════════════
# Part 1: Confusion Matrices (from saved eval results)
# ═══════════════════════════════════════════════════════════

def cm_from_saved_preds(eval_dir, gt_dir, test_names, nc=4):
    """
    Compute row-normalized confusion matrix from saved pred masks.
    Reads {eval_dir}/{name}_pred.png and {gt_dir}/{name}.png.
    This guarantees 100% consistency with test_metrics.txt.
    """
    cm = np.zeros((nc, nc), dtype=np.float64)
    found = 0
    for name in test_names:
        pred_path = os.path.join(eval_dir, f'{name}_pred.png')
        gt_path = os.path.join(gt_dir, f'{name}.png')
        if not os.path.exists(pred_path):
            print(f'    WARNING: {pred_path} not found, skipping')
            continue
        if not os.path.exists(gt_path):
            print(f'    WARNING: {gt_path} not found, skipping')
            continue
        pred = np.array(Image.open(pred_path))
        gt = np.array(Image.open(gt_path))
        for t in range(nc):
            for p in range(nc):
                cm[t, p] += ((gt == t) & (pred == p)).sum()
        found += 1
    if found == 0:
        raise RuntimeError(f'ERROR: no pred masks found in {eval_dir}')
    if found < len(test_names):
        print(f'    WARNING: only {found}/{len(test_names)} preds found')
    row_sums = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, row_sums, where=row_sums > 0, out=np.zeros_like(cm))


def plot_confusion_matrix(cm, save_path):
    """Plot clean confusion matrix matching reference: vertical colorbar, bold serif."""
    import matplotlib.gridspec as gridspec
    nc = cm.shape[0]

    fig = plt.figure(figsize=(4.5, 4.0))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 0.035], wspace=0.05)
    ax = fig.add_subplot(gs[0])
    cax = fig.add_subplot(gs[1])

    cmap = plt.cm.Blues
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1,
                   aspect='equal')

    for i in range(nc):
        for j in range(nc):
            val = cm[i, j]
            color = 'white' if val > 0.45 else '#1a1a1a'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=10, fontweight='bold',
                    color=color, fontfamily='serif')

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(''); ax.set_ylabel(''); ax.set_title('')

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('#333333')

    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.ax.tick_params(labelsize=9, length=3, width=0.5)
    cbar.outline.set_linewidth(0.5)

    fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
    print(f'  CM saved: {save_path}')


# ═══════════════════════════════════════════════════════════
# Part 2: Smoothed Training Curves

def parse_train_log(log_path):
    """Parse train.log → epochs, train_loss, val_miou, lr."""
    epochs, train_loss, val_miou, lr = [], [], [], []
    with open(log_path) as f:
        for line in f:
            # Epoch line
            m_epoch = re.search(r'Epoch \[(\d+)/\d+\] LR[=:]?([\d.e-]+)?', line)
            if m_epoch:
                ep = int(m_epoch.group(1))
                if m_epoch.group(2):
                    lr.append(float(m_epoch.group(2)))
                continue
            # Train line
            m_train = re.search(r'Train\s+Loss[=:]?\s*([\d.]+)\s+mIoU[=:]?\s*([\d.]+)', line)
            if not m_train:
                m_train = re.search(r'Train Loss:\s*([\d.]+)\s+mIoU:\s*([\d.]+)', line)
            if m_train:
                train_loss.append(float(m_train.group(1)))
                continue
            # Val line
            m_val = re.search(r'Val\s+.*mIoU[=:]?\s*([\d.]+)', line)
            if not m_val:
                m_val = re.search(r'Val mIoU:\s*([\d.]+)', line)
            if m_val:
                val_miou.append(float(m_val.group(1)))
                continue

    n = min(len(train_loss), len(val_miou))
    return {
        'epochs': list(range(1, n+1)),
        'train_loss': train_loss[:n],
        'val_miou': val_miou[:n],
        'lr': lr[:n] if len(lr) >= n else None,
    }


def smooth(values, window=7):
    """Smooth with uniform filter."""
    if len(values) < window:
        return values
    return uniform_filter1d(np.array(values, dtype=float), size=window, mode='nearest').tolist()


def plot_training_curves(data_dict, save_path, title_prefix=''):
    """Plot SCI-style dual-axis training curves."""
    fig, ax1 = plt.subplots(figsize=(5.5, 3.5))

    epochs = data_dict['epochs']
    loss_smooth = smooth(data_dict['train_loss'], window=7)
    miou_smooth = smooth(data_dict['val_miou'], window=7)

    # Colors
    c_loss = '#1f77b4'   # blue
    c_miou = '#d62728'   # red

    # Loss (left axis)
    ax1.plot(epochs, data_dict['train_loss'], color=c_loss, alpha=0.15, linewidth=0.5)
    ax1.plot(epochs, loss_smooth, color=c_loss, linewidth=1.8, label='Train Loss')
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Train Loss', color=c_loss, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=c_loss)
    ax1.set_xlim(1, len(epochs))
    ax1.grid(True, alpha=0.2, linewidth=0.5)

    # mIoU (right axis)
    ax2 = ax1.twinx()
    ax2.plot(epochs, [v*100 for v in data_dict['val_miou']], color=c_miou, alpha=0.15, linewidth=0.5)
    ax2.plot(epochs, [v*100 for v in miou_smooth], color=c_miou, linewidth=1.8, label='Val mIoU')
    ax2.set_ylabel('Val mIoU (%)', color=c_miou, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=c_miou)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right',
               frameon=True, fancybox=False, edgecolor='gray',
               framealpha=0.9, fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print(f'  Curve saved: {save_path}')


# ═══════════════════════════════════════════════════════════
# Main Execution
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    # ── 1. Confusion Matrices (from saved eval.py predictions) ──
    print('=== Confusion Matrices (from saved pred masks) ===')
    GT_DIR = 'Mos2_data/mask'
    test_names = open('splits/test.txt').read().strip().split('\n')

    # All CM sources: (output_name, eval_dir)
    cm_jobs = [
        # Baseline
        ('cm_mos2_baseline', f'output/eval_results/{BASELINE_SEED}'),
        # Ablation
        ('cm_ablation_no_ela', 'output/eval_results/ablation_no_ela'),
        ('cm_ablation_no_dwmff', 'output/eval_results/ablation_no_dwmff'),
        ('cm_ablation_no_boundary', 'output/eval_results/ablation_no_boundary'),
        ('cm_ablation_no_rep', 'output/eval_results/ablation_no_rep'),
        # Decoder comparison
        ('cm_decoder_unet', 'output/eval_results/decoder_unet'),
        ('cm_decoder_fpn', 'output/eval_results/decoder_fpn'),
        ('cm_decoder_aspp', 'output/eval_results/decoder_aspp'),
        ('cm_decoder_ppm', 'output/eval_results/decoder_ppm'),
        ('cm_decoder_segformer', 'output/eval_results/decoder_segformer'),
        ('cm_decoder_hamburger', 'output/eval_results/decoder_hamburger'),
        ('cm_decoder_ours', 'output/eval_results/decoder_ours'),
    ]

    for name, eval_dir in cm_jobs:
        if not os.path.isdir(eval_dir):
            print(f'  SKIP {name}: {eval_dir} not found')
            continue
        print(f'  {name} <- {eval_dir}')
        cm = cm_from_saved_preds(eval_dir, GT_DIR, test_names, nc=4)
        plot_confusion_matrix(cm, f'{OUT}/{name}.png')

    # ── 4. Training Curves ──
    print('\n=== Training Curves ===')

    # MoS2 baseline (representative single run)
    log_path = glob.glob(f'output/seed_test/{BASELINE_SEED}/*/train.log')[0]
    data = parse_train_log(log_path)
    if data['epochs']:
        plot_training_curves(data, f'{OUT}/curve_mos2_baseline.png')

    # Ablation curves
    for abl_log in sorted(glob.glob('output/ablation/*/train.log')):
        abl_name = os.path.basename(os.path.dirname(abl_log)).split('_2026')[0]
        data = parse_train_log(abl_log)
        if data['epochs']:
            plot_training_curves(data, f'{OUT}/curve_ablation_{abl_name}.png')

    # Decoder comparison curves
    for dec_log in sorted(glob.glob('output/decoder_compare/*/train.log')):
        dec_name = os.path.basename(os.path.dirname(dec_log)).split('_2026')[0]
        data = parse_train_log(dec_log)
        if data['epochs']:
            plot_training_curves(data, f'{OUT}/curve_decoder_{dec_name}.png')

    # Transfer: key experiments
    transfer_logs = {
        'ws2_scratch': 'output/finetune_ws2_scratch/finetune.log',
        'ws2_ft_resethead': 'output/finetune_ws2_r2_resethead/finetune.log',
        'graphene_scratch': 'output/finetune_graphene_scratch/finetune.log',
        'graphene_ft_partial': 'output/finetune_graphene_r3_partial/finetune.log',
        'mos2v2_scratch': 'output/finetune_mos2v2_scratch/finetune.log',
        'mos2v2_ft_resethead': 'output/finetune_mos2v2_ft_resethead/finetune.log',
    }
    for name, lp in transfer_logs.items():
        if os.path.exists(lp):
            data = parse_train_log(lp)
            if data['epochs']:
                plot_training_curves(data, f'{OUT}/curve_transfer_{name}.png')

    print(f'\n✅ All figures saved to {OUT}/')
