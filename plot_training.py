"""
Training Log Visualization for RepELA-Net.

Parses training logs and generates publication-quality plots:
  - Loss curves (total, focal, dice)
  - mIoU curves (train & val)
  - Per-class IoU curves
  - Learning rate schedule
  - Pixel accuracy & F1 curves

Usage:
    python plot_training.py --log output/train_log.txt --output output/plots/
"""

import os
import re
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Publication-quality style
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Color palette
COLORS = {
    'train': '#2196F3',
    'val': '#F44336',
    'focal': '#FF9800',
    'dice': '#9C27B0',
    'total': '#2196F3',
    'background': '#607D8B',
    'monolayer': '#E91E63',
    'fewlayer': '#4CAF50',
    'multilayer': '#3F51B5',
    'lr': '#795548',
    'acc': '#009688',
    'f1': '#FF5722',
}


def parse_log(log_path):
    """Parse training log file and extract metrics."""
    data = {
        'epoch': [],
        'lr': [],
        'train_loss': [], 'train_focal': [], 'train_dice': [],
        'train_miou': [], 'train_acc': [],
        'val_loss': [], 'val_miou': [], 'val_acc': [], 'val_f1': [],
        'val_iou_bg': [], 'val_iou_mono': [],
        'val_iou_few': [], 'val_iou_multi': [],
    }

    with open(log_path, 'r') as f:
        lines = f.readlines()

    epoch = None
    lr = None

    for line in lines:
        # Parse epoch and LR
        m = re.search(r'Epoch \[(\d+)/\d+\] LR: ([\d.]+)', line)
        if m:
            epoch = int(m.group(1))
            lr = float(m.group(2))
            continue

        # Parse train metrics (supports both 'Focal' and 'CE' label)
        m = re.search(
            r'Train Loss: ([\d.]+) \((?:Focal|CE): ([\d.]+), Dice: ([\d.]+)\) '
            r'mIoU: ([\d.]+) Acc: ([\d.]+)', line
        )
        if m and epoch is not None:
            data['epoch'].append(epoch)
            data['lr'].append(lr)
            data['train_loss'].append(float(m.group(1)))
            data['train_focal'].append(float(m.group(2)))
            data['train_dice'].append(float(m.group(3)))
            data['train_miou'].append(float(m.group(4)))
            data['train_acc'].append(float(m.group(5)))
            continue

        # Parse val metrics (v1 format: "Val   Loss: ... mIoU: ...")
        m = re.search(
            r'Val   Loss: ([\d.]+) mIoU: ([\d.]+) Acc: ([\d.]+) F1: ([\d.]+)',
            line
        )
        if m:
            data['val_loss'].append(float(m.group(1)))
            data['val_miou'].append(float(m.group(2)))
            data['val_acc'].append(float(m.group(3)))
            data['val_f1'].append(float(m.group(4)))
            continue

        # Parse val metrics (v2 format: "Val [sliding-window] mIoU: ...")
        m = re.search(
            r'Val \[sliding-window\] mIoU: ([\d.]+) Acc: ([\d.]+) F1: ([\d.]+)',
            line
        )
        if m:
            data['val_loss'].append(0.0)
            data['val_miou'].append(float(m.group(1)))
            data['val_acc'].append(float(m.group(2)))
            data['val_f1'].append(float(m.group(3)))
            continue

        # Parse val metrics (v3 format: "Val Loss: ... (CE: ..., Dice: ...) mIoU: ...")
        m = re.search(
            r'Val Loss: ([\d.]+) \((?:CE|Focal): [\d.]+, Dice: [\d.]+\) '
            r'mIoU: ([\d.]+) Acc: ([\d.]+) F1: ([\d.]+)',
            line
        )
        if m:
            data['val_loss'].append(float(m.group(1)))
            data['val_miou'].append(float(m.group(2)))
            data['val_acc'].append(float(m.group(3)))
            data['val_f1'].append(float(m.group(4)))
            continue

        # Parse per-class IoU
        m = re.search(
            r'Per-class IoU: background: ([\d.]+) \| monolayer: ([\d.]+) '
            r'\| fewlayer: ([\d.]+) \| multilayer: ([\d.]+)', line
        )
        if m:
            data['val_iou_bg'].append(float(m.group(1)))
            data['val_iou_mono'].append(float(m.group(2)))
            data['val_iou_few'].append(float(m.group(3)))
            data['val_iou_multi'].append(float(m.group(4)))
            continue

    # Truncate to min length
    min_len = min(len(v) for v in data.values() if len(v) > 0)
    for k in data:
        data[k] = data[k][:min_len]

    return data


def smooth(values, weight=0.9):
    """Exponential moving average smoothing."""
    smoothed = []
    last = values[0]
    for v in values:
        s = weight * last + (1 - weight) * v
        smoothed.append(s)
        last = s
    return smoothed


def plot_loss_curves(data, output_dir):
    """Plot training loss curves."""
    has_val_loss = any(v > 0 for v in data['val_loss'])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = data['epoch']

    # Left: Loss decomposition (train)
    ax1.plot(epochs, data['train_loss'], alpha=0.3, color=COLORS['total'])
    ax1.plot(epochs, smooth(data['train_loss']), color=COLORS['total'],
             linewidth=2, label='Total Loss')
    ax1.plot(epochs, data['train_focal'], alpha=0.3, color=COLORS['focal'])
    ax1.plot(epochs, smooth(data['train_focal']), color=COLORS['focal'],
             linewidth=2, label='Focal Loss')
    ax1.plot(epochs, data['train_dice'], alpha=0.3, color=COLORS['dice'])
    ax1.plot(epochs, smooth(data['train_dice']), color=COLORS['dice'],
             linewidth=2, label='Dice Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Decomposition')
    ax1.legend()
    ax1.set_ylim(bottom=0)

    # Right: Train loss (+ Val loss if available)
    ax2.plot(epochs, data['train_loss'], alpha=0.2, color=COLORS['train'])
    ax2.plot(epochs, smooth(data['train_loss']), color=COLORS['train'],
             linewidth=2, label='Train Loss')
    if has_val_loss:
        ax2.plot(epochs, smooth(data['val_loss']), color=COLORS['val'],
                 linewidth=2, label='Val Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss')
    ax2.legend()
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'loss_curves.png')
    plt.savefig(save_path)
    plt.close()
    print(f'Saved: {save_path}')


def plot_miou_curves(data, output_dir):
    """Plot mIoU curves."""
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = data['epoch']

    ax.plot(epochs, data['train_miou'], alpha=0.2, color=COLORS['train'])
    ax.plot(epochs, smooth(data['train_miou']), color=COLORS['train'],
            linewidth=2, label='Train mIoU')
    ax.plot(epochs, data['val_miou'], alpha=0.2, color=COLORS['val'])
    ax.plot(epochs, smooth(data['val_miou']), color=COLORS['val'],
            linewidth=2, label='Val mIoU')

    # Mark best val mIoU
    best_idx = np.argmax(data['val_miou'])
    best_miou = data['val_miou'][best_idx]
    best_epoch = data['epoch'][best_idx]
    ax.scatter([best_epoch], [best_miou], s=100, c='gold', edgecolors='black',
               zorder=5, marker='*')
    ax.annotate(f'Best: {best_miou:.4f}\n(Epoch {best_epoch})',
                xy=(best_epoch, best_miou),
                xytext=(best_epoch + 10, best_miou - 0.05),
                fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black'))

    ax.set_xlabel('Epoch')
    ax.set_ylabel('mIoU')
    ax.set_title('Mean Intersection over Union (mIoU)')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'miou_curves.png')
    plt.savefig(save_path)
    plt.close()
    print(f'Saved: {save_path}')


def plot_per_class_iou(data, output_dir):
    """Plot per-class IoU curves."""
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = data['epoch']
    classes = [
        ('val_iou_bg', 'Background', COLORS['background']),
        ('val_iou_mono', 'Monolayer', COLORS['monolayer']),
        ('val_iou_few', 'Fewlayer', COLORS['fewlayer']),
        ('val_iou_multi', 'Multilayer', COLORS['multilayer']),
    ]

    for key, name, color in classes:
        ax.plot(epochs, data[key], alpha=0.2, color=color)
        ax.plot(epochs, smooth(data[key]), color=color, linewidth=2, label=name)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('IoU')
    ax.set_title('Per-Class IoU (Validation)')
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'per_class_iou.png')
    plt.savefig(save_path)
    plt.close()
    print(f'Saved: {save_path}')


def plot_acc_f1(data, output_dir):
    """Plot accuracy and F1 curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = data['epoch']

    # Pixel accuracy
    ax1.plot(epochs, smooth(data['train_acc']), color=COLORS['train'],
             linewidth=2, label='Train Acc')
    ax1.plot(epochs, smooth(data['val_acc']), color=COLORS['val'],
             linewidth=2, label='Val Acc')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Pixel Accuracy')
    ax1.set_title('Pixel Accuracy')
    ax1.legend()
    ax1.set_ylim(0, 1)

    # F1 score
    ax2.plot(epochs, data['val_f1'], alpha=0.2, color=COLORS['f1'])
    ax2.plot(epochs, smooth(data['val_f1']), color=COLORS['f1'],
             linewidth=2, label='Val F1')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('Mean F1 Score (Validation)')
    ax2.legend()
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'acc_f1_curves.png')
    plt.savefig(save_path)
    plt.close()
    print(f'Saved: {save_path}')


def plot_lr_schedule(data, output_dir):
    """Plot learning rate schedule."""
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(data['epoch'], data['lr'], color=COLORS['lr'], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule (Cosine Warmup)')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-4, -4))

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'lr_schedule.png')
    plt.savefig(save_path)
    plt.close()
    print(f'Saved: {save_path}')


def plot_summary_dashboard(data, output_dir):
    """Plot a single summary dashboard with all key metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    epochs = data['epoch']

    # (0,0) Total loss
    has_val_loss = any(v > 0 for v in data['val_loss'])
    axes[0, 0].plot(epochs, data['train_loss'], alpha=0.2, color=COLORS['train'])
    axes[0, 0].plot(epochs, smooth(data['train_loss']), color=COLORS['train'],
                    linewidth=2, label='Train')
    if has_val_loss:
        axes[0, 0].plot(epochs, smooth(data['val_loss']), color=COLORS['val'],
                        linewidth=2, label='Val')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].set_ylim(bottom=0)

    # (0,1) mIoU
    axes[0, 1].plot(epochs, smooth(data['train_miou']), color=COLORS['train'],
                    linewidth=2, label='Train')
    axes[0, 1].plot(epochs, smooth(data['val_miou']), color=COLORS['val'],
                    linewidth=2, label='Val')
    best_idx = np.argmax(data['val_miou'])
    axes[0, 1].scatter([data['epoch'][best_idx]], [data['val_miou'][best_idx]],
                       s=80, c='gold', edgecolors='black', zorder=5, marker='*')
    axes[0, 1].set_title(f'mIoU (Best Val: {data["val_miou"][best_idx]:.4f})')
    axes[0, 1].legend()
    axes[0, 1].set_ylim(0, 1)

    # (0,2) Per-class IoU
    for key, name, color in [
        ('val_iou_bg', 'BG', COLORS['background']),
        ('val_iou_mono', 'Mono', COLORS['monolayer']),
        ('val_iou_few', 'Few', COLORS['fewlayer']),
        ('val_iou_multi', 'Multi', COLORS['multilayer']),
    ]:
        axes[0, 2].plot(epochs, smooth(data[key]), color=color, linewidth=2, label=name)
    axes[0, 2].set_title('Per-Class IoU (Val)')
    axes[0, 2].legend()
    axes[0, 2].set_ylim(0, 1)

    # (1,0) Focal + Dice
    axes[1, 0].plot(epochs, smooth(data['train_focal']), color=COLORS['focal'],
                    linewidth=2, label='Focal')
    axes[1, 0].plot(epochs, smooth(data['train_dice']), color=COLORS['dice'],
                    linewidth=2, label='Dice')
    axes[1, 0].set_title('Loss Components (Train)')
    axes[1, 0].legend()
    axes[1, 0].set_ylim(bottom=0)

    # (1,1) Acc + F1
    axes[1, 1].plot(epochs, smooth(data['val_acc']), color=COLORS['acc'],
                    linewidth=2, label='Pixel Acc')
    axes[1, 1].plot(epochs, smooth(data['val_f1']), color=COLORS['f1'],
                    linewidth=2, label='Mean F1')
    axes[1, 1].set_title('Accuracy & F1 (Val)')
    axes[1, 1].legend()
    axes[1, 1].set_ylim(0, 1)

    # (1,2) LR
    axes[1, 2].plot(epochs, data['lr'], color=COLORS['lr'], linewidth=2)
    axes[1, 2].set_title('Learning Rate')
    axes[1, 2].ticklabel_format(axis='y', style='sci', scilimits=(-4, -4))

    for ax in axes.flat:
        ax.set_xlabel('Epoch')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle('RepELA-Net Training Dashboard', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'training_dashboard.png')
    plt.savefig(save_path)
    plt.close()
    print(f'Saved: {save_path}')


def main():
    parser = argparse.ArgumentParser(description='Plot training curves')
    parser.add_argument('--log', type=str, default='output/train_log.txt',
                        help='Path to training log')
    parser.add_argument('--output', type=str, default='output/plots/',
                        help='Output directory for plots')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f'Parsing log: {args.log}')
    data = parse_log(args.log)
    print(f'Found {len(data["epoch"])} epochs')

    if len(data['epoch']) == 0:
        print('No data found in log!')
        return

    # Print summary
    best_idx = np.argmax(data['val_miou'])
    print(f'\n=== Training Summary ===')
    print(f'Total epochs: {len(data["epoch"])}')
    print(f'Best val mIoU: {data["val_miou"][best_idx]:.4f} (Epoch {data["epoch"][best_idx]})')
    print(f'  Background: {data["val_iou_bg"][best_idx]:.4f}')
    print(f'  Monolayer:  {data["val_iou_mono"][best_idx]:.4f}')
    print(f'  Fewlayer:   {data["val_iou_few"][best_idx]:.4f}')
    print(f'  Multilayer: {data["val_iou_multi"][best_idx]:.4f}')
    print(f'Final train loss: {data["train_loss"][-1]:.4f}')
    print()

    # Generate plots
    plot_loss_curves(data, args.output)
    plot_miou_curves(data, args.output)
    plot_per_class_iou(data, args.output)
    plot_acc_f1(data, args.output)
    plot_lr_schedule(data, args.output)
    plot_summary_dashboard(data, args.output)

    print(f'\nAll plots saved to {args.output}')


if __name__ == '__main__':
    main()
