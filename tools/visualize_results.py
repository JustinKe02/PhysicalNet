"""
Comprehensive Visualization for RepELA-Net Training Results.

Generates:
  1. Training curves (loss, mIoU, LR vs epoch)
  2. Per-class IoU over epochs
  3. Confusion matrix from test evaluation
  4. Inference results: original / prediction / ground truth grid
  5. Summary metrics table image

Usage:
    python tools/visualize_results.py \
      --log_file output/seed_test/seed_42/repela_small_*/train.log \
      --checkpoint output/seed_test/seed_42/repela_small_*/best_model.pth \
      --output output/visualization/seed_42_results
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os as _os
_os.chdir(str(Path(__file__).resolve().parents[1]))

import os
import re
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix

from models.repela_net import repela_net_small
from datasets.mos2_dataset import MoS2Dataset


CLASS_NAMES = ['background', 'monolayer', 'fewlayer', 'multilayer']
CLASS_COLORS = {
    0: (128, 128, 128),  # background
    1: (0, 255, 0),      # monolayer
    2: (0, 0, 255),      # fewlayer
    3: (255, 165, 0),    # multilayer
}


# ─── Log Parsing ─────────────────────────────────────────────────────

def parse_train_log(log_path):
    """Parse train.log to extract per-epoch metrics."""
    epochs = []
    train_loss, val_loss = [], []
    train_miou, val_miou = [], []
    val_f1 = []
    lr_list = []
    per_class_iou = {c: [] for c in CLASS_NAMES}

    current_epoch = {}

    with open(log_path) as f:
        for line in f:
            line = line.strip()

            # Epoch header
            m = re.search(r'Epoch \[(\d+)/\d+\] LR=([\d.]+)', line)
            if m:
                if current_epoch:
                    epochs.append(current_epoch)
                current_epoch = {
                    'epoch': int(m.group(1)),
                    'lr': float(m.group(2)),
                }
                continue

            # Train metrics
            m = re.search(r'Train\s+Loss=([\d.]+)\s+mIoU=([\d.]+)', line)
            if m and current_epoch:
                current_epoch['train_loss'] = float(m.group(1))
                current_epoch['train_miou'] = float(m.group(2))
                continue

            # Val metrics
            m = re.search(r'Val\s+Loss=([\d.]+).*mIoU=([\d.]+)\s+F1=([\d.]+)', line)
            if m and current_epoch:
                current_epoch['val_loss'] = float(m.group(1))
                current_epoch['val_miou'] = float(m.group(2))
                current_epoch['val_f1'] = float(m.group(3))
                continue

            # Per-class IoU
            m = re.search(
                r'IoU: background: ([\d.]+) \| monolayer: ([\d.]+) '
                r'\| fewlayer: ([\d.]+) \| multilayer: ([\d.]+)', line)
            if m and current_epoch:
                current_epoch['iou_background'] = float(m.group(1))
                current_epoch['iou_monolayer'] = float(m.group(2))
                current_epoch['iou_fewlayer'] = float(m.group(3))
                current_epoch['iou_multilayer'] = float(m.group(4))
                continue

    if current_epoch:
        epochs.append(current_epoch)

    return epochs


# ─── Plot Functions ──────────────────────────────────────────────────

def plot_training_curves(epochs, output_dir):
    """Plot loss and mIoU curves."""
    ep = [e['epoch'] for e in epochs]
    train_loss = [e.get('train_loss', 0) for e in epochs]
    val_loss = [e.get('val_loss', 0) for e in epochs]
    train_miou = [e.get('train_miou', 0) for e in epochs]
    val_miou = [e.get('val_miou', 0) for e in epochs]
    lr = [e.get('lr', 0) for e in epochs]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(ep, train_loss, 'b-', linewidth=1.5, label='Train')
    axes[0].plot(ep, val_loss, 'r-', linewidth=1.5, label='Val')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # mIoU
    axes[1].plot(ep, train_miou, 'b-', linewidth=1.5, label='Train')
    axes[1].plot(ep, val_miou, 'r-', linewidth=1.5, label='Val')
    best_idx = np.argmax(val_miou)
    axes[1].scatter([ep[best_idx]], [val_miou[best_idx]], c='red', s=100,
                    zorder=5, marker='*')
    axes[1].annotate(f'Best: {val_miou[best_idx]:.4f}\n(Epoch {ep[best_idx]})',
                     xy=(ep[best_idx], val_miou[best_idx]),
                     xytext=(ep[best_idx] - 30, val_miou[best_idx] - 0.05),
                     fontsize=10, fontweight='bold', color='red',
                     arrowprops=dict(arrowstyle='->', color='red'))
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('mIoU', fontsize=12)
    axes[1].set_title('Training & Validation mIoU', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.0)

    # LR
    axes[2].plot(ep, lr, 'g-', linewidth=1.5)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Learning Rate', fontsize=12)
    axes[2].set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'training_curves.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def plot_per_class_iou(epochs, output_dir):
    """Plot per-class IoU over epochs."""
    ep = [e['epoch'] for e in epochs]
    colors = ['#808080', '#00AA00', '#0066CC', '#FF8C00']

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, cls in enumerate(CLASS_NAMES):
        iou = [e.get(f'iou_{cls}', 0) for e in epochs]
        ax.plot(ep, iou, color=colors[i], linewidth=1.5, label=cls)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('IoU', fontsize=12)
    ax.set_title('Per-Class IoU over Epochs', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    path = os.path.join(output_dir, 'per_class_iou.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def plot_confusion_matrix(cm, output_dir, normalize=True):
    """Plot confusion matrix."""
    if normalize:
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    else:
        cm_norm = cm.astype(float)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046)

    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            val = cm_norm[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=11, color=color, fontweight='bold')

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, fontsize=10, rotation=30)
    ax.set_yticklabels(CLASS_NAMES, fontsize=10)
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('True', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix (Normalized)', fontsize=13, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(output_dir, 'confusion_matrix.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def mask_to_color(mask):
    """Convert class mask to RGB."""
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, rgb in CLASS_COLORS.items():
        color[mask == cls_id] = rgb
    return color


# ─── Inference & Confusion Matrix ────────────────────────────────────

def run_inference(model, image_names, image_dir, mask_dir, device, output_dir):
    """Run inference on images, save prediction grid and compute confusion matrix."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    all_preds = []
    all_gts = []
    results = []

    for name in image_names:
        img_path = os.path.join(image_dir, f'{name}.jpg')
        mask_path = os.path.join(mask_dir, f'{name}.png')
        if not os.path.exists(img_path):
            continue

        # Load
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        H, W = img_rgb.shape[:2]

        # Preprocess
        img_float = img_rgb.astype(np.float32) / 255.0
        img_norm = (img_float - mean) / std
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0)

        # Pad
        pad_h = (32 - H % 32) % 32
        pad_w = (32 - W % 32) % 32
        if pad_h > 0 or pad_w > 0:
            img_tensor = F.pad(img_tensor, [0, pad_w, 0, pad_h], mode='reflect')

        # Predict
        with torch.no_grad():
            output = model(img_tensor.to(device))
        logits = output[0] if isinstance(output, tuple) else output
        logits = logits[:, :, :H, :W]
        pred = logits.argmax(dim=1)[0].cpu().numpy()

        results.append({
            'name': name,
            'image': img_rgb,
            'mask': mask,
            'pred': pred,
        })

        # For confusion matrix
        all_preds.append(pred.flatten())
        all_gts.append(mask.flatten())

    # Confusion matrix
    all_preds = np.concatenate(all_preds)
    all_gts = np.concatenate(all_gts)
    cm = confusion_matrix(all_gts, all_preds, labels=list(range(4)))

    return results, cm


def plot_inference_grid(results, output_dir, cols=4):
    """Plot inference results: Original / Prediction / Ground Truth."""
    n = len(results)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols * 3, figsize=(cols * 9, rows * 3.2))
    if rows == 1:
        axes = axes[np.newaxis, :]

    for idx, res in enumerate(results):
        r = idx // cols
        c = (idx % cols) * 3

        # Original
        axes[r, c].imshow(res['image'])
        axes[r, c].axis('off')

        # Prediction
        axes[r, c + 1].imshow(mask_to_color(res['pred']))
        axes[r, c + 1].axis('off')

        # Ground Truth
        axes[r, c + 2].imshow(mask_to_color(res['mask']))
        axes[r, c + 2].axis('off')

    # Hide unused
    for idx in range(n, rows * cols):
        r = idx // cols
        c = (idx % cols) * 3
        for k in range(3):
            axes[r, c + k].axis('off')

    fig.subplots_adjust(wspace=0.02, hspace=0.05, left=0, right=1, top=1, bottom=0)
    path = os.path.join(output_dir, 'inference_grid.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f'Saved: {path}')


def plot_individual_inference(results, output_dir):
    """Save individual inference comparison: original / pred / gt per image."""
    indiv_dir = os.path.join(output_dir, 'inference_individual')
    os.makedirs(indiv_dir, exist_ok=True)

    for res in results:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.subplots_adjust(wspace=0.02, left=0, right=1, top=1, bottom=0)

        axes[0].imshow(res['image'])
        axes[0].axis('off')
        axes[1].imshow(mask_to_color(res['pred']))
        axes[1].axis('off')
        axes[2].imshow(mask_to_color(res['mask']))
        axes[2].axis('off')

        path = os.path.join(indiv_dir, f'{res["name"]}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0.01)
        plt.close(fig)

    print(f'Saved {len(results)} individual inference images to: {indiv_dir}/')


# ─── Main ────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description='Comprehensive visualization')
    parser.add_argument('--log_file', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--image_dir', type=str, default='Mos2_data/ori/MoS2')
    parser.add_argument('--mask_dir', type=str, default='Mos2_data/mask')
    parser.add_argument('--split', type=str, default='test',
                        help='Which split to run inference on')
    parser.add_argument('--output', type=str,
                        default='output/visualization/seed_42_results')
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.output, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Parse log & plot training curves
    print('=== Parsing training log ===')
    epochs = parse_train_log(args.log_file)
    print(f'Parsed {len(epochs)} epochs')

    plot_training_curves(epochs, args.output)
    plot_per_class_iou(epochs, args.output)

    # 2. Load model
    print('\n=== Loading model ===')
    model = repela_net_small(num_classes=4, deep_supervision=False, use_cse=False)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    model = model.to(device)
    model.eval()
    print(f'Loaded from {args.checkpoint} (best mIoU={ckpt.get("best_miou", "N/A")})')

    # 3. Get image list
    with open(f'splits/{args.split}.txt') as f:
        image_names = [l.strip() for l in f if l.strip()]
    print(f'\n=== Running inference on {len(image_names)} {args.split} images ===')

    # 4. Run inference & get confusion matrix
    results, cm = run_inference(model, image_names, args.image_dir,
                                args.mask_dir, device, args.output)

    # 5. Plot confusion matrix
    plot_confusion_matrix(cm, args.output)

    # 6. Plot inference grid
    plot_inference_grid(results, args.output, cols=4)

    # 7. Save individual inference images
    plot_individual_inference(results, args.output)

    # 8. Print summary
    print(f'\n=== Summary ===')
    best_epoch = epochs[np.argmax([e.get('val_miou', 0) for e in epochs])]
    print(f'Best Val mIoU: {best_epoch.get("val_miou", 0):.4f} (Epoch {best_epoch["epoch"]})')
    print(f'  background: {best_epoch.get("iou_background", 0):.4f}')
    print(f'  monolayer:  {best_epoch.get("iou_monolayer", 0):.4f}')
    print(f'  fewlayer:   {best_epoch.get("iou_fewlayer", 0):.4f}')
    print(f'  multilayer: {best_epoch.get("iou_multilayer", 0):.4f}')
    print(f'\nAll outputs saved to: {args.output}/')


if __name__ == '__main__':
    main()
