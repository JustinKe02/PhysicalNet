"""
V4 Experiment Results Visualization.

Parses nohup_v4_experiments.log and generates:
  1. Training curves (loss, mIoU, per-class IoU, LR) for both experiments
  2. Side-by-side comparison dashboard
  3. Model inference on val set with best model

Usage:
    python plot_v4_results.py              # plots only
    python plot_v4_results.py --inference  # plots + inference
"""

import os
import re
import argparse
import numpy as np
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLORS = {
    'train': '#2196F3', 'val': '#F44336',
    'exp1': '#2196F3', 'exp2': '#FF9800',
    'bg': '#607D8B', 'mono': '#E91E63',
    'few': '#4CAF50', 'multi': '#3F51B5',
    'lr': '#795548',
}


# ── Log Parsing ───────────────────────────────────────────────────────────────
def parse_nohup_log(log_path):
    """Parse nohup_v4_experiments.log → list of two experiment dicts."""
    with open(log_path, 'r') as f:
        content = f.read()

    # Split on experiment markers
    parts = re.split(r'=== \[\d/\d\]', content)
    experiments = []

    for part in parts[1:]:  # skip header
        data = {
            'epoch': [], 'lr': [],
            'train_loss': [], 'train_miou': [],
            'val_loss': [], 'val_ce': [], 'val_dice': [],
            'val_miou': [], 'val_f1': [],
            'iou_bg': [], 'iou_mono': [], 'iou_few': [], 'iou_multi': [],
        }

        for line in part.split('\n'):
            # Epoch + LR
            m = re.search(r'Epoch \[(\d+)/\d+\] LR=([\d.]+)', line)
            if m:
                cur_epoch = int(m.group(1))
                cur_lr = float(m.group(2))
                continue

            # Train line: "Train  Loss=X.XXXX mIoU=X.XXXX"
            # or "Train  Loss=X.XXXX (CE=X.XXXX Dice=X.XXXX) mIoU=X.XXXX"
            m = re.search(r'Train\s+Loss=([\d.]+).*?mIoU=([\d.]+)', line)
            if m:
                data['train_loss'].append(float(m.group(1)))
                data['train_miou'].append(float(m.group(2)))
                data['epoch'].append(cur_epoch)
                data['lr'].append(cur_lr)
                continue

            # Val line: "Val    Loss=X.XXXX (CE=X.XXXX Dice=X.XXXX) mIoU=X.XXXX F1=X.XXXX"
            m = re.search(
                r'Val\s+Loss=([\d.]+)\s+\(CE=([\d.]+)\s+Dice=([\d.]+)\)\s+'
                r'mIoU=([\d.]+)\s+F1=([\d.]+)', line)
            if m:
                data['val_loss'].append(float(m.group(1)))
                data['val_ce'].append(float(m.group(2)))
                data['val_dice'].append(float(m.group(3)))
                data['val_miou'].append(float(m.group(4)))
                data['val_f1'].append(float(m.group(5)))
                continue

            # IoU line
            m = re.search(
                r'IoU: background: ([\d.]+) \| monolayer: ([\d.]+) '
                r'\| fewlayer: ([\d.]+) \| multilayer: ([\d.]+)', line)
            if m:
                data['iou_bg'].append(float(m.group(1)))
                data['iou_mono'].append(float(m.group(2)))
                data['iou_few'].append(float(m.group(3)))
                data['iou_multi'].append(float(m.group(4)))
                continue

        # Truncate to min length
        min_len = min(len(v) for v in data.values() if len(v) > 0)
        for k in data:
            data[k] = data[k][:min_len]

        if min_len > 0:
            experiments.append(data)

    return experiments


def smooth(values, weight=0.85):
    s = []
    last = values[0]
    for v in values:
        last = weight * last + (1 - weight) * v
        s.append(last)
    return s


# ── Plot Functions ────────────────────────────────────────────────────────────

def plot_comparison_dashboard(exp1, exp2, output_dir):
    """6-panel comparison dashboard."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))

    labels = ['Exp1: Old Config (Focal α)', 'Exp2: New Config (DS+CopyPaste)']
    colors = [COLORS['exp1'], COLORS['exp2']]

    # ── (0,0) Val mIoU ──
    ax = axes[0, 0]
    for d, lab, col in zip([exp1, exp2], labels, colors):
        ax.plot(d['epoch'], d['val_miou'], alpha=0.15, color=col)
        ax.plot(d['epoch'], smooth(d['val_miou']), color=col, lw=2, label=lab)
        bi = np.argmax(d['val_miou'])
        ax.scatter([d['epoch'][bi]], [d['val_miou'][bi]], s=120, c='gold',
                   edgecolors='black', zorder=5, marker='*')
        ax.annotate(f"{d['val_miou'][bi]:.4f}\n(E{d['epoch'][bi]})",
                    xy=(d['epoch'][bi], d['val_miou'][bi]),
                    xytext=(d['epoch'][bi]+8, d['val_miou'][bi]-0.04),
                    fontsize=9, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='black', lw=0.8))
    ax.set_title('Val mIoU Comparison')
    ax.set_ylabel('mIoU')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_ylim(0.3, 0.90)

    # ── (0,1) Train Loss ──
    ax = axes[0, 1]
    for d, lab, col in zip([exp1, exp2], labels, colors):
        ax.plot(d['epoch'], d['train_loss'], alpha=0.15, color=col)
        ax.plot(d['epoch'], smooth(d['train_loss']), color=col, lw=2, label=lab)
    ax.set_title('Train Loss')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 2.5)

    # ── (0,2) Val Loss ──
    ax = axes[0, 2]
    for d, lab, col in zip([exp1, exp2], labels, colors):
        ax.plot(d['epoch'], smooth(d['val_loss']), color=col, lw=2, label=lab)
    ax.set_title('Val Loss')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.5)

    # ── (1,0) Per-class IoU – Exp1 ──
    ax = axes[1, 0]
    for key, name, col in [('iou_bg','Background',COLORS['bg']),
                            ('iou_mono','Monolayer',COLORS['mono']),
                            ('iou_few','Fewlayer',COLORS['few']),
                            ('iou_multi','Multilayer',COLORS['multi'])]:
        ax.plot(exp1['epoch'], smooth(exp1[key]), color=col, lw=2, label=name)
    ax.set_title('Per-Class IoU – Exp1 (Old Config)')
    ax.set_ylabel('IoU')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

    # ── (1,1) Per-class IoU – Exp2 ──
    ax = axes[1, 1]
    for key, name, col in [('iou_bg','Background',COLORS['bg']),
                            ('iou_mono','Monolayer',COLORS['mono']),
                            ('iou_few','Fewlayer',COLORS['few']),
                            ('iou_multi','Multilayer',COLORS['multi'])]:
        ax.plot(exp2['epoch'], smooth(exp2[key]), color=col, lw=2, label=name)
    ax.set_title('Per-Class IoU – Exp2 (New Config)')
    ax.set_ylabel('IoU')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

    # ── (1,2) LR Schedule ──
    ax = axes[1, 2]
    ax.plot(exp1['epoch'], exp1['lr'], color=COLORS['lr'], lw=2)
    ax.set_title('Learning Rate Schedule')
    ax.set_ylabel('LR')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-4, -4))

    for ax in axes.flat:
        ax.set_xlabel('Epoch')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle('V4 Experiment Comparison Dashboard — RepELA-Net (no_color)',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, 'v4_comparison_dashboard.png')
    plt.savefig(path)
    plt.close()
    print(f'  ✓ {path}')


def plot_single_experiment(data, name, output_dir):
    """4-panel dashboard for a single experiment."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = data['epoch']

    # mIoU
    ax = axes[0, 0]
    ax.plot(epochs, data['train_miou'], alpha=0.15, color=COLORS['train'])
    ax.plot(epochs, smooth(data['train_miou']), color=COLORS['train'], lw=2, label='Train')
    ax.plot(epochs, data['val_miou'], alpha=0.15, color=COLORS['val'])
    ax.plot(epochs, smooth(data['val_miou']), color=COLORS['val'], lw=2, label='Val')
    bi = np.argmax(data['val_miou'])
    ax.scatter([epochs[bi]], [data['val_miou'][bi]], s=120, c='gold',
               edgecolors='black', zorder=5, marker='*')
    ax.annotate(f"Best: {data['val_miou'][bi]:.4f} (E{epochs[bi]})",
                xy=(epochs[bi], data['val_miou'][bi]),
                xytext=(epochs[bi]+8, data['val_miou'][bi]-0.06),
                fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black'))
    ax.set_title('mIoU')
    ax.set_ylabel('mIoU')
    ax.legend()
    ax.set_ylim(0, 1)

    # Loss
    ax = axes[0, 1]
    ax.plot(epochs, data['train_loss'], alpha=0.15, color=COLORS['train'])
    ax.plot(epochs, smooth(data['train_loss']), color=COLORS['train'], lw=2, label='Train')
    ax.plot(epochs, smooth(data['val_loss']), color=COLORS['val'], lw=2, label='Val')
    ax.set_title('Loss')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.set_ylim(0)

    # Per-class IoU
    ax = axes[1, 0]
    for key, lbl, col in [('iou_bg','Background',COLORS['bg']),
                           ('iou_mono','Monolayer',COLORS['mono']),
                           ('iou_few','Fewlayer',COLORS['few']),
                           ('iou_multi','Multilayer',COLORS['multi'])]:
        ax.plot(epochs, data[key], alpha=0.15, color=col)
        ax.plot(epochs, smooth(data[key]), color=col, lw=2, label=lbl)
    ax.set_title('Per-Class IoU (Val)')
    ax.set_ylabel('IoU')
    ax.legend()
    ax.set_ylim(0, 1.05)

    # Val F1
    ax = axes[1, 1]
    ax.plot(epochs, data['val_f1'], alpha=0.15, color='#FF5722')
    ax.plot(epochs, smooth(data['val_f1']), color='#FF5722', lw=2, label='Val F1')
    ax.set_title('Val F1 Score')
    ax.set_ylabel('F1')
    ax.legend()
    ax.set_ylim(0, 1)

    for ax in axes.flat:
        ax.set_xlabel('Epoch')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle(f'{name}', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    fname = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    path = os.path.join(output_dir, f'{fname}.png')
    plt.savefig(path)
    plt.close()
    print(f'  ✓ {path}')


def plot_best_iou_bar(exp1, exp2, output_dir):
    """Bar chart comparing best-epoch per-class IoU."""
    bi1 = np.argmax(exp1['val_miou'])
    bi2 = np.argmax(exp2['val_miou'])

    classes = ['Background', 'Monolayer', 'Fewlayer', 'Multilayer', 'mIoU']
    vals1 = [exp1['iou_bg'][bi1], exp1['iou_mono'][bi1],
             exp1['iou_few'][bi1], exp1['iou_multi'][bi1], exp1['val_miou'][bi1]]
    vals2 = [exp2['iou_bg'][bi2], exp2['iou_mono'][bi2],
             exp2['iou_few'][bi2], exp2['iou_multi'][bi2], exp2['val_miou'][bi2]]

    x = np.arange(len(classes))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - w/2, vals1, w, label=f'Exp1 Old (E{exp1["epoch"][bi1]})',
                   color=COLORS['exp1'], alpha=0.85, edgecolor='white')
    bars2 = ax.bar(x + w/2, vals2, w, label=f'Exp2 New (E{exp2["epoch"][bi2]})',
                   color=COLORS['exp2'], alpha=0.85, edgecolor='white')

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.008,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_ylabel('IoU', fontsize=13)
    ax.set_title('Best-Epoch Per-Class IoU Comparison', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'v4_best_iou_comparison.png')
    plt.savefig(path)
    plt.close()
    print(f'  ✓ {path}')


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(checkpoint_path, output_dir, data_root='./Mos2_data',
                  split_dir='splits/', split='val', deep_supervision=False):
    """Run inference on val set with best model and save visualizations."""
    import torch
    import torch.nn.functional as F
    import torchvision.transforms.functional as TF
    from PIL import Image
    from matplotlib.patches import Patch
    from train_ablation import build_ablation_model

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]
    CLASS_COLORS = np.array([
        [0,0,0], [239,41,41], [0,170,0], [114,159,207]
    ], dtype=np.uint8)
    CLASSES = ['Background', 'Monolayer', 'Fewlayer', 'Multilayer']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Use correct no_color ablation model (ZeroPadChannel instead of ColorSpaceEnhancement)
    model = build_ablation_model('no_color', num_classes=4,
                                  deep_supervision=deep_supervision).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(sd, strict=True)
    model.eval()

    epoch_info = ckpt.get('epoch', -1) + 1 if isinstance(ckpt, dict) and 'epoch' in ckpt else '?'
    miou_info = f"{ckpt.get('best_miou', 0):.4f}" if isinstance(ckpt, dict) and 'best_miou' in ckpt else '?'
    print(f'  Model loaded: {checkpoint_path}')
    print(f'  Epoch {epoch_info}, best mIoU={miou_info}')

    # Read split
    split_file = os.path.join(split_dir, f'{split}.txt')
    with open(split_file) as f:
        basenames = [l.strip() for l in f if l.strip()]
    img_dir = os.path.join(data_root, 'ori', 'MoS2')
    mask_dir = os.path.join(data_root, 'mask')

    os.makedirs(output_dir, exist_ok=True)

    crop_size, stride = 512, 384

    def sliding_window_predict(img_tensor):
        _, H, W = img_tensor.shape
        pred_sum = torch.zeros(4, H, W, dtype=torch.float32, device=device)
        count = torch.zeros(H, W, dtype=torch.float32, device=device)
        pad_h, pad_w = max(0, crop_size-H), max(0, crop_size-W)
        padded = F.pad(img_tensor, [0,pad_w,0,pad_h], mode='reflect') if (pad_h>0 or pad_w>0) else img_tensor
        _, pH, pW = padded.shape
        ys = sorted(set(list(range(0,max(1,pH-crop_size+1),stride))+[max(0,pH-crop_size)]))
        xs = sorted(set(list(range(0,max(1,pW-crop_size+1),stride))+[max(0,pW-crop_size)]))
        for y in ys:
            for x in xs:
                crop = padded[:,y:y+crop_size,x:x+crop_size].unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model(crop)
                    logits = out[0] if isinstance(out, tuple) else out
                    probs = F.softmax(logits, dim=1)[0]
                ye, xe = min(y+crop_size,H), min(x+crop_size,W)
                pred_sum[:,y:ye,x:xe] += probs[:,:ye-y,:xe-x]
                count[y:ye,x:xe] += 1
        return (pred_sum / count.clamp(min=1).unsqueeze(0)).argmax(dim=0).cpu().numpy()

    def colorize(mask):
        out = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for c, col in enumerate(CLASS_COLORS):
            out[mask == c] = col
        return out

    for i, bn in enumerate(basenames):
        img_path = os.path.join(img_dir, f'{bn}.jpg')
        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir, f'{bn}.png')
        if not os.path.exists(img_path):
            continue

        img_pil = Image.open(img_path).convert('RGB')
        img_np = np.array(img_pil)
        img_tensor = TF.normalize(TF.to_tensor(img_pil), MEAN, STD)
        pred = sliding_window_predict(img_tensor)

        # GT
        gt_path = os.path.join(mask_dir, f'{bn}.png')
        gt_mask = np.array(Image.open(gt_path)) if os.path.exists(gt_path) else None

        # Figure
        n_cols = 4 if gt_mask is not None else 3
        fig, axes = plt.subplots(1, n_cols, figsize=(6*n_cols, 6))

        axes[0].imshow(img_np)
        axes[0].set_title('Original', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        pred_color = colorize(pred)
        axes[1].imshow(pred_color)
        axes[1].set_title('Prediction (mask)', fontsize=14, fontweight='bold')
        axes[1].axis('off')

        overlay = img_np.copy()
        fg = pred > 0
        overlay[fg] = (0.55*img_np[fg] + 0.45*pred_color[fg]).astype(np.uint8)
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay', fontsize=14, fontweight='bold')
        axes[2].axis('off')

        if gt_mask is not None:
            gt_color = colorize(gt_mask)
            gt_overlay = img_np.copy()
            fg_gt = gt_mask > 0
            gt_overlay[fg_gt] = (0.55*img_np[fg_gt] + 0.45*gt_color[fg_gt]).astype(np.uint8)
            axes[3].imshow(gt_overlay)
            axes[3].set_title('Ground Truth', fontsize=14, fontweight='bold')
            axes[3].axis('off')

        legend = [Patch(facecolor=np.array(c)/255., label=n)
                  for n, c in zip(CLASSES[1:], CLASS_COLORS[1:])]
        fig.legend(handles=legend, loc='lower center', ncol=3, fontsize=11,
                   bbox_to_anchor=(0.5, -0.02))
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'{bn}_result.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f'  [{i+1}/{len(basenames)}] {bn}')

    print(f'  Inference results saved to: {output_dir}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='V4 experiment results visualization')
    parser.add_argument('--log', default='nohup_v4_experiments.log')
    parser.add_argument('--output', default='outputv4_plots/')
    parser.add_argument('--inference', action='store_true',
                        help='Also run inference with best model on val set')
    parser.add_argument('--best_model',
                        default='outputv4_oldcfg/nocolor_oldcfg_20260322_233322/best_model.pth')
    parser.add_argument('--best_model2',
                        default='outputv4/ablation_no_color_20260323_005458/best_model.pth')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ── Parse ──
    print(f'Parsing: {args.log}')
    experiments = parse_nohup_log(args.log)
    print(f'Found {len(experiments)} experiments')

    for i, exp in enumerate(experiments):
        n = len(exp['epoch'])
        bi = np.argmax(exp['val_miou'])
        print(f'  Exp{i+1}: {n} epochs, best mIoU={exp["val_miou"][bi]:.4f} (E{exp["epoch"][bi]})')

    # ── Plot ──
    print('\nGenerating plots...')
    if len(experiments) >= 1:
        plot_single_experiment(experiments[0],
            'Exp1 — Old Config (Focal α, no DS, no CopyPaste)', args.output)
    if len(experiments) >= 2:
        plot_single_experiment(experiments[1],
            'Exp2 — New Config (DS + CopyPaste)', args.output)
    if len(experiments) >= 2:
        plot_comparison_dashboard(experiments[0], experiments[1], args.output)
        plot_best_iou_bar(experiments[0], experiments[1], args.output)

    # ── Inference ──
    if args.inference:
        print('\n=== Inference (Exp1 best model — no DS) ===')
        inf_dir1 = os.path.join(args.output, 'inference_exp1')
        run_inference(args.best_model, inf_dir1, deep_supervision=False)

        print('\n=== Inference (Exp2 best model — with DS) ===')
        inf_dir2 = os.path.join(args.output, 'inference_exp2')
        run_inference(args.best_model2, inf_dir2, deep_supervision=True)

    print(f'\n✅ All done! Results in: {args.output}')


if __name__ == '__main__':
    main()
