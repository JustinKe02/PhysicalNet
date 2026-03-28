"""
Transfer Learning Visualization Script.
Generates: training curves, inference grids, confusion matrices.
"""
import sys, os, re, glob
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from models.repela_net import RepELANet
from utils.metrics import SegmentationMetrics
from transfer.material_dataset import compute_dataset_stats

# ── Config ────────────────────────────────────────────────────────
EXPERIMENTS = {
    'ws2': {
        'data_root': 'other data/WS2_data',
        'eval_split': 'test',
        'num_classes': 4,
        'class_names': ['BG', '1L', 'FL', 'ML'],
        'colors': np.array([[0,0,0],[255,0,0],[0,255,0],[0,0,255]], dtype=np.uint8),
        'models': {
            'Scratch': 'output/finetune_ws2_scratch/best_model.pth',
            'FT+reset_head': 'output/finetune_ws2_r2_resethead/best_model.pth',
        },
        'logs': {
            'Scratch': 'output/finetune_ws2_scratch/finetune.log',
            'FT+reset_head': 'output/finetune_ws2_r2_resethead/finetune.log',
        },
    },
    'graphene': {
        'data_root': 'other data/graphene',
        'eval_split': 'val',
        'num_classes': 3,
        'class_names': ['BG', '1L', '>1L'],
        'colors': np.array([[0,0,0],[255,0,0],[0,255,0]], dtype=np.uint8),
        'models': {
            'Scratch': 'output/finetune_graphene_scratch/best_model.pth',
            'FT partial(1+2)': 'output/finetune_graphene_r3_partial/best_model.pth',
        },
        'logs': {
            'Scratch': 'output/finetune_graphene_scratch/finetune.log',
            'FT partial(1+2)': 'output/finetune_graphene_r3_partial/finetune.log',
        },
    },
}

OUT_DIR = 'output/transfer_vis'
os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Log Parsing ───────────────────────────────────────────────────
def parse_log(log_path):
    """Extract epoch-level train_loss, val_mIoU, per-class IoU from log."""
    epochs, train_losses, val_mious, lrs = [], [], [], []
    per_class = []
    with open(log_path) as f:
        lines = f.readlines()
    epoch = None
    for line in lines:
        m = re.search(r'Epoch \[(\d+)/\d+\] LR\(dec\): ([\d.]+)', line)
        if m:
            epoch = int(m.group(1))
            lrs.append(float(m.group(2)))
        m = re.search(r'Train Loss: ([\d.]+) mIoU', line)
        if m and epoch is not None:
            train_losses.append(float(m.group(1)))
        m = re.search(r'Val mIoU: ([\d.]+)', line)
        if m and epoch is not None:
            epochs.append(epoch)
            val_mious.append(float(m.group(1)))
        m = re.findall(r'(?:C\d|BG|1L|FL|ML|>1L): ([\d.]+)', line)
        if m and 'Per-class IoU' in line and epoch is not None:
            per_class.append([float(x) for x in m])
    return {
        'epochs': epochs, 'train_loss': train_losses,
        'val_miou': val_mious, 'lr': lrs, 'per_class': per_class
    }


# ── Plot Training Curves ─────────────────────────────────────────
def plot_training_curves(dataset_name, cfg):
    """Plot loss and mIoU curves for all models in a dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors_line = ['#e74c3c', '#2ecc71', '#3498db', '#9b59b6']

    for i, (name, log_path) in enumerate(cfg['logs'].items()):
        data = parse_log(log_path)
        c = colors_line[i % len(colors_line)]
        # Loss
        axes[0].plot(data['epochs'][:len(data['train_loss'])],
                     data['train_loss'], color=c, alpha=0.8, label=name, linewidth=1.5)
        # mIoU
        axes[1].plot(data['epochs'], data['val_miou'],
                     color=c, alpha=0.8, label=name, linewidth=1.5)
        # Mark best
        best_idx = np.argmax(data['val_miou'])
        axes[1].scatter(data['epochs'][best_idx], data['val_miou'][best_idx],
                        color=c, s=80, zorder=5, edgecolors='black', linewidth=1.2)
        axes[1].annotate(f'{data["val_miou"][best_idx]:.4f}',
                         (data['epochs'][best_idx], data['val_miou'][best_idx]),
                         textcoords="offset points", xytext=(5, 8), fontsize=9,
                         fontweight='bold', color=c)

    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Train Loss')
    axes[0].set_title(f'{dataset_name} — Training Loss')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Val mIoU')
    axes[1].set_title(f'{dataset_name} — Validation mIoU')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f'{dataset_name}_training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ── Plot Per-class IoU ────────────────────────────────────────────
def plot_perclass_iou(dataset_name, cfg):
    """Bar chart comparing per-class IoU across models."""
    class_names = cfg['class_names']
    n_classes = len(class_names)
    model_names = list(cfg['logs'].keys())
    n_models = len(model_names)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(n_classes)
    width = 0.8 / n_models
    colors_bar = ['#e74c3c', '#2ecc71', '#3498db', '#9b59b6']

    for i, name in enumerate(model_names):
        data = parse_log(cfg['logs'][name])
        # Get best epoch's per-class IoU
        best_idx = np.argmax(data['val_miou'])
        if best_idx < len(data['per_class']):
            ious = data['per_class'][best_idx]
        else:
            ious = data['per_class'][-1] if data['per_class'] else [0]*n_classes
        ious = ious[:n_classes]
        bars = ax.bar(x + i*width - width*(n_models-1)/2, ious, width,
                      label=name, color=colors_bar[i], alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, ious):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x); ax.set_xticklabels(class_names, fontsize=11)
    ax.set_ylabel('IoU'); ax.set_title(f'{dataset_name} — Per-class IoU Comparison')
    ax.set_ylim(0, 1.1); ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f'{dataset_name}_perclass_iou.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ── Inference ─────────────────────────────────────────────────────
def build_model(num_classes):
    return RepELANet(num_classes=num_classes, channels=(32,64,128,256),
                     num_blocks=(2,2,4,2), num_heads=(0,0,4,8),
                     decoder_channels=128, deep_supervision=False, use_cse=False)

def sliding_window(model, img_t, crop=512, stride=384, nc=4):
    _, H, W = img_t.shape
    pred_sum = torch.zeros(nc, H, W, device=device)
    count = torch.zeros(H, W, device=device)
    pad_h = max(0, crop-H); pad_w = max(0, crop-W)
    if pad_h>0 or pad_w>0: img_t = F.pad(img_t, [0,pad_w,0,pad_h], mode='reflect')
    _, pH, pW = img_t.shape
    ys = sorted(set(list(range(0, max(1,pH-crop+1), stride)) + [max(0,pH-crop)]))
    xs = sorted(set(list(range(0, max(1,pW-crop+1), stride)) + [max(0,pW-crop)]))
    for y in ys:
        for x in xs:
            c = img_t[:, y:y+crop, x:x+crop].unsqueeze(0).to(device)
            with torch.no_grad():
                o = model(c); o = o[0] if isinstance(o,tuple) else o
                p = F.softmax(o, dim=1)[0]
            ye=min(y+crop,H); xe=min(x+crop,W)
            pred_sum[:,y:ye,x:xe]+=p[:,:ye-y,:xe-x]; count[y:ye,x:xe]+=1
    return (pred_sum/count.clamp(min=1).unsqueeze(0)).argmax(0).cpu().numpy()

def colorize(mask, colors):
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(len(colors)):
        out[mask == c] = colors[c]
    return out


# ── Inference Grid ────────────────────────────────────────────────
def plot_inference_grid(dataset_name, cfg, max_images=5):
    """Grid: [Original | GT | Model1_pred | Model2_pred] per row."""
    data_root = cfg['data_root']
    split = cfg['eval_split']
    nc = cfg['num_classes']
    colors = cfg['colors']

    stats = compute_dataset_stats(data_root, split='train')
    mean, std = stats['mean'], stats['std']

    img_dir = os.path.join(data_root, 'img_dir', split)
    ann_dir = os.path.join(data_root, 'ann_dir', split)
    imgs = sorted(glob.glob(os.path.join(img_dir, '*.jpg')) +
                  glob.glob(os.path.join(img_dir, '*.png')))[:max_images]

    model_names = list(cfg['models'].keys())
    n_models = len(model_names)
    n_cols = 2 + n_models  # Original + GT + N models

    # Load models
    models = {}
    for mname, ckpt_path in cfg['models'].items():
        m = build_model(nc)
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        m.load_state_dict(ckpt['model'], strict=False)
        m = m.to(device).eval()
        models[mname] = m

    fig, axes = plt.subplots(len(imgs), n_cols, figsize=(4*n_cols, 4*len(imgs)))
    if len(imgs) == 1:
        axes = axes[np.newaxis, :]

    for row, ip in enumerate(imgs):
        bn = os.path.splitext(os.path.basename(ip))[0]
        ap = os.path.join(ann_dir, f'{bn}.png')

        img = Image.open(ip).convert('RGB')
        gt = np.array(Image.open(ap))
        img_t = TF.normalize(TF.to_tensor(img), mean, std)

        # Original
        axes[row, 0].imshow(np.array(img))
        axes[row, 0].set_title('Original' if row == 0 else '', fontsize=11)
        axes[row, 0].axis('off')

        # GT
        axes[row, 1].imshow(colorize(gt, colors))
        axes[row, 1].set_title('Ground Truth' if row == 0 else '', fontsize=11)
        axes[row, 1].axis('off')

        # Predictions
        for col, mname in enumerate(model_names):
            pred = sliding_window(models[mname], img_t, nc=nc)
            axes[row, 2+col].imshow(colorize(pred, colors))
            axes[row, 2+col].set_title(mname if row == 0 else '', fontsize=11)
            axes[row, 2+col].axis('off')

    # Legend
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=colors[i]/255.0, label=cfg['class_names'][i])
                      for i in range(nc)]
    fig.legend(handles=legend_patches, loc='lower center', ncol=nc,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle(f'{dataset_name} — Inference Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f'{dataset_name}_inference_grid.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')

    # Cleanup
    for m in models.values():
        del m
    torch.cuda.empty_cache()


# ── Confusion Matrix ──────────────────────────────────────────────
def plot_confusion_matrices(dataset_name, cfg):
    """Side-by-side confusion matrices for all models."""
    data_root = cfg['data_root']
    split = cfg['eval_split']
    nc = cfg['num_classes']
    class_names = cfg['class_names']

    stats = compute_dataset_stats(data_root, split='train')
    mean, std = stats['mean'], stats['std']

    img_dir = os.path.join(data_root, 'img_dir', split)
    ann_dir = os.path.join(data_root, 'ann_dir', split)
    imgs = sorted(glob.glob(os.path.join(img_dir, '*.jpg')) +
                  glob.glob(os.path.join(img_dir, '*.png')))

    model_names = list(cfg['models'].keys())
    n_models = len(model_names)

    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    if n_models == 1:
        axes = [axes]

    for idx, (mname, ckpt_path) in enumerate(cfg['models'].items()):
        model = build_model(nc)
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'], strict=False)
        model = model.to(device).eval()

        metrics = SegmentationMetrics(nc)
        for ip in imgs:
            bn = os.path.splitext(os.path.basename(ip))[0]
            ap = os.path.join(ann_dir, f'{bn}.png')
            if not os.path.exists(ap): continue
            img = Image.open(ip).convert('RGB')
            gt = np.array(Image.open(ap))
            img_t = TF.normalize(TF.to_tensor(img), mean, std)
            pred = sliding_window(model, img_t, nc=nc)
            metrics.update(pred, gt)

        cm = metrics.confusion_matrix
        row_sums = np.maximum(cm.sum(axis=1, keepdims=True), 1)
        cm_norm = cm / row_sums * 100

        ax = axes[idx]
        im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues,
                       vmin=0, vmax=100)
        for i in range(nc):
            for j in range(nc):
                val = cm_norm[i, j]
                color = 'white' if val > 50 else 'black'
                ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                        fontsize=12, fontweight='bold', color=color)
        ax.set_xticks(range(nc)); ax.set_xticklabels(class_names, fontsize=10)
        ax.set_yticks(range(nc)); ax.set_yticklabels(class_names, fontsize=10)
        ax.set_xlabel('Predicted', fontsize=11)
        if idx == 0: ax.set_ylabel('Ground Truth', fontsize=11)
        ax.set_title(mname, fontsize=12, fontweight='bold')

        del model; torch.cuda.empty_cache()

    plt.suptitle(f'{dataset_name} — Confusion Matrices', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f'{dataset_name}_confusion_matrices.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ── Main ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    for ds_name, cfg in EXPERIMENTS.items():
        print(f'\n{"="*60}')
        print(f'  {ds_name.upper()} Visualizations')
        print(f'{"="*60}')

        print('\n  Training curves...')
        plot_training_curves(ds_name.upper(), cfg)

        print('  Per-class IoU...')
        plot_perclass_iou(ds_name.upper(), cfg)

        print('  Inference grid...')
        plot_inference_grid(ds_name.upper(), cfg, max_images=5)

        print('  Confusion matrices...')
        plot_confusion_matrices(ds_name.upper(), cfg)

    print(f'\nAll visualizations saved to: {OUT_DIR}/')
