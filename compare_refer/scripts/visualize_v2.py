"""
V2 Filtered Supplementary — Training Curves + Inference Visualization
"""
import os, sys, re, glob, random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image
import torch, torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms as T

sys.path.insert(0, '/root/autodl-tmp/PhysicalNet')
from models.repela_net import RepELANet
from utils.metrics import SegmentationMetrics

OUT_DIR = '/root/autodl-tmp/PhysicalNet/output/transfer_vis'
os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Parse finetune.log ──
def parse_log(path):
    epochs, mious = [], []
    cur_ep = 0
    with open(path) as f:
        for line in f:
            m = re.search(r'Epoch \[(\d+)/\d+\]', line)
            if m: cur_ep = int(m.group(1))
            m = re.search(r'Val mIoU:\s*([\d.]+)', line)
            if m:
                epochs.append(cur_ep)
                mious.append(float(m.group(1)))
    return epochs, mious

# ══════════════════════════════════════════════════════════
# 1. Training curves: v1 vs v2 side-by-side
# ══════════════════════════════════════════════════════════
def plot_v2_training_curves():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # WS2_supp
    ax = axes[0]
    ax.set_title('WS2_supp: v1 vs v2', fontweight='bold', fontsize=12)
    configs = [
        ('finetune_ws2supp_scratch', 'v1 Scratch', '#90CAF9', '--'),
        ('finetune_ws2supp_ft_resethead', 'v1 FT+reset', '#EF9A9A', '--'),
        ('finetune_ws2supp_v2_scratch', 'v2 Scratch', '#1565C0', '-'),
        ('finetune_ws2supp_v2_ft_resethead', 'v2 FT+reset', '#C62828', '-'),
    ]
    for name, label, color, ls in configs:
        log = f'output/{name}/finetune.log'
        if not os.path.exists(log): continue
        ep, miou = parse_log(log)
        if ep: ax.plot(ep, miou, color=color, linestyle=ls, label=label, linewidth=1.8)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Val mIoU')
    ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Gr_supp
    ax = axes[1]
    ax.set_title('Gr_supp: v1 vs v2', fontweight='bold', fontsize=12)
    configs = [
        ('finetune_grsupp_scratch', 'v1 Scratch', '#A5D6A7', '--'),
        ('finetune_grsupp_ft_resethead', 'v1 FT+reset', '#FFAB91', '--'),
        ('finetune_grsupp_v2_scratch', 'v2 Scratch', '#2E7D32', '-'),
        ('finetune_grsupp_v2_ft_resethead', 'v2 FT+reset', '#D84315', '-'),
    ]
    for name, label, color, ls in configs:
        log = f'output/{name}/finetune.log'
        if not os.path.exists(log): continue
        ep, miou = parse_log(log)
        if ep: ax.plot(ep, miou, color=color, linestyle=ls, label=label, linewidth=1.8)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Val mIoU')
    ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # WS2_combined
    ax = axes[2]
    ax.set_title('WS2_combined: v1 vs v2', fontweight='bold', fontsize=12)
    configs = [
        ('finetune_ws2combined_scratch', 'v1 Scratch', '#CE93D8', '--'),
        ('finetune_ws2combined_ft_resethead', 'v1 FT+reset', '#FFCC80', '--'),
        ('finetune_ws2combined_v2_scratch', 'v2 Scratch', '#6A1B9A', '-'),
        ('finetune_ws2combined_v2_ft_resethead', 'v2 FT+reset', '#E65100', '-'),
    ]
    for name, label, color, ls in configs:
        log = f'output/{name}/finetune.log'
        if not os.path.exists(log): continue
        ep, miou = parse_log(log)
        if ep: ax.plot(ep, miou, color=color, linestyle=ls, label=label, linewidth=1.8)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Val mIoU')
    ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.suptitle('Data Filtering Effect — v1 (dashed) vs v2 (solid)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = f'{OUT_DIR}/v2_training_curves.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')

# ══════════════════════════════════════════════════════════
# 2. v1 vs v2 comparison bar chart
# ══════════════════════════════════════════════════════════
def plot_v1v2_comparison():
    labels = ['WS2s\nScratch', 'WS2s\nFT+reset', 'Grs\nScratch', 'Grs\nFT+reset',
              'Comb\nScratch', 'Comb\nFT+reset']
    v1 = [70.87, 64.82, 71.96, 66.44, 86.06, 89.69]
    v2 = [75.07, 81.21, 77.79, 77.85, 88.04, 89.65]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - w/2, v1, w, label='v1 (unfiltered)', color='#BBDEFB', edgecolor='#1565C0', linewidth=1)
    bars2 = ax.bar(x + w/2, v2, w, label='v2 (filtered)', color='#1565C0', edgecolor='#0D47A1', linewidth=1)

    for bar, val in zip(bars1, v1):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8, color='#666')
    for bar, val in zip(bars2, v2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Highlight negative→positive reversal
    ax.annotate('−6.05→+6.14', xy=(1, 81.21), fontsize=9, fontweight='bold',
                color='#C62828', ha='center',
                xytext=(1, 87), arrowprops=dict(arrowstyle='->', color='#C62828'))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Val mIoU (%)')
    ax.set_ylim(55, 98)
    ax.set_title('Data Filtering Impact — v1 vs v2 Comparison', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out = f'{OUT_DIR}/v1_vs_v2_bar.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')

# ══════════════════════════════════════════════════════════
# 3. Inference Grid for key v2 experiments
# ══════════════════════════════════════════════════════════
def build_model(nc=4, use_cse=False):
    return RepELANet(num_classes=nc, channels=(32,64,128,256),
                     num_blocks=(2,2,4,2), num_heads=(0,0,4,8),
                     decoder_channels=128, deep_supervision=False, use_cse=use_cse)

def sliding_window(model, img_t, crop=512, stride=384, nc=4):
    _, H, W = img_t.shape
    pred_sum = torch.zeros(nc, H, W, device=device)
    count = torch.zeros(H, W, device=device)
    pad_h = max(0, crop - H); pad_w = max(0, crop - W)
    if pad_h > 0 or pad_w > 0:
        img_t = F.pad(img_t, [0, pad_w, 0, pad_h], mode='reflect')
    _, pH, pW = img_t.shape
    ys = sorted(set(list(range(0, max(1, pH-crop+1), stride)) + [max(0, pH-crop)]))
    xs = sorted(set(list(range(0, max(1, pW-crop+1), stride)) + [max(0, pW-crop)]))
    for y in ys:
        for x in xs:
            c = img_t[:, y:y+crop, x:x+crop].unsqueeze(0).to(device)
            with torch.no_grad():
                o = model(c)
                o = o[0] if isinstance(o, tuple) else o
                p = F.softmax(o, dim=1)[0]
            ye = min(y+crop, H); xe = min(x+crop, W)
            pred_sum[:, y:ye, x:xe] += p[:, :ye-y, :xe-x]
            count[y:ye, x:xe] += 1
    return (pred_sum / count.clamp(min=1).unsqueeze(0)).argmax(0).cpu().numpy()

# Color palettes — V: Nature/ColorBrewer colorblind-safe
WS2_COLORS = {0: (210,210,210), 1: (253,174,97), 2: (116,173,209), 3: (69,117,180)}
GR_COLORS  = {0: (210,210,210), 1: (116,173,209), 2: (69,117,180)}

def colorize(mask, palette):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in palette.items():
        rgb[mask == cls] = color
    return rgb

def inference_grid(exp_name, data_dir, split, nc, palette, max_samples=6):
    """Run inference and create a grid: original | prediction | GT"""
    log_dir = f'output/finetune_{exp_name}'
    ckpt_path = os.path.join(log_dir, 'best_model.pth')
    if not os.path.exists(ckpt_path):
        print(f'  No checkpoint: {ckpt_path}')
        return

    model = build_model(nc=nc)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('model', ckpt)
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()

    # Compute dataset mean/std
    img_dir = os.path.join(data_dir, 'img_dir', split)
    ann_dir = os.path.join(data_dir, 'ann_dir', split)
    train_img_dir = os.path.join(data_dir, 'img_dir', 'train')

    imgs_for_stats = sorted(glob.glob(os.path.join(train_img_dir, '*')))[:30]
    accum = torch.zeros(3); accum2 = torch.zeros(3); npx = 0
    for p in imgs_for_stats:
        t = T.ToTensor()(Image.open(p).convert('RGB'))
        accum += t.mean(dim=[1,2]); accum2 += (t**2).mean(dim=[1,2]); npx += 1
    mean = (accum / npx).tolist()
    std = ((accum2 / npx - (accum / npx)**2).sqrt()).tolist()

    # Get test images
    test_imgs = sorted(glob.glob(os.path.join(img_dir, '*.jpg')) +
                       glob.glob(os.path.join(img_dir, '*.png')))
    if len(test_imgs) > max_samples:
        random.seed(42)
        test_imgs = random.sample(test_imgs, max_samples)

    rows = []
    for img_path in test_imgs:
        bn = os.path.splitext(os.path.basename(img_path))[0]
        gt_path = os.path.join(ann_dir, bn + '.png')
        if not os.path.exists(gt_path): continue

        img = Image.open(img_path).convert('RGB')
        gt = np.array(Image.open(gt_path))
        img_t = TF.normalize(T.ToTensor()(img), mean, std)
        pred = sliding_window(model, img_t, nc=nc)

        img_np = np.array(img)
        pred_rgb = colorize(pred, palette)
        gt_rgb = colorize(gt, palette)

        # Resize all to same height
        h = 256
        w = int(img_np.shape[1] * h / img_np.shape[0])
        img_r = np.array(Image.fromarray(img_np).resize((w, h)))
        pred_r = np.array(Image.fromarray(pred_rgb).resize((w, h), Image.NEAREST))
        gt_r = np.array(Image.fromarray(gt_rgb).resize((w, h), Image.NEAREST))
        rows.append(np.concatenate([img_r, pred_r, gt_r], axis=1))

    del model; torch.cuda.empty_cache()

    if not rows:
        print(f'  No valid images for {exp_name}')
        return

    # Stack rows with separation
    sep = np.ones((4, rows[0].shape[1], 3), dtype=np.uint8) * 255
    grid_parts = []
    for i, row in enumerate(rows):
        grid_parts.append(row)
        if i < len(rows) - 1:
            grid_parts.append(sep)
    grid = np.concatenate(grid_parts, axis=0)

    # Add column headers
    fig, ax = plt.subplots(figsize=(12, 2 * len(rows)))
    ax.imshow(grid)
    w_col = rows[0].shape[1] // 3
    for i, title in enumerate(['Original', 'Prediction', 'Ground Truth']):
        ax.text(w_col * i + w_col // 2, -10, title, ha='center', va='bottom',
                fontsize=12, fontweight='bold')
    ax.set_title(f'{exp_name}', fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')

    out = f'{OUT_DIR}/v2_inference_{exp_name}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f'Saved: {out}')


if __name__ == '__main__':
    os.chdir('/root/autodl-tmp/PhysicalNet')

    # 1. Training curves
    print('=== Training Curves ===')
    plot_v2_training_curves()

    # 2. v1 vs v2 bar chart
    print('=== v1 vs v2 Bar Chart ===')
    plot_v1v2_comparison()

    # 3. Inference grids
    print('=== Inference Grids ===')
    experiments = [
        ('ws2supp_v2_ft_resethead', 'supplementary_prepared/WS2_supp', 'val', 4, WS2_COLORS),
        ('ws2supp_v2_scratch', 'supplementary_prepared/WS2_supp', 'val', 4, WS2_COLORS),
        ('grsupp_v2_ft_resethead', 'supplementary_prepared/Gr_supp', 'val', 3, GR_COLORS),
        ('grsupp_v2_scratch', 'supplementary_prepared/Gr_supp', 'val', 3, GR_COLORS),
        ('ws2combined_v2_ft_resethead', 'supplementary_prepared/WS2_combined', 'val', 4, WS2_COLORS),
    ]
    for exp_name, data_dir, split, nc, pal in experiments:
        print(f'  Processing {exp_name}...')
        inference_grid(exp_name, data_dir, split, nc, pal)

    print('\n✅ All V2 visualizations complete!')
