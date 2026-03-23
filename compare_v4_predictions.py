"""
Side-by-side comparison: Original | Exp1 (Old) | Exp2 (New) | Ground Truth.

Runs inference with both models and generates one comparison image per val sample.

Usage:
    python compare_v4_predictions.py
"""

import os
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from train_ablation import build_ablation_model

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT = './Mos2_data'
SPLIT_DIR = 'splits/'
SPLIT = 'val'
OUTPUT_DIR = 'outputv4_plots/comparison'

EXP1_CKPT = 'outputv4_oldcfg/nocolor_oldcfg_20260322_233322/best_model.pth'
EXP2_CKPT = 'outputv4/ablation_no_color_20260323_005458/best_model.pth'

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
CROP_SIZE, STRIDE = 512, 384

CLASS_COLORS = np.array([
    [0, 0, 0],        # background
    [239, 41, 41],     # monolayer  (red)
    [0, 170, 0],       # fewlayer   (green)
    [114, 159, 207],   # multilayer (blue)
], dtype=np.uint8)


# ── Helpers ───────────────────────────────────────────────────────────────────

def colorize(mask):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c, col in enumerate(CLASS_COLORS):
        out[mask == c] = col
    return out


def overlay(img_np, mask, alpha=0.45):
    color = colorize(mask)
    result = img_np.copy()
    fg = mask > 0
    result[fg] = ((1 - alpha) * img_np[fg] + alpha * color[fg]).astype(np.uint8)
    return result


def sliding_window_predict(model, img_tensor, device):
    _, H, W = img_tensor.shape
    pred_sum = torch.zeros(4, H, W, dtype=torch.float32, device=device)
    count = torch.zeros(H, W, dtype=torch.float32, device=device)

    pad_h, pad_w = max(0, CROP_SIZE - H), max(0, CROP_SIZE - W)
    padded = F.pad(img_tensor, [0, pad_w, 0, pad_h], mode='reflect') \
             if (pad_h > 0 or pad_w > 0) else img_tensor
    _, pH, pW = padded.shape

    ys = sorted(set(list(range(0, max(1, pH - CROP_SIZE + 1), STRIDE))
                    + [max(0, pH - CROP_SIZE)]))
    xs = sorted(set(list(range(0, max(1, pW - CROP_SIZE + 1), STRIDE))
                    + [max(0, pW - CROP_SIZE)]))

    for y in ys:
        for x in xs:
            crop = padded[:, y:y+CROP_SIZE, x:x+CROP_SIZE].unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(crop)
                logits = out[0] if isinstance(out, tuple) else out
                probs = F.softmax(logits, dim=1)[0]
            ye, xe = min(y + CROP_SIZE, H), min(x + CROP_SIZE, W)
            pred_sum[:, y:ye, x:xe] += probs[:, :ye-y, :xe-x]
            count[y:ye, x:xe] += 1

    return (pred_sum / count.clamp(min=1).unsqueeze(0)).argmax(dim=0).cpu().numpy()


def load_model(ckpt_path, deep_supervision, device):
    model = build_ablation_model('no_color', num_classes=4,
                                  deep_supervision=deep_supervision).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()
    epoch = ckpt.get('epoch', -1) + 1 if isinstance(ckpt, dict) else '?'
    miou = f"{ckpt.get('best_miou', 0):.4f}" if isinstance(ckpt, dict) else '?'
    print(f'  Loaded: {ckpt_path} (Epoch {epoch}, mIoU={miou})')
    return model


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading models...')
    model_exp1 = load_model(EXP1_CKPT, deep_supervision=False, device=device)
    model_exp2 = load_model(EXP2_CKPT, deep_supervision=True, device=device)

    # Read split
    with open(os.path.join(SPLIT_DIR, f'{SPLIT}.txt')) as f:
        basenames = [l.strip() for l in f if l.strip()]

    img_dir = os.path.join(DATA_ROOT, 'ori', 'MoS2')
    mask_dir = os.path.join(DATA_ROOT, 'mask')

    legend_elements = [
        Patch(facecolor=np.array(CLASS_COLORS[1]) / 255., label='Monolayer'),
        Patch(facecolor=np.array(CLASS_COLORS[2]) / 255., label='Fewlayer'),
        Patch(facecolor=np.array(CLASS_COLORS[3]) / 255., label='Multilayer'),
    ]

    print(f'\nGenerating {len(basenames)} comparisons...\n')

    for i, bn in enumerate(basenames):
        # Load original image
        img_path = os.path.join(img_dir, f'{bn}.jpg')
        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir, f'{bn}.png')
        if not os.path.exists(img_path):
            continue

        img_pil = Image.open(img_path).convert('RGB')
        img_np = np.array(img_pil)
        img_tensor = TF.normalize(TF.to_tensor(img_pil), MEAN, STD)

        # Run inference with both models
        pred_exp1 = sliding_window_predict(model_exp1, img_tensor, device)
        pred_exp2 = sliding_window_predict(model_exp2, img_tensor, device)

        # Ground truth
        gt_path = os.path.join(mask_dir, f'{bn}.png')
        gt_mask = np.array(Image.open(gt_path)) if os.path.exists(gt_path) else None

        # ── Create figure ──
        fig, axes = plt.subplots(1, 4, figsize=(28, 7))

        # Col 1: Original
        axes[0].imshow(img_np)
        axes[0].set_title('Original', fontsize=16, fontweight='bold')
        axes[0].axis('off')

        # Col 2: Exp1 prediction overlay
        axes[1].imshow(colorize(pred_exp1))
        axes[1].set_title('Exp1: Old Config\n(Focal α, best mIoU=0.8333)',
                          fontsize=13, fontweight='bold', color='#1565C0')
        axes[1].axis('off')

        # Col 3: Exp2 prediction overlay
        axes[2].imshow(colorize(pred_exp2))
        axes[2].set_title('Exp2: New Config\n(DS+CopyPaste, best mIoU=0.8264)',
                          fontsize=13, fontweight='bold', color='#E65100')
        axes[2].axis('off')

        # Col 4: Ground truth overlay
        if gt_mask is not None:
            axes[3].imshow(colorize(gt_mask))
        else:
            axes[3].imshow(img_np)
            axes[3].text(0.5, 0.5, 'No GT', ha='center', va='center',
                         fontsize=24, color='red', transform=axes[3].transAxes)
        axes[3].set_title('Ground Truth', fontsize=16, fontweight='bold',
                          color='#2E7D32')
        axes[3].axis('off')

        fig.legend(handles=legend_elements, loc='lower center', ncol=3,
                   fontsize=13, bbox_to_anchor=(0.5, -0.01),
                   frameon=True, edgecolor='#ccc', fancybox=True)

        plt.tight_layout(pad=0.5)
        save_path = os.path.join(OUTPUT_DIR, f'{bn}_compare.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f'  [{i+1}/{len(basenames)}] {bn}')

    print(f'\n✅ All saved to: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
