"""
4-column comparison: Original | UNet-MiT-B0 | RepELA Old | RepELA New.

All predictions shown as colorized masks.

Usage:
    python compare_models.py
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
OUTPUT_DIR = 'outputv4_plots/model_comparison'

UNET_CKPT = 'outputv3/unet_mit_b0_20260322_193307/best_model.pth'
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


def sliding_window_predict(model, img_tensor, device, is_smp=False):
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
            crop = padded[:, y:y+CROP_SIZE, x:x+CROP_SIZE]

            # SMP models need 32-multiple padding
            if is_smp:
                _, cH, cW = crop.shape
                smp_pad_h = (32 - cH % 32) % 32
                smp_pad_w = (32 - cW % 32) % 32
                if smp_pad_h > 0 or smp_pad_w > 0:
                    crop = F.pad(crop, [0, smp_pad_w, 0, smp_pad_h], mode='reflect')

            with torch.no_grad():
                out = model(crop.unsqueeze(0).to(device))
                logits = out[0] if isinstance(out, tuple) else out
                # Crop back smp padding
                if is_smp and (smp_pad_h > 0 or smp_pad_w > 0):
                    logits = logits[:, :, :cH, :cW]
                probs = F.softmax(logits, dim=1)[0]

            ye, xe = min(y + CROP_SIZE, H), min(x + CROP_SIZE, W)
            pred_sum[:, y:ye, x:xe] += probs[:, :ye-y, :xe-x]
            count[y:ye, x:xe] += 1

    return (pred_sum / count.clamp(min=1).unsqueeze(0)).argmax(dim=0).cpu().numpy()


def load_unet(ckpt_path, device):
    import segmentation_models_pytorch as smp
    model = smp.Unet(encoder_name='mit_b0', encoder_weights=None,
                     in_channels=3, classes=4).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()
    epoch = ckpt.get('epoch', -1) + 1
    miou = ckpt.get('best_miou', 0)
    print(f'  UNet-MiT-B0: Epoch {epoch}, mIoU={miou:.4f}')
    return model


def load_repela(ckpt_path, deep_supervision, device):
    model = build_ablation_model('no_color', num_classes=4,
                                  deep_supervision=deep_supervision).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()
    epoch = ckpt.get('epoch', -1) + 1
    miou = ckpt.get('best_miou', 0)
    print(f'  RepELA-Net: Epoch {epoch}, mIoU={miou:.4f} (DS={deep_supervision})')
    return model


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading models...')
    model_unet = load_unet(UNET_CKPT, device)
    model_exp1 = load_repela(EXP1_CKPT, deep_supervision=False, device=device)
    model_exp2 = load_repela(EXP2_CKPT, deep_supervision=True, device=device)

    # Read split
    with open(os.path.join(SPLIT_DIR, f'{SPLIT}.txt')) as f:
        basenames = [l.strip() for l in f if l.strip()]

    img_dir = os.path.join(DATA_ROOT, 'ori', 'MoS2')

    legend_elements = [
        Patch(facecolor=np.array(CLASS_COLORS[1]) / 255., label='Monolayer'),
        Patch(facecolor=np.array(CLASS_COLORS[2]) / 255., label='Fewlayer'),
        Patch(facecolor=np.array(CLASS_COLORS[3]) / 255., label='Multilayer'),
    ]

    print(f'\nGenerating {len(basenames)} comparisons...\n')

    for i, bn in enumerate(basenames):
        img_path = os.path.join(img_dir, f'{bn}.jpg')
        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir, f'{bn}.png')
        if not os.path.exists(img_path):
            continue

        img_pil = Image.open(img_path).convert('RGB')
        img_np = np.array(img_pil)
        img_tensor = TF.normalize(TF.to_tensor(img_pil), MEAN, STD)

        # Run inference
        pred_unet = sliding_window_predict(model_unet, img_tensor, device, is_smp=True)
        pred_exp1 = sliding_window_predict(model_exp1, img_tensor, device, is_smp=False)
        pred_exp2 = sliding_window_predict(model_exp2, img_tensor, device, is_smp=False)

        # ── Figure ──
        fig, axes = plt.subplots(1, 4, figsize=(28, 7))

        # Col 1: Original
        axes[0].imshow(img_np)
        axes[0].set_title('Original', fontsize=16, fontweight='bold')
        axes[0].axis('off')

        # Col 2: UNet-MiT-B0
        axes[1].imshow(colorize(pred_unet))
        axes[1].set_title('UNet-MiT-B0\n(3.72M, mIoU=0.8901)',
                          fontsize=13, fontweight='bold', color='#7B1FA2')
        axes[1].axis('off')

        # Col 3: RepELA Exp1 (Old Config)
        axes[2].imshow(colorize(pred_exp1))
        axes[2].set_title('RepELA-Net (Old)\n(2.12M, mIoU=0.8333)',
                          fontsize=13, fontweight='bold', color='#1565C0')
        axes[2].axis('off')

        # Col 4: RepELA Exp2 (New Config)
        axes[3].imshow(colorize(pred_exp2))
        axes[3].set_title('RepELA-Net (New)\n(2.12M, mIoU=0.8264)',
                          fontsize=13, fontweight='bold', color='#E65100')
        axes[3].axis('off')

        fig.legend(handles=legend_elements, loc='lower center', ncol=3,
                   fontsize=13, bbox_to_anchor=(0.5, -0.01),
                   frameon=True, edgecolor='#ccc', fancybox=True)

        plt.tight_layout(pad=0.5)
        save_path = os.path.join(OUTPUT_DIR, f'{bn}_models.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f'  [{i+1}/{len(basenames)}] {bn}')

    print(f'\n✅ All saved to: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
