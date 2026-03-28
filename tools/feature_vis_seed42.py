"""
Per-module feature map visualization for seed_42 model.

For each test image, generates pure images (no axes/titles) for:
  - original.png          : original RGB
  - stem.png              : Stem output heatmap overlay
  - stage1_repconv.png    : Stage1 RepConv heatmap overlay
  - stage2_repconv.png    : Stage2 RepConv heatmap overlay
  - stage3_ela.png        : Stage3 ELA heatmap overlay
  - stage4_ela.png        : Stage4 ELA heatmap overlay
  - decoder_boundary.png  : Decoder BoundaryEnhancement heatmap overlay
  - pred_color.png        : Colorized prediction (final output)
  - gt_color.png          : Colorized ground truth

All saved to output/eval_results/seed_42/visualizations/<image_name>/
"""

import sys
from pathlib import Path
sys.path.insert(0, '/root/autodl-tmp/PhysicalNet')

import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.repela_net import repela_net_small

# ─── Config ──────────────────────────────────────────────────────────
PROJECT      = '/root/autodl-tmp/PhysicalNet'
CKPT_PATH    = os.path.join(PROJECT, 'output/seed_test/seed_42/repela_small_20260324_080734/best_model.pth')
IMAGE_DIR    = os.path.join(PROJECT, 'Mos2_data/ori/MoS2')
MASK_DIR     = os.path.join(PROJECT, 'Mos2_data/mask')
SPLIT_FILE   = os.path.join(PROJECT, 'splits/test.txt')
OUTPUT_ROOT  = os.path.join(PROJECT, 'output/eval_results/seed_42/visualizations')

CLASS_COLORS = {
    0: (128, 128, 128),  # background
    1: (0,   200, 0),    # monolayer
    2: (30,  80,  255),  # fewlayer
    3: (255, 140, 0),    # multilayer
}

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

# Module names and corresponding save filenames
HOOK_POINTS = [
    ('Stem',             'stem'),
    ('Stage1_RepConv',   'stage1_repconv'),
    ('Stage2_RepConv',   'stage2_repconv'),
    ('Stage3_ELA',       'stage3_ela'),
    ('Stage4_ELA',       'stage4_ela'),
    ('Decoder_Boundary', 'decoder_boundary'),
]


# ─── Feature Extractor ──────────────────────────────────────────────

class FeatureExtractor:
    def __init__(self, model):
        self.features = {}
        self.hooks = []
        modules = {
            'Stem':             model.stem,
            'Stage1_RepConv':   model.stage1,
            'Stage2_RepConv':   model.stage2,
            'Stage3_ELA':       model.stage3,
            'Stage4_ELA':       model.stage4,
            'Decoder_Boundary': model.decoder.boundary,
        }
        for name, module in modules.items():
            self.hooks.append(
                module.register_forward_hook(self._hook(name))
            )

    def _hook(self, name):
        def fn(module, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            self.features[name] = o.detach().cpu()
        return fn

    def clear(self):
        self.features.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()


# ─── Helpers ─────────────────────────────────────────────────────────

def feature_to_heatmap(feat, target_hw):
    """[1, C, H, W] → normalized heatmap [H_t, W_t] in [0,1]."""
    hm = feat[0].mean(dim=0).numpy()
    hm = cv2.resize(hm, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR)
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    return hm


def heatmap_to_rgb(hm):
    """Normalized heatmap [H,W] → RGB [H,W,3] uint8 using jet colormap."""
    hm_uint8 = (hm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)  # BGR
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def overlay_heatmap(image_rgb, hm, alpha=0.5):
    """Overlay jet heatmap on image. Both [H,W,3] uint8. Returns RGB uint8."""
    hm_rgb = heatmap_to_rgb(hm)
    blended = cv2.addWeighted(image_rgb, 1 - alpha, hm_rgb, alpha, 0)
    return blended


def mask_to_color(mask):
    h, w = mask.shape
    c = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, rgb in CLASS_COLORS.items():
        c[mask == cid] = rgb
    return c


def save_pure_image(arr_rgb, path):
    """Save RGB array as image without any matplotlib decoration."""
    cv2.imwrite(path, cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR))


# ─── Main ────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    print('Loading model...')
    model = repela_net_small(num_classes=4, deep_supervision=False, use_cse=False)
    ckpt  = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    model = model.to(device).eval()
    print(f'  Loaded: {CKPT_PATH}')

    extractor = FeatureExtractor(model)

    # Load test image list
    with open(SPLIT_FILE) as f:
        names = [l.strip() for l in f if l.strip()]
    print(f'Processing {len(names)} test images...\n')

    for name in names:
        img_path  = os.path.join(IMAGE_DIR, f'{name}.jpg')
        mask_path = os.path.join(MASK_DIR,  f'{name}.png')
        out_dir   = os.path.join(OUTPUT_ROOT, name)
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.exists(img_path):
            print(f'  Skip {name}: image not found')
            continue

        # Load image
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = img_rgb.shape[:2]

        # Load mask
        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(mask_path) else None

        # Preprocess
        img_float = img_rgb.astype(np.float32) / 255.0
        img_norm  = (img_float - MEAN) / STD
        tensor    = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0)

        pad_h = (32 - H % 32) % 32
        pad_w = (32 - W % 32) % 32
        if pad_h > 0 or pad_w > 0:
            tensor = F.pad(tensor, [0, pad_w, 0, pad_h], mode='reflect')

        # Forward (hooks capture features)
        extractor.clear()
        with torch.no_grad():
            output = model(tensor.to(device))
        logits = output[0] if isinstance(output, tuple) else output
        logits = logits[:, :, :H, :W]
        pred   = logits.argmax(dim=1)[0].cpu().numpy()

        # Save original
        save_pure_image(img_rgb, os.path.join(out_dir, 'original.png'))

        # Save per-module heatmap overlays
        for hook_name, file_prefix in HOOK_POINTS:
            if hook_name not in extractor.features:
                print(f'    Warning: {hook_name} not captured for {name}')
                continue
            hm = feature_to_heatmap(extractor.features[hook_name], (H, W))
            overlay = overlay_heatmap(img_rgb, hm, alpha=0.5)
            save_pure_image(overlay, os.path.join(out_dir, f'{file_prefix}.png'))

        # Save colorized prediction
        pred_color = mask_to_color(pred)
        save_pure_image(pred_color, os.path.join(out_dir, 'pred_color.png'))

        # Save colorized GT
        if gt is not None:
            gt_color = mask_to_color(gt)
            save_pure_image(gt_color, os.path.join(out_dir, 'gt_color.png'))

        # Save prediction overlay on original
        pred_overlay = cv2.addWeighted(img_rgb, 0.6, pred_color, 0.4, 0)
        save_pure_image(pred_overlay, os.path.join(out_dir, 'pred_overlay.png'))

        n_files = 2 + len(HOOK_POINTS) + (1 if gt is not None else 0) + 1
        print(f'  ✓ {name:>5s}  ({n_files} images) → {out_dir}/')

    extractor.remove()
    print(f'\nDone! All feature visualizations saved to: {OUTPUT_ROOT}/')


if __name__ == '__main__':
    main()
