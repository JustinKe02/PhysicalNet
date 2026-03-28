"""
Progressive feature-to-prediction visualization for seed_42 model.

For each test image, generates colorized probe predictions from 3 key
network modules plus the final prediction. Shows progressive feature
refinement from coarse to fine.

Output per image (saved to output/eval_results/seed_42/visualizations_uinified/<name>/):
  - original.png       : original RGB image
  - stem.png           : Stem probe prediction (trained 1x1 conv on stem features)
  - repconv.png        : RepConv probe prediction (trained 1x1 conv on stage2 features)
  - ela.png            : ELA probe prediction (trained 1x1 conv on stage4 features)
  - pred_color.png     : Final prediction (model's trained decoder output)
  - gt_color.png       : Ground truth mask (colorized)
  - comparison.png     : Side-by-side plot (Original | Pred | GT)
"""

import sys
sys.path.insert(0, '/root/autodl-tmp/PhysicalNet')

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
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
OUTPUT_ROOT  = os.path.join(PROJECT, 'output/eval_results/seed_42/visualizations_uinified')

NUM_CLASSES  = 4
CLASS_NAMES  = ['background', 'monolayer', 'fewlayer', 'multilayer']

# Color palette — V: Nature/ColorBrewer colorblind-safe
CLASS_COLORS = {
    0: (210, 210, 210),  # background  - light gray
    1: (253, 174, 97),   # monolayer   - warm orange
    2: (116, 173, 209),  # fewlayer    - light blue
    3: (69,  117, 180),  # multilayer  - steel blue
}

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

# 3 probe hook points: (display_name, save_filename, model_attribute, out_channels)
PROBE_HOOKS = [
    ('Stem',    'stem',    'stem',   32),
    ('RepConv', 'repconv', 'stage2', 64),
    ('ELA',     'ela',     'stage4', 256),
]


# ─── Feature Extractor ──────────────────────────────────────────────

class FeatureExtractor:
    """Register forward hooks to capture intermediate feature maps."""
    def __init__(self, model):
        self.features = {}
        self.hooks = []
        for name, _, attr, _ in PROBE_HOOKS:
            module = getattr(model, attr)
            self.hooks.append(
                module.register_forward_hook(self._hook(name))
            )

    def _hook(self, name):
        def fn(module, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            self.features[name] = o.detach()
        return fn

    def clear(self):
        self.features.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()


# ─── Probe Classifier Heads ─────────────────────────────────────────

def build_probe_heads(device):
    """Build lightweight 1x1 conv probe classifiers for each hook point."""
    heads = {}
    for name, _, _, c_in in PROBE_HOOKS:
        head = nn.Sequential(nn.Conv2d(c_in, NUM_CLASSES, 1, bias=True))
        nn.init.kaiming_normal_(head[0].weight)
        heads[name] = head.to(device)
    return heads


def train_probe_heads(heads, extractor, model, device, names):
    """Train probe heads on GT masks to produce meaningful predictions."""
    print("Training probe classifier heads on GT masks...")

    optimizers = {k: torch.optim.Adam(v.parameters(), lr=1e-2) for k, v in heads.items()}
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    for epoch in range(30):
        total_loss = {k: 0.0 for k in heads}
        n_samples = 0

        for img_name in names:
            img_path  = os.path.join(IMAGE_DIR, f'{img_name}.jpg')
            mask_path = os.path.join(MASK_DIR, f'{img_name}.png')
            if not os.path.exists(img_path) or not os.path.exists(mask_path):
                continue

            # Load and preprocess
            img_bgr = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            H, W = img_rgb.shape[:2]

            gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            gt_tensor = torch.from_numpy(gt.astype(np.int64)).unsqueeze(0).to(device)

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
                _ = model(tensor.to(device))

            # Train each probe head
            for head_name, head in heads.items():
                feat = extractor.features[head_name]
                logits = head(feat)
                logits_up = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)

                loss = criterion(logits_up, gt_tensor)
                optimizers[head_name].zero_grad()
                loss.backward()
                optimizers[head_name].step()
                total_loss[head_name] += loss.item()

            n_samples += 1

        if (epoch + 1) % 10 == 0:
            avg = {k: v / max(n_samples, 1) for k, v in total_loss.items()}
            print(f"  Epoch {epoch+1}/30: " + ', '.join(f'{k}: {v:.4f}' for k, v in avg.items()))

    for head in heads.values():
        head.eval()
    print("  Done.\n")


# ─── Helpers ─────────────────────────────────────────────────────────

def mask_to_color(mask):
    """Convert class-index mask [H,W] → RGB [H,W,3]."""
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, rgb in CLASS_COLORS.items():
        color[mask == cls_id] = rgb
    return color


def features_to_prediction(feat, head, target_hw):
    """Feature map → classifier → upsample → argmax → class map."""
    with torch.no_grad():
        logits = head(feat)
        logits_up = F.interpolate(logits, size=target_hw, mode='bilinear', align_corners=False)
        return logits_up.argmax(dim=1)[0].cpu().numpy()


def save_pure_image(arr_rgb, path):
    """Save RGB array as PNG without matplotlib decoration."""
    cv2.imwrite(path, cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR))


def make_comparison(img_rgb, pred_color, gt_color, name, save_path):
    """Side-by-side: Original | Prediction | Ground Truth."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, im, title in zip(axes,
                              [img_rgb, pred_color, gt_color],
                              ['Original', 'Final Prediction', 'Ground Truth']):
        ax.imshow(im)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')

    patches = [plt.Line2D([0], [0], marker='s', color='w',
                           markerfacecolor=np.array(CLASS_COLORS[i])/255.,
                           markersize=12, label=CLASS_NAMES[i])
               for i in range(NUM_CLASSES)]
    fig.legend(handles=patches, loc='lower center', ncol=4,
               fontsize=12, frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(name, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


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

    # Set up
    extractor = FeatureExtractor(model)
    with open(SPLIT_FILE) as f:
        names = [l.strip() for l in f if l.strip()]
    print(f'{len(names)} test images\n')

    # Build and train probe heads
    heads = build_probe_heads(device)
    train_probe_heads(heads, extractor, model, device, names)

    # Generate visualizations
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    print('Generating visualizations...\n')

    for name in names:
        img_path  = os.path.join(IMAGE_DIR, f'{name}.jpg')
        mask_path = os.path.join(MASK_DIR,  f'{name}.png')
        out_dir   = os.path.join(OUTPUT_ROOT, name)
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.exists(img_path):
            print(f'  Skip {name}: not found')
            continue

        # Load
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = img_rgb.shape[:2]
        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(mask_path) else None

        # Preprocess
        img_float = img_rgb.astype(np.float32) / 255.0
        img_norm  = (img_float - MEAN) / STD
        tensor    = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0)
        pad_h = (32 - H % 32) % 32
        pad_w = (32 - W % 32) % 32
        if pad_h > 0 or pad_w > 0:
            tensor = F.pad(tensor, [0, pad_w, 0, pad_h], mode='reflect')

        # Forward
        extractor.clear()
        with torch.no_grad():
            output = model(tensor.to(device))
        logits = output[0] if isinstance(output, tuple) else output
        logits = logits[:, :, :H, :W]
        pred = logits.argmax(dim=1)[0].cpu().numpy()

        # Save original
        save_pure_image(img_rgb, os.path.join(out_dir, 'original.png'))

        # Save probe predictions (Stem, RepConv, ELA)
        for hook_name, file_prefix, _, _ in PROBE_HOOKS:
            if hook_name not in extractor.features:
                print(f'    Warning: {hook_name} not captured for {name}')
                continue
            probe_pred = features_to_prediction(extractor.features[hook_name], heads[hook_name], (H, W))
            save_pure_image(mask_to_color(probe_pred), os.path.join(out_dir, f'{file_prefix}.png'))

        # Save final prediction
        pred_color = mask_to_color(pred)
        save_pure_image(pred_color, os.path.join(out_dir, 'pred_color.png'))

        # Save GT
        gt_color = None
        if gt is not None:
            gt_color = mask_to_color(gt)
            save_pure_image(gt_color, os.path.join(out_dir, 'gt_color.png'))

        # Save comparison
        if gt_color is not None:
            make_comparison(img_rgb, pred_color, gt_color, name,
                            os.path.join(out_dir, 'comparison.png'))

        print(f'  ✓ {name:>5s} → {out_dir}/')

    extractor.remove()
    print(f'\nDone! Saved to: {OUTPUT_ROOT}/')


if __name__ == '__main__':
    main()
