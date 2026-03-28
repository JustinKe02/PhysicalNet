"""
Colorize seed_42 inference results.

For each test image, creates a folder under output/eval_results/seed_42/visualizations/<image_name>/
containing:
  - original.png       : the original RGB image
  - gt_color.png       : ground truth mask (colorized)
  - pred_color.png     : prediction mask (colorized)
  - comparison.png     : side-by-side: Original | Prediction | Ground Truth
"""

import os
import sys
import re
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── Config ──────────────────────────────────────────────────────────
PROJECT_ROOT = '/root/autodl-tmp/PhysicalNet'
PRED_DIR     = os.path.join(PROJECT_ROOT, 'output/eval_results/seed_42')
IMAGE_DIR    = os.path.join(PROJECT_ROOT, 'Mos2_data/ori/MoS2')
MASK_DIR     = os.path.join(PROJECT_ROOT, 'Mos2_data/mask')
OUTPUT_ROOT  = os.path.join(PRED_DIR, 'visualizations')
METRICS_FILE = os.path.join(PRED_DIR, 'test_metrics.txt')

CLASS_NAMES = ['background', 'monolayer', 'fewlayer', 'multilayer']
# More visually appealing colors (RGB)
CLASS_COLORS = {
    0: (128, 128, 128),  # background - gray
    1: (0,   200, 0),    # monolayer  - green
    2: (30,  80,  255),  # fewlayer   - blue
    3: (255, 140, 0),    # multilayer - orange
}

# ─── Helpers ─────────────────────────────────────────────────────────

def mask_to_color(mask):
    """Convert class-index mask [H,W] → RGB [H,W,3]."""
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, rgb in CLASS_COLORS.items():
        color[mask == cls_id] = rgb
    return color


def parse_per_image_miou(metrics_path):
    """Parse per-image mIoU from test_metrics.txt."""
    miou_map = {}
    with open(metrics_path) as f:
        for line in f:
            m = re.match(r'\s+(\w+):\s+mIoU=([0-9.]+)', line)
            if m:
                miou_map[m.group(1)] = float(m.group(2))
    return miou_map


def make_comparison(img_rgb, pred_color, gt_color, name, miou, save_path):
    """Create side-by-side comparison figure: Original | Pred | GT."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img_rgb)
    axes[0].set_title('Original', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(pred_color)
    axes[1].set_title(f'Prediction (mIoU={miou:.4f})', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(gt_color)
    axes[2].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    # Legend
    legend_patches = []
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        c = np.array(CLASS_COLORS[cls_id]) / 255.0
        legend_patches.append(plt.Line2D([0], [0], marker='s', color='w',
                                          markerfacecolor=c, markersize=12,
                                          label=cls_name))
    fig.legend(handles=legend_patches, loc='lower center', ncol=4,
               fontsize=12, frameon=True, fancybox=True,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f'{name}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


# ─── Main ────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # Parse per-image mIoU
    miou_map = parse_per_image_miou(METRICS_FILE)
    print(f'Parsed mIoU for {len(miou_map)} images')

    # Find all prediction files
    pred_files = sorted([f for f in os.listdir(PRED_DIR) if f.endswith('_pred.png')])
    print(f'Found {len(pred_files)} prediction files')

    for pred_file in pred_files:
        name = pred_file.replace('_pred.png', '')
        pred_path = os.path.join(PRED_DIR, pred_file)
        img_path  = os.path.join(IMAGE_DIR, f'{name}.jpg')
        mask_path = os.path.join(MASK_DIR,  f'{name}.png')

        if not os.path.exists(img_path):
            print(f'  Skip {name}: image not found')
            continue

        # Create per-image output directory
        out_dir = os.path.join(OUTPUT_ROOT, name)
        os.makedirs(out_dir, exist_ok=True)

        # Load
        img_bgr  = cv2.imread(img_path)
        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pred     = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt       = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(mask_path) else None

        # Colorize
        pred_color = mask_to_color(pred)
        gt_color   = mask_to_color(gt) if gt is not None else np.zeros_like(img_rgb)

        # Save individual images
        cv2.imwrite(os.path.join(out_dir, 'original.png'), img_bgr)
        cv2.imwrite(os.path.join(out_dir, 'pred_color.png'), cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))
        if gt is not None:
            cv2.imwrite(os.path.join(out_dir, 'gt_color.png'), cv2.cvtColor(gt_color, cv2.COLOR_RGB2BGR))

        # Create overlay (prediction on original, 40% opacity)
        overlay = cv2.addWeighted(img_rgb, 0.6, pred_color, 0.4, 0)
        cv2.imwrite(os.path.join(out_dir, 'pred_overlay.png'),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        # Create comparison figure
        miou = miou_map.get(name, 0.0)
        make_comparison(img_rgb, pred_color, gt_color, name, miou,
                        os.path.join(out_dir, 'comparison.png'))

        print(f'  ✓ {name} (mIoU={miou:.4f}) → {out_dir}/')

    print(f'\nDone! All visualizations saved to: {OUTPUT_ROOT}/')


if __name__ == '__main__':
    main()
