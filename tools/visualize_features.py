"""
Feature Map Visualization for RepELA-Net.

Visualizes intermediate feature maps from each module of RepELA-Net
using forward hooks. Produces a multi-panel figure showing:
  - Original image
  - Stem output
  - Stage 1 (RepConv)
  - Stage 2 (RepConv)
  - Stage 3 (ELA)
  - Stage 4 (ELA)
  - Decoder (after BoundaryEnhancement)
  - Prediction

Usage:
    python tools/visualize_features.py \\
      --checkpoint output/seed_test/seed_42/repela_small_*/best_model.pth \\
      --image Mos2_data/ori/MoS2/m10.jpg \\
      --mask  Mos2_data/mask/m10.png \\
      --output output/visualization/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os as _os
_os.chdir(str(Path(__file__).resolve().parents[1]))

import os
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from models.repela_net import repela_net_small
from datasets.mos2_dataset import MoS2Dataset


# ─── Color Definitions ───────────────────────────────────────────────

CLASS_COLORS = {
    0: (128, 128, 128),  # background - gray
    1: (0, 255, 0),      # monolayer - green
    2: (0, 0, 255),      # fewlayer - blue
    3: (255, 165, 0),    # multilayer - orange
}
CLASS_NAMES = ['background', 'monolayer', 'fewlayer', 'multilayer']


# ─── Feature Extraction with Hooks ───────────────────────────────────

class FeatureExtractor:
    """Register hooks on RepELA-Net modules to capture intermediate features."""

    def __init__(self, model):
        self.features = {}
        self.hooks = []

        # Register hooks on key modules
        hook_points = {
            'Stem': model.stem,
            'Stage1_RepConv': model.stage1,
            'Stage2_RepConv': model.stage2,
            'Stage3_ELA': model.stage3,
            'Stage4_ELA': model.stage4,
            'Decoder_Boundary': model.decoder.boundary,
            'Decoder_Output': model.decoder.seg_head,
        }

        for name, module in hook_points.items():
            hook = module.register_forward_hook(self._make_hook(name))
            self.hooks.append(hook)

    def _make_hook(self, name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            self.features[name] = output.detach().cpu()
        return hook

    def clear(self):
        self.features.clear()

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()


# ─── Visualization Helpers ───────────────────────────────────────────

def feature_to_heatmap(feat, target_size):
    """Convert feature map [1, C, H, W] -> heatmap [H_target, W_target]."""
    # Channel-wise mean
    heatmap = feat[0].mean(dim=0).numpy()  # [H, W]
    # Resize to target
    heatmap = cv2.resize(heatmap, (target_size[1], target_size[0]),
                         interpolation=cv2.INTER_LINEAR)
    # Normalize to [0, 1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap


def mask_to_color(mask):
    """Convert class mask to RGB color image."""
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, rgb in CLASS_COLORS.items():
        color[mask == cls_id] = rgb
    return color


def overlay_heatmap(image, heatmap, alpha=0.5, cmap='jet'):
    """Overlay heatmap on image."""
    cm = plt.cm.get_cmap(cmap)
    heatmap_colored = cm(heatmap)[:, :, :3]  # [H, W, 3] float
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    # Blend
    image_uint8 = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image
    overlay = cv2.addWeighted(image_uint8, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay


# ─── Main Visualization ─────────────────────────────────────────────

def visualize_single_image(model, extractor, image_path, mask_path, output_dir,
                           device, image_name=None):
    """Generate feature map visualization for a single image."""
    extractor.clear()

    # Load image
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    H, W = image_rgb.shape[:2]

    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Preprocess (same as dataset)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_float = image_rgb.astype(np.float32) / 255.0
    img_norm = (img_float - mean) / std
    img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0)

    # Pad to 32
    pad_h = (32 - H % 32) % 32
    pad_w = (32 - W % 32) % 32
    if pad_h > 0 or pad_w > 0:
        img_tensor = F.pad(img_tensor, [0, pad_w, 0, pad_h], mode='reflect')

    # Forward
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        output = model(img_tensor)
    logits = output[0] if isinstance(output, tuple) else output
    logits = logits[:, :, :H, :W]
    pred = logits.argmax(dim=1)[0].cpu().numpy()

    # ─── Create multi-panel figure ───
    viz_names = [
        ('Original', None),
        ('Stem', 'Stem'),
        ('Stage1 RepConv', 'Stage1_RepConv'),
        ('Stage2 RepConv', 'Stage2_RepConv'),
        ('Stage3 ELA', 'Stage3_ELA'),
        ('Stage4 ELA', 'Stage4_ELA'),
        ('Boundary Enh.', 'Decoder_Boundary'),
        ('Prediction', None),
        ('Ground Truth', None),
    ]

    fig, axes = plt.subplots(1, len(viz_names), figsize=(3 * len(viz_names), 3.5))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02)

    for ax, (title, feat_key) in zip(axes, viz_names):
        if title == 'Original':
            ax.imshow(image_rgb)
        elif title == 'Prediction':
            ax.imshow(mask_to_color(pred))
        elif title == 'Ground Truth':
            ax.imshow(mask_to_color(mask))
        elif feat_key in extractor.features:
            heatmap = feature_to_heatmap(extractor.features[feat_key], (H, W))
            overlay = overlay_heatmap(img_float, heatmap, alpha=0.6)
            ax.imshow(overlay)
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)

        ax.axis('off')

    name = image_name or Path(image_path).stem
    save_path = os.path.join(output_dir, f'feature_vis_{name}.png')
    fig.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)
    print(f'Saved: {save_path}')

    # ─── Also save individual heatmaps (higher quality) ───
    indiv_dir = os.path.join(output_dir, name)
    os.makedirs(indiv_dir, exist_ok=True)

    for feat_key, feat_tensor in extractor.features.items():
        heatmap = feature_to_heatmap(feat_tensor, (H, W))
        # Pure heatmap
        fig_h, ax_h = plt.subplots(1, 1, figsize=(6, 6))
        ax_h.imshow(heatmap, cmap='jet', vmin=0, vmax=1)
        ax_h.axis('off')
        fig_h.savefig(os.path.join(indiv_dir, f'{feat_key}_heatmap.png'),
                      dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close(fig_h)

        # Overlay on image
        overlay = overlay_heatmap(img_float, heatmap, alpha=0.5)
        fig_o, ax_o = plt.subplots(1, 1, figsize=(6, 6))
        ax_o.imshow(overlay)
        ax_o.axis('off')
        fig_o.savefig(os.path.join(indiv_dir, f'{feat_key}_overlay.png'),
                      dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close(fig_o)

    print(f'Individual heatmaps saved to: {indiv_dir}/')


# ─── Main ────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description='Feature Map Visualization')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to best_model.pth')
    parser.add_argument('--image', type=str, nargs='+', default=None,
                        help='Image path(s). If not given, uses first 3 test images.')
    parser.add_argument('--mask_dir', type=str, default='Mos2_data/mask',
                        help='Directory containing mask .png files')
    parser.add_argument('--image_dir', type=str, default='Mos2_data/ori/MoS2',
                        help='Directory containing image .jpg files')
    parser.add_argument('--output', type=str, default='output/visualization',
                        help='Output directory')
    parser.add_argument('--num_images', type=int, default=3,
                        help='Number of test images to visualize (if --image not given)')
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.output, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = repela_net_small(num_classes=4, deep_supervision=False, use_cse=False)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    model = model.to(device)
    model.eval()
    print(f'Model loaded from {args.checkpoint}')

    # Register hooks
    extractor = FeatureExtractor(model)

    # Determine images to visualize
    if args.image:
        image_paths = args.image
    else:
        # Use first N test images
        with open('splits/test.txt') as f:
            test_names = [l.strip() for l in f if l.strip()]
        test_names = test_names[:args.num_images]
        image_paths = [os.path.join(args.image_dir, f'{n}.jpg') for n in test_names]

    # Visualize each
    for img_path in image_paths:
        name = Path(img_path).stem
        mask_path = os.path.join(args.mask_dir, f'{name}.png')
        if not os.path.exists(img_path):
            print(f'Skip: {img_path} not found')
            continue
        if not os.path.exists(mask_path):
            print(f'Warning: mask not found {mask_path}, using zeros')
            # Create dummy mask
            img_tmp = cv2.imread(img_path)
            mask_path = '/tmp/dummy_mask.png'
            cv2.imwrite(mask_path, np.zeros(img_tmp.shape[:2], dtype=np.uint8))

        visualize_single_image(model, extractor, img_path, mask_path,
                               args.output, device, name)

    extractor.remove_hooks()
    print(f'\nAll done! Output: {args.output}/')


if __name__ == '__main__':
    main()
