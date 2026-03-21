"""
Grad-CAM Feature Visualization for RepELA-Net.

Shows WHERE the model truly focuses at each stage using gradient-weighted
class activation mapping (Grad-CAM), not just raw feature magnitudes.

For each target layer, Grad-CAM:
  1. Picks a target class (e.g., all foreground classes)
  2. Backpropagates the class score to get gradients at that layer
  3. Weights feature channels by global-avg-pooled gradients
  4. Applies ReLU to keep only positive contributions

This reveals the progressive refinement of attention:
  - RepConv: local edges & textures
  - ELA: global semantic regions
  - Decoder: boundary refinement

Output layout:
  Original | RepConv Grad-CAM | ELA Grad-CAM | Decoder Grad-CAM | Prediction [| GT]

Usage:
    python visualize_features.py \
        --image Mos2_data/ori/MoS2/m7.jpg \
        --checkpoint output/repela_small_*/best_model.pth

    python visualize_features.py --split test --max_images 4 \
        --checkpoint output/repela_small_*/best_model.pth
"""

import os
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.repela_net import repela_net_tiny, repela_net_small, repela_net_base


CLASSES = ['background', 'monolayer', 'fewlayer', 'multilayer']
CLASS_COLORS = np.array([
    [0, 0, 0], [239, 41, 41], [0, 170, 0], [114, 159, 207]
], dtype=np.uint8)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Layers to visualize
LAYER_CONFIG = [
    ('stage2', 'After RepConv'),
    ('stage3', 'After ELA'),
    ('decoder', 'After Decoder'),
]


class GradCAM:
    """Grad-CAM for semantic segmentation models.

    For each target layer, computes gradient-weighted activation maps
    that show which spatial regions contribute most to the model's
    prediction of foreground classes.
    """

    def __init__(self, model, target_layer_names):
        self.model = model
        self.activations = {}
        self.gradients = {}
        self._hooks = []

        # Map names to actual modules
        layer_map = {
            'stage1': model.stage1,
            'stage2': model.stage2,
            'stage3': model.stage3,
            'stage4': model.stage4,
            'decoder': model.decoder,
        }

        for name in target_layer_names:
            module = layer_map[name]
            self._hooks.append(
                module.register_forward_hook(self._fwd_hook(name))
            )
            self._hooks.append(
                module.register_full_backward_hook(self._bwd_hook(name))
            )

    def _fwd_hook(self, name):
        def hook(module, inp, out):
            t = out[0] if isinstance(out, (list, tuple)) else out
            self.activations[name] = t
        return hook

    def _bwd_hook(self, name):
        def hook(module, grad_input, grad_output):
            t = grad_output[0] if isinstance(grad_output, (list, tuple)) else grad_output
            self.gradients[name] = t
        return hook

    def compute(self, input_tensor, target_classes=None):
        """Compute Grad-CAM heatmaps for all registered layers.

        Args:
            input_tensor: [1, 3, H, W]
            target_classes: list of class indices to visualize.
                If None, uses all foreground classes (1,2,3).

        Returns:
            dict of {layer_name: heatmap [H_input, W_input] numpy}
        """
        if target_classes is None:
            target_classes = [1, 2, 3]  # all foreground

        self.model.zero_grad()
        self.activations = {}
        self.gradients = {}

        # Forward
        logits = self.model(input_tensor)  # [1, C, H, W]

        # Target score: sum of logits for target classes across all pixels
        target_score = 0
        for cls_id in target_classes:
            target_score = target_score + logits[0, cls_id].sum()

        # Backward
        target_score.backward(retain_graph=False)

        # Compute Grad-CAM for each layer
        _, _, H_in, W_in = input_tensor.shape
        heatmaps = {}

        for name in self.activations:
            act = self.activations[name]  # [1, C, h, w]
            grad = self.gradients[name]   # [1, C, h, w]

            # Global average pool gradients → channel weights
            weights = grad.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]

            # Weighted combination of activation maps
            cam = (weights * act).sum(dim=1, keepdim=True)  # [1, 1, h, w]
            cam = F.relu(cam)  # Only positive contributions

            # Upsample to input resolution
            cam = F.interpolate(cam, size=(H_in, W_in),
                                mode='bilinear', align_corners=False)
            cam = cam[0, 0].detach().cpu().numpy()

            # Normalize to [0, 1]
            cmin, cmax = cam.min(), cam.max()
            if cmax - cmin > 1e-8:
                cam = (cam - cmin) / (cmax - cmin)
            else:
                cam = np.zeros_like(cam)

            heatmaps[name] = cam

        return heatmaps, logits

    def remove(self):
        for h in self._hooks:
            h.remove()


def sliding_window_gradcam(model, img_tensor, layer_names,
                           crop_size, stride, device,
                           target_classes=None):
    """Full-image Grad-CAM using sliding window.

    For each window:
      1. Run Grad-CAM (forward + backward)
      2. Accumulate heatmaps and predictions at original resolution

    Returns:
        prediction: [H, W] numpy int
        heatmaps: dict of {layer_name: [H, W] numpy float}
    """
    _, H, W = img_tensor.shape
    num_classes = 4

    pred_sum = torch.zeros(num_classes, H, W, dtype=torch.float32, device=device)
    pred_count = torch.zeros(H, W, dtype=torch.float32, device=device)

    heat_sum = {n: np.zeros((H, W), dtype=np.float64) for n in layer_names}
    heat_count = {n: np.zeros((H, W), dtype=np.float64) for n in layer_names}

    # Pad if needed
    pad_h = max(0, crop_size - H)
    pad_w = max(0, crop_size - W)
    padded = img_tensor
    if pad_h > 0 or pad_w > 0:
        padded = F.pad(img_tensor, [0, pad_w, 0, pad_h], mode='reflect')
    _, pH, pW = padded.shape

    # Full-coverage window positions
    ys = sorted(set(
        list(range(0, max(1, pH - crop_size + 1), stride)) +
        [max(0, pH - crop_size)]
    ))
    xs = sorted(set(
        list(range(0, max(1, pW - crop_size + 1), stride)) +
        [max(0, pW - crop_size)]
    ))

    gcam = GradCAM(model, layer_names)

    for y in ys:
        for x in xs:
            crop = padded[:, y:y+crop_size, x:x+crop_size].unsqueeze(0).to(device)
            crop.requires_grad_(False)  # gradients only on model params

            heatmaps, logits = gcam.compute(crop, target_classes)
            probs = F.softmax(logits.detach(), dim=1)[0]

            y_end = min(y + crop_size, H)
            x_end = min(x + crop_size, W)
            vh, vw = y_end - y, x_end - x

            pred_sum[:, y:y_end, x:x_end] += probs[:, :vh, :vw]
            pred_count[y:y_end, x:x_end] += 1

            for name in layer_names:
                hm = heatmaps[name]  # [crop_size, crop_size]
                heat_sum[name][y:y_end, x:x_end] += hm[:vh, :vw]
                heat_count[name][y:y_end, x:x_end] += 1

    gcam.remove()

    # Average
    pred_count = pred_count.clamp(min=1)
    prediction = (pred_sum / pred_count.unsqueeze(0)).argmax(dim=0).cpu().numpy()

    final_heatmaps = {}
    for name in layer_names:
        cnt = np.maximum(heat_count[name], 1)
        hm = heat_sum[name] / cnt
        hmin, hmax = hm.min(), hm.max()
        if hmax - hmin > 1e-8:
            hm = (hm - hmin) / (hmax - hmin)
        final_heatmaps[name] = hm.astype(np.float32)

    return prediction, final_heatmaps


def apply_heatmap(image_np, heatmap, alpha=0.5):
    """Overlay Grad-CAM heatmap on image using jet colormap."""
    cmap = plt.get_cmap('inferno')
    colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    return ((1 - alpha) * image_np + alpha * colored).astype(np.uint8)


def colorize_mask(mask):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cid, color in enumerate(CLASS_COLORS):
        out[mask == cid] = color
    return out


def visualize_image(image_path, model, device, output_dir,
                    crop_size=512, stride=384, gt_mask=None):
    """Generate Grad-CAM visualization for one full-resolution image."""
    bn = os.path.splitext(os.path.basename(image_path))[0]
    img_pil = Image.open(image_path).convert('RGB')
    img_np = np.array(img_pil)
    img_tensor = TF.normalize(TF.to_tensor(img_pil), MEAN, STD)

    layer_names = [cfg[0] for cfg in LAYER_CONFIG]

    print(f'  Computing Grad-CAM (sliding window)...')
    prediction, heatmaps = sliding_window_gradcam(
        model, img_tensor, layer_names, crop_size, stride, device
    )

    # Build columns
    cols = [('Original Image', img_np)]
    for lname, label in LAYER_CONFIG:
        overlay = apply_heatmap(img_np, heatmaps[lname], alpha=0.7)
        cols.append((label, overlay))

    # Prediction overlay
    pred_color = colorize_mask(prediction)
    labeled = img_np.copy()
    fg = prediction > 0
    labeled[fg] = (0.5 * img_np[fg] + 0.5 * pred_color[fg]).astype(np.uint8)
    cols.append(('Labeled Image', labeled))

    # GT if available
    if gt_mask is not None:
        gt_color = colorize_mask(gt_mask)
        gt_overlay = img_np.copy()
        fg_gt = gt_mask > 0
        gt_overlay[fg_gt] = (0.5 * img_np[fg_gt] + 0.5 * gt_color[fg_gt]).astype(np.uint8)
        cols.append(('Ground Truth', gt_overlay))

    # Plot
    n_cols = len(cols)
    H, W = img_np.shape[:2]
    aspect = H / W
    fig_w = 4.5 * n_cols
    fig_h = max(fig_w / n_cols * aspect, 3.5)

    fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, fig_h))
    for ax, (title, img) in zip(axes, cols):
        ax.imshow(img)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

    plt.tight_layout(pad=0.3)
    save_path = os.path.join(output_dir, f'{bn}_gradcam.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'  Saved: {save_path}')
    return save_path


def main():
    parser = argparse.ArgumentParser(description='Grad-CAM Visualization')
    parser.add_argument('--image', type=str, nargs='+', default=None)
    parser.add_argument('--data_root', type=str, default='Mos2_data')
    parser.add_argument('--split', type=str, default=None,
                        choices=['val', 'test'])
    parser.add_argument('--split_dir', type=str, default='splits/')
    parser.add_argument('--max_images', type=int, default=5)
    parser.add_argument('--model', type=str, default='small',
                        choices=['tiny', 'small', 'base'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--stride', type=int, default=384)
    parser.add_argument('--output', type=str, default='output/gradcam/')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model_fn = {'tiny': repela_net_tiny, 'small': repela_net_small,
                'base': repela_net_base}[args.model]
    model = model_fn(num_classes=args.num_classes).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'Loaded: {args.checkpoint} (Epoch {ckpt["epoch"]+1})')

    # Collect images
    if args.image:
        paths = args.image
    elif args.split:
        sf = os.path.join(args.split_dir, f'{args.split}.txt')
        with open(sf) as f:
            bns = [l.strip() for l in f if l.strip()][:args.max_images]
        paths = [os.path.join(args.data_root, 'ori', 'MoS2', f'{b}.jpg')
                 for b in bns]
    else:
        parser.error('Provide --image or --split')

    os.makedirs(args.output, exist_ok=True)
    mask_dir = os.path.join(args.data_root, 'mask')

    for img_path in paths:
        bn = os.path.splitext(os.path.basename(img_path))[0]
        sz = Image.open(img_path).size
        print(f'\n[{bn}] ({sz[0]}x{sz[1]})')

        gt = None
        mp = os.path.join(mask_dir, f'{bn}.png')
        if os.path.exists(mp):
            gt = np.array(Image.open(mp))

        visualize_image(img_path, model, device, args.output,
                        args.crop_size, args.stride, gt)

    print(f'\nAll Grad-CAM results saved to: {args.output}')


if __name__ == '__main__':
    main()
