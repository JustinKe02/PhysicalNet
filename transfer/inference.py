"""
Cross-Material Inference Visualization.

Runs inference on fine-tuned models using domain-adaptive normalization.
Generates side-by-side comparison: Original | Prediction | Ground Truth.

Usage:
    python inference_finetune.py \
        --data_root "other data/graphene" \
        --checkpoint output/finetune_graphene_v3/best_model.pth \
        --num_classes 3 --name graphene \
        --split val
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
# Ensure cwd = project root so relative paths (Mos2_data/, splits/) work
import os as _os
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
_os.chdir(_PROJECT_ROOT)

import os
import json
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
from transfer.material_dataset import compute_dataset_stats

CLASS_COLORS = {
    3: np.array([[0,0,0],[239,41,41],[114,159,207]], dtype=np.uint8),        # graphene
    4: np.array([[0,0,0],[239,41,41],[0,170,0],[114,159,207]], dtype=np.uint8),  # WS2/MoS2
}
CLASS_NAMES = {
    3: ['Background', 'Monolayer', '>1L'],
    4: ['Background', 'Monolayer', 'Fewlayer', 'Multilayer'],
}


def sliding_window_predict(model, img_tensor, crop_size, stride, device):
    _, H, W = img_tensor.shape
    nc = model.num_classes
    pred_sum = torch.zeros(nc, H, W, dtype=torch.float32, device=device)
    count = torch.zeros(H, W, dtype=torch.float32, device=device)

    pad_h, pad_w = max(0, crop_size - H), max(0, crop_size - W)
    padded = F.pad(img_tensor, [0, pad_w, 0, pad_h], mode='reflect') if (pad_h > 0 or pad_w > 0) else img_tensor
    _, pH, pW = padded.shape

    ys = sorted(set(list(range(0, max(1, pH-crop_size+1), stride)) + [max(0, pH-crop_size)]))
    xs = sorted(set(list(range(0, max(1, pW-crop_size+1), stride)) + [max(0, pW-crop_size)]))

    for y in ys:
        for x in xs:
            crop = padded[:, y:y+crop_size, x:x+crop_size].unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(crop)
                logits = output[0] if isinstance(output, tuple) else output
                probs = F.softmax(logits, dim=1)[0]
            ye, xe = min(y+crop_size, H), min(x+crop_size, W)
            pred_sum[:, y:ye, x:xe] += probs[:, :ye-y, :xe-x]
            count[y:ye, x:xe] += 1
    count = count.clamp(min=1)
    return (pred_sum / count.unsqueeze(0)).argmax(dim=0).cpu().numpy()


def colorize(mask, nc):
    colors = CLASS_COLORS[nc]
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c, col in enumerate(colors):
        out[mask == c] = col
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--model', type=str, default='small')
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--stride', type=int, default=384)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--max_images', type=int, default=20)
    args = parser.parse_args()

    if args.output is None:
        args.output = f'output/inference_{args.name}/'
    os.makedirs(args.output, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load domain-adaptive stats
    stats = compute_dataset_stats(args.data_root, split='train')
    ds_mean, ds_std = stats['mean'], stats['std']
    print(f'Domain stats: mean={[f"{m:.3f}" for m in ds_mean]}, '
          f'std={[f"{s:.3f}" for s in ds_std]}')

    # Load model
    model_fn = {'tiny': repela_net_tiny, 'small': repela_net_small,
                'base': repela_net_base}[args.model]
    model = model_fn(num_classes=args.num_classes,
                     norm_mean=ds_mean, norm_std=ds_std).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_sd = ckpt['model'] if 'model' in ckpt else ckpt

    # Exclude source-domain norm buffers (keep target-domain values)
    for k in list(ckpt_sd.keys()):
        if 'color_enhance.mean' in k or 'color_enhance.std' in k:
            del ckpt_sd[k]

    missing, unexpected = model.load_state_dict(ckpt_sd, strict=False)
    if missing:
        print(f'  Note: {len(missing)} missing keys (new model has extra params)')
    if unexpected:
        print(f'  Note: {len(unexpected)} unexpected keys in checkpoint (ignored)')
    model.eval()
    epoch_info = ckpt.get('epoch', -1) + 1 if isinstance(ckpt, dict) and 'epoch' in ckpt else '?'
    miou_info = f"{ckpt.get('best_miou', 0):.4f}" if isinstance(ckpt, dict) and 'best_miou' in ckpt else '?'
    print(f'Loaded: {args.checkpoint} (Epoch {epoch_info}, mIoU={miou_info})')

    # Collect images
    import glob
    img_dir = os.path.join(args.data_root, 'img_dir', args.split)
    ann_dir = os.path.join(args.data_root, 'ann_dir', args.split)
    imgs = sorted(glob.glob(os.path.join(img_dir, '*.jpg')) +
                  glob.glob(os.path.join(img_dir, '*.png')))[:args.max_images]

    print(f'\nInferring on {len(imgs)} images from {args.split} set\n')

    for i, img_path in enumerate(imgs):
        bn = os.path.splitext(os.path.basename(img_path))[0]
        img_pil = Image.open(img_path).convert('RGB')
        img_np = np.array(img_pil)

        img_tensor = TF.normalize(TF.to_tensor(img_pil), ds_mean, ds_std)
        pred = sliding_window_predict(model, img_tensor, args.crop_size,
                                      args.stride, device)

        # GT
        gt_path = os.path.join(ann_dir, f'{bn}.png')
        has_gt = os.path.exists(gt_path)
        gt = np.array(Image.open(gt_path)) if has_gt else None

        # Create figure
        n_cols = 3 if has_gt else 2
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))

        axes[0].imshow(img_np)
        axes[0].set_title('Original', fontsize=13, fontweight='bold')
        axes[0].axis('off')

        pred_color = colorize(pred, args.num_classes)
        pred_overlay = img_np.copy()
        fg = pred > 0
        pred_overlay[fg] = (0.5 * img_np[fg] + 0.5 * pred_color[fg]).astype(np.uint8)
        axes[1].imshow(pred_overlay)
        axes[1].set_title('Prediction', fontsize=13, fontweight='bold')
        axes[1].axis('off')

        if has_gt:
            gt_color = colorize(gt, args.num_classes)
            gt_overlay = img_np.copy()
            fg_gt = gt > 0
            gt_overlay[fg_gt] = (0.5 * img_np[fg_gt] + 0.5 * gt_color[fg_gt]).astype(np.uint8)
            axes[2].imshow(gt_overlay)
            axes[2].set_title('Ground Truth', fontsize=13, fontweight='bold')
            axes[2].axis('off')

        plt.tight_layout(pad=0.3)
        save_path = os.path.join(args.output, f'{bn}_result.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f'  [{i+1}/{len(imgs)}] {bn}')

    print(f'\nAll saved to: {args.output}')


if __name__ == '__main__':
    main()
