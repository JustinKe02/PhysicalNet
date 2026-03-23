"""
RepELA-Net Inference / Visualization Script.

Responsibilities: visual outputs ONLY (color masks, overlays, comparison figures).
For quantitative metrics, use eval.py instead.

Features:
  - Sliding window inference (default for high-res images)
  - Visualization: original / GT / prediction / overlay
  - Saves both raw single-channel masks and color visualizations

Usage:
    # Visualize val set predictions
    python inference.py --data_root Mos2_data --split val \
        --checkpoint output/repela_small_*/best_model.pth

    # Single image
    python inference.py --image Mos2_data/ori/MoS2/m1.jpg \
        --checkpoint output/repela_small_*/best_model.pth
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import os
import glob
import time
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from models.repela_net import repela_net_tiny, repela_net_small, repela_net_base, infer_use_cse
from train_ablation import build_ablation_model

CLASSES = ['background', 'monolayer', 'fewlayer', 'multilayer']
CLASS_COLORS = np.array([
    [0, 0, 0],        # background
    [239, 41, 41],     # monolayer
    [0, 170, 0],       # fewlayer
    [114, 159, 207],   # multilayer
], dtype=np.uint8)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def sliding_window_predict(model, img_tensor, crop_size, stride, device):
    """Full-coverage sliding window inference.

    Guaranteed complete coverage: explicitly adds boundary windows.
    """
    _, H, W = img_tensor.shape
    num_classes = 4

    pred_sum = torch.zeros(num_classes, H, W, dtype=torch.float32, device=device)
    count = torch.zeros(H, W, dtype=torch.float32, device=device)

    pad_h = max(0, crop_size - H)
    pad_w = max(0, crop_size - W)
    if pad_h > 0 or pad_w > 0:
        img_tensor = F.pad(img_tensor, [0, pad_w, 0, pad_h], mode='reflect')
    _, pH, pW = img_tensor.shape

    # Full coverage: stride grid + boundary windows
    ys = sorted(set(
        list(range(0, max(1, pH - crop_size + 1), stride)) +
        [max(0, pH - crop_size)]
    ))
    xs = sorted(set(
        list(range(0, max(1, pW - crop_size + 1), stride)) +
        [max(0, pW - crop_size)]
    ))

    for y in ys:
        for x in xs:
            crop = img_tensor[:, y:y+crop_size, x:x+crop_size].unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(crop)
                logits = output[0] if isinstance(output, tuple) else output
                probs = F.softmax(logits, dim=1)[0]
            y_end = min(y + crop_size, H)
            x_end = min(x + crop_size, W)
            pred_sum[:, y:y_end, x:x_end] += probs[:, :y_end-y, :x_end-x]
            count[y:y_end, x:x_end] += 1

    count = count.clamp(min=1)
    return (pred_sum / count.unsqueeze(0)).argmax(dim=0).cpu().numpy()


def colorize_mask(mask):
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, color in enumerate(CLASS_COLORS):
        out[mask == cid] = color
    return out


def visualize_result(image_np, prediction, save_path, gt_mask=None):
    color_pred = colorize_mask(prediction)

    overlay = image_np.copy()
    mask_fg = prediction > 0
    overlay[mask_fg] = (0.6 * image_np[mask_fg] + 0.4 * color_pred[mask_fg]).astype(np.uint8)

    if gt_mask is not None:
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        for ax, img, title in zip(axes,
            [image_np, colorize_mask(gt_mask), color_pred, overlay],
            ['Original', 'Ground Truth', 'Prediction', 'Overlay']):
            ax.imshow(img); ax.set_title(title, fontsize=14); ax.axis('off')
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for ax, img, title in zip(axes,
            [image_np, color_pred, overlay],
            ['Original', 'Prediction', 'Overlay']):
            ax.imshow(img); ax.set_title(title, fontsize=14); ax.axis('off')

    legend = [Patch(facecolor=np.array(c)/255., label=n)
              for n, c in zip(CLASSES[1:], CLASS_COLORS[1:])]
    fig.legend(handles=legend, loc='lower center', ncol=3, fontsize=12,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def load_model(args, device):
    # Ablation model (e.g. no_color)
    if args.ablation:
        if args.deploy_model:
            model = build_ablation_model(args.ablation, num_classes=args.num_classes,
                                          deep_supervision=False).to(device)
            model.switch_to_deploy()
            sd = torch.load(args.deploy_model, map_location=device, weights_only=True)
            model.load_state_dict(sd, strict=False)
            print(f'Loaded deploy model: {args.deploy_model} (ablation={args.ablation})')
        else:
            ds = getattr(args, 'deep_supervision', False)
            model = build_ablation_model(args.ablation, num_classes=args.num_classes,
                                          deep_supervision=ds).to(device)
            ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model'], strict=True)
            print(f'Loaded: {args.checkpoint} (ablation={args.ablation}, Epoch {ckpt["epoch"]+1})')
    else:
        use_cse = getattr(args, 'use_cse', False)
        model_fn = {'tiny': repela_net_tiny, 'small': repela_net_small,
                    'base': repela_net_base}[args.model]

        if args.deploy_model:
            model = model_fn(num_classes=args.num_classes, deploy=True,
                             use_cse=use_cse).to(device)
            sd = torch.load(args.deploy_model, map_location=device, weights_only=True)
            model.load_state_dict(sd, strict=False)
            print(f'Loaded deploy model: {args.deploy_model} (use_cse={use_cse})')
        else:
            ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
            use_cse = infer_use_cse(ckpt, cli_use_cse=use_cse)
            print(f'Inferred use_cse={use_cse}')
            model = model_fn(num_classes=args.num_classes,
                             use_cse=use_cse).to(device)
            model.load_state_dict(ckpt['model'], strict=False)
            print(f'Loaded: {args.checkpoint} (Epoch {ckpt["epoch"]+1})')

    model.eval()
    return model


def get_image_paths(args):
    """Resolve image list from args."""
    mask_dir = None

    if args.split:
        # Use split file
        split_file = os.path.join(args.split_dir, f'{args.split}.txt')
        with open(split_file) as f:
            basenames = [l.strip() for l in f if l.strip()]
        img_dir = os.path.join(args.data_root, 'ori', 'MoS2')
        mask_dir = os.path.join(args.data_root, 'mask')
        paths = [(os.path.join(img_dir, f'{bn}.jpg'), bn) for bn in basenames]
    elif args.image:
        bn = os.path.splitext(os.path.basename(args.image))[0]
        paths = [(args.image, bn)]
        mask_dir = args.mask_dir
    elif args.image_dir:
        imgs = sorted(glob.glob(os.path.join(args.image_dir, '*.jpg')) +
                       glob.glob(os.path.join(args.image_dir, '*.png')))
        paths = [(p, os.path.splitext(os.path.basename(p))[0]) for p in imgs]
        mask_dir = args.mask_dir
    else:
        raise ValueError('Provide --split, --image, or --image_dir')

    return paths, mask_dir


def main():
    parser = argparse.ArgumentParser(description='RepELA-Net Inference (Visualization)')

    # Input
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--image_dir', type=str, default=None)
    parser.add_argument('--data_root', type=str, default='Mos2_data')
    parser.add_argument('--split', type=str, default=None,
                        choices=['train', 'val', 'test'],
                        help='Use a fixed split file')
    parser.add_argument('--split_dir', type=str, default='splits/')
    parser.add_argument('--mask_dir', type=str, default=None)

    # Model
    parser.add_argument('--model', type=str, default='small',
                        choices=['tiny', 'small', 'base'])
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--deploy_model', type=str, default=None)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--ablation', type=str, default=None,
                        help='Ablation variant (e.g. no_color)')
    parser.add_argument('--deep_supervision', action='store_true', default=False,
                        help='Enable deep supervision (match training config)')
    parser.add_argument('--use_cse', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable ColorSpaceEnhancement (--use_cse / --no-use_cse)')

    # Sliding window (default ON for 2560x1922 images)
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--stride', type=int, default=384)

    # Output
    parser.add_argument('--output', type=str, default='output/inference/')

    args = parser.parse_args()
    if args.checkpoint is None and args.deploy_model is None:
        parser.error('Provide --checkpoint or --deploy_model')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args, device)
    paths, mask_dir = get_image_paths(args)

    print(f'\nProcessing {len(paths)} images (sliding window: '
          f'crop={args.crop_size}, stride={args.stride})')
    os.makedirs(args.output, exist_ok=True)

    for i, (img_path, bn) in enumerate(paths):
        print(f'[{i+1}/{len(paths)}] {bn}', end=' ')

        img_pil = Image.open(img_path).convert('RGB')
        img_np = np.array(img_pil)
        img_tensor = TF.normalize(TF.to_tensor(img_pil), MEAN, STD)

        t0 = time.time()
        pred = sliding_window_predict(model, img_tensor, args.crop_size, args.stride, device)
        print(f'({time.time()-t0:.2f}s)')

        # GT mask
        gt_mask = None
        if mask_dir:
            mp = os.path.join(mask_dir, f'{bn}.png')
            if os.path.exists(mp):
                gt_mask = np.array(Image.open(mp))

        # Save visualization
        visualize_result(img_np, pred,
                         os.path.join(args.output, f'{bn}_result.png'), gt_mask)

        # Save raw single-channel class-index mask (item 7)
        Image.fromarray(pred.astype(np.uint8)).save(
            os.path.join(args.output, f'{bn}_pred.png'))

        # Save color mask
        Image.fromarray(colorize_mask(pred)).save(
            os.path.join(args.output, f'{bn}_color.png'))

    print(f'\nResults saved to: {args.output}')


if __name__ == '__main__':
    main()
