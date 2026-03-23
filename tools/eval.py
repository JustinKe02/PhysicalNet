"""
Unified Evaluation Script for RepELA-Net, smp baselines, and ablation variants.

Computes mIoU, per-class IoU, F1, pixel accuracy on a specified split
using deterministic sliding-window inference.

Usage (from project root):
    # RepELA-Small on test set
    python tools/eval.py --model repela_small --split test \
        --checkpoint output/.../best_model.pth

    # smp baseline
    python tools/eval.py --model unet_r18 --split test \
        --checkpoint output/baselines/.../best_model.pth

    # Ablation variant
    python tools/eval.py --model repela_small --ablation no_ela --split test \
        --checkpoint output/ablation/.../best_model.pth
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
import glob
import time
import argparse
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from models.repela_net import repela_net_tiny, repela_net_small, repela_net_base, infer_use_cse
from train_ablation import build_ablation_model, ALL_ABLATIONS, ABLATION_NAMES
from utils.metrics import SegmentationMetrics

# ── Model Registry (mirrors tools/train.py) ──
REPELA_MODELS = {
    'repela_tiny': repela_net_tiny,
    'repela_small': repela_net_small,
    'repela_base': repela_net_base,
}

SMP_MODEL_SPECS = {
    # Tier 1: Lightweight (<5M params)
    'fpn_mnv3s':       ('FPN',            'timm-mobilenetv3_small_100'),
    'unet_mnv3s':      ('Unet',           'timm-mobilenetv3_small_100'),
    'fpn_mv2':         ('FPN',            'mobilenet_v2'),
    'deeplabv3p_mv2':  ('DeepLabV3Plus',  'mobilenet_v2'),
    'deeplabv3p_effb0':('DeepLabV3Plus',  'efficientnet-b0'),
    # Tier 2: Standard (>10M params)
    'unet_r18':        ('Unet',           'resnet18'),
    'unet_r34':        ('Unet',           'resnet34'),
    'deeplabv3p_r18':  ('DeepLabV3Plus',  'resnet18'),
    'pspnet_r18':      ('PSPNet',         'resnet18'),
    'fpn_r18':         ('FPN',            'resnet18'),
    'unet_mit_b0':     ('Unet',           'mit_b0'),
}

ALL_MODEL_NAMES = list(REPELA_MODELS.keys()) + list(SMP_MODEL_SPECS.keys())

CLASSES = ['background', 'monolayer', 'fewlayer', 'multilayer']
CLASS_LABELS_SHORT = ['BG', '1L', '2L', 'ML']  # For confusion matrix axis
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def plot_confusion_matrix(confusion, class_labels, save_path):
    """Plot normalized confusion matrix (matching 2D-TLK paper Fig.3b style)."""
    # Row-normalize (per ground-truth class)
    row_sums = confusion.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1)  # avoid div-by-zero
    cm_norm = confusion / row_sums * 100  # percentage

    n = len(class_labels)
    fig, ax = plt.subplots(figsize=(5.5, 5))

    # Dark blue colormap similar to reference paper
    cmap = plt.cm.Blues
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=cmap,
                   vmin=0, vmax=100)

    # Annotate each cell with percentage
    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            color = 'white' if val > 50 else 'black'
            ax.text(j, i, f'{val:.2f}%', ha='center', va='center',
                    fontsize=13, fontweight='bold', color=color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_labels, fontsize=12)
    ax.set_yticklabels(class_labels, fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=13)
    ax.set_ylabel('Ground Truth Label', fontsize=13)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f'Confusion matrix saved: {save_path}')


def load_split(split_dir, split):
    """Load image basenames from split file."""
    path = os.path.join(split_dir, f'{split}.txt')
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


@torch.no_grad()
def predict_full_image(model, img_tensor, device, is_smp=False):
    """Full-image inference (no sliding window).

    Sends the entire image through the model at once.
    For smp models, pads input to multiple of 32.
    Returns: prediction [H,W] numpy int array, or None if OOM.
    """
    try:
        _, H, W = img_tensor.shape
        img = img_tensor.unsqueeze(0).to(device)
        if is_smp:
            pad_h = (32 - H % 32) % 32
            pad_w = (32 - W % 32) % 32
            if pad_h > 0 or pad_w > 0:
                img = F.pad(img, [0, pad_w, 0, pad_h], mode='reflect')
        output = model(img)
        logits = output[0] if isinstance(output, tuple) else output
        if is_smp:
            logits = logits[:, :, :H, :W]
        pred = logits.argmax(dim=1)[0].cpu().numpy()
        return pred
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None  # Caller should fallback to sliding window


def sliding_window_predict(model, img_tensor, crop_size, stride, device, is_smp=False):
    """Full-coverage sliding window inference.

    Guarantees every pixel is covered at least once by explicitly
    adding boundary windows for the last row, column, and corner.
    For smp models, pads each crop to a multiple of 32.

    Returns: prediction [H,W] numpy int array
    """
    _, H, W = img_tensor.shape
    num_classes = 4

    pred_sum = torch.zeros(num_classes, H, W, dtype=torch.float32, device=device)
    count = torch.zeros(H, W, dtype=torch.float32, device=device)

    # Pad if image smaller than crop
    pad_h = max(0, crop_size - H)
    pad_w = max(0, crop_size - W)
    if pad_h > 0 or pad_w > 0:
        img_tensor = F.pad(img_tensor, [0, pad_w, 0, pad_h], mode='reflect')
    _, pH, pW = img_tensor.shape

    # Build window positions with guaranteed full coverage
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
            cH, cW = crop.shape[2], crop.shape[3]
            if is_smp:
                smp_pad_h = (32 - cH % 32) % 32
                smp_pad_w = (32 - cW % 32) % 32
                if smp_pad_h > 0 or smp_pad_w > 0:
                    crop = F.pad(crop, [0, smp_pad_w, 0, smp_pad_h], mode='reflect')
            with torch.no_grad():
                output = model(crop)
                logits = output[0] if isinstance(output, tuple) else output
                if is_smp:
                    logits = logits[:, :, :cH, :cW]
                probs = F.softmax(logits, dim=1)[0]
            y_end = min(y + crop_size, H)
            x_end = min(x + crop_size, W)
            pred_sum[:, y:y_end, x:x_end] += probs[:, :y_end-y, :x_end-x]
            count[y:y_end, x:x_end] += 1

    count = count.clamp(min=1)
    return (pred_sum / count.unsqueeze(0)).argmax(dim=0).cpu().numpy()


def smart_predict(model, img_tensor, crop_size, stride, device, use_full=True, is_smp=False):
    """Try full-image inference first, fallback to sliding window on OOM."""
    if use_full:
        pred = predict_full_image(model, img_tensor, device, is_smp=is_smp)
        if pred is not None:
            return pred, 'full'
    pred = sliding_window_predict(model, img_tensor, crop_size, stride, device, is_smp=is_smp)
    return pred, 'sliding'


@torch.no_grad()
def predict_tta(model, img_tensor, device, is_smp=False):
    """Test-Time Augmentation: average predictions over 4 transforms.

    Transforms: original, horizontal flip, vertical flip, 180° rotation.
    Averages softmax probabilities then takes argmax.
    For smp models, pads input to multiple of 32.
    """
    C, H, W = img_tensor.shape
    num_classes = 4
    prob_sum = torch.zeros(num_classes, H, W, dtype=torch.float32, device=device)

    transforms = [
        lambda x: x,                          # original
        lambda x: x.flip(2),                   # horizontal flip
        lambda x: x.flip(1),                   # vertical flip
        lambda x: x.flip(1).flip(2),           # 180° rotation
    ]
    inverse_transforms = [
        lambda x: x,
        lambda x: x.flip(2),
        lambda x: x.flip(1),
        lambda x: x.flip(1).flip(2),
    ]

    success_count = 0
    for tfm, inv_tfm in zip(transforms, inverse_transforms):
        augmented = tfm(img_tensor)
        try:
            inp = augmented.unsqueeze(0).to(device)
            if is_smp:
                pad_h = (32 - H % 32) % 32
                pad_w = (32 - W % 32) % 32
                if pad_h > 0 or pad_w > 0:
                    inp = F.pad(inp, [0, pad_w, 0, pad_h], mode='reflect')
            output = model(inp)
            logits = output[0] if isinstance(output, tuple) else output
            if is_smp:
                logits = logits[:, :, :H, :W]
            probs = F.softmax(logits, dim=1)[0]
            probs = inv_tfm(probs)
            prob_sum += probs
            success_count += 1
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue

    if success_count == 0:
        raise RuntimeError('TTA failed: all augmentation branches ran out of GPU memory')
    if success_count < len(transforms):
        import warnings
        warnings.warn(f'TTA: only {success_count}/{len(transforms)} branches succeeded (OOM on others)')

    return prob_sum.argmax(dim=0).cpu().numpy()


def _build_smp_model(model_name, num_classes):
    """Lazily import smp and build a baseline model."""
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        raise ImportError(
            f'segmentation_models_pytorch is required for "{model_name}". '
            f'Install with: pip install segmentation-models-pytorch')
    arch, encoder = SMP_MODEL_SPECS[model_name]
    cls = getattr(smp, arch)
    return cls(encoder_name=encoder, encoder_weights='imagenet',
               in_channels=3, classes=num_classes)


def main():
    parser = argparse.ArgumentParser(description='Unified Evaluation')
    parser.add_argument('--data_root', type=str, default='Mos2_data')
    parser.add_argument('--split_dir', type=str, default='splits/')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test', 'train'])
    parser.add_argument('--model', type=str, default='repela_small',
                        choices=ALL_MODEL_NAMES)
    parser.add_argument('--ablation', type=str, default=None,
                        choices=ALL_ABLATIONS + [None],
                        help='Ablation variant (only with repela_small)')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--deploy_model', type=str, default=None)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--stride', type=int, default=384)
    parser.add_argument('--output', type=str, default=None,
                        help='Output dir for per-image results (optional)')
    parser.add_argument('--tta', action='store_true', default=False,
                        help='Enable test-time augmentation')
    parser.add_argument('--use_cse', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable ColorSpaceEnhancement')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_smp = args.model in SMP_MODEL_SPECS
    print(f'Device: {device}')

    # ── Build model ──
    if args.ablation:
        # Ablation variant
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model = build_ablation_model(args.ablation, args.num_classes,
                                     deep_supervision=False)
        model.load_state_dict(ckpt['model'], strict=False)
        model_display = ABLATION_NAMES.get(args.ablation, args.ablation)
        print(f'Loaded ablation: {model_display}')
        print(f'  Checkpoint: {args.checkpoint}')

    elif is_smp:
        # smp baseline
        model = _build_smp_model(args.model, args.num_classes)
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        sd = ckpt['model'] if 'model' in ckpt else ckpt
        model.load_state_dict(sd, strict=False)
        model_display = args.model
        print(f'Loaded smp model: {args.model}')
        print(f'  Checkpoint: {args.checkpoint}')

    elif args.model in REPELA_MODELS:
        # RepELA-Net
        model_fn = REPELA_MODELS[args.model]
        if args.deploy_model:
            use_cse = args.use_cse
            model = model_fn(num_classes=args.num_classes, deploy=True,
                             use_cse=use_cse)
            sd = torch.load(args.deploy_model, map_location=device, weights_only=True)
            model.load_state_dict(sd, strict=False)
            print(f'Loaded deploy model: {args.deploy_model}')
        else:
            ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
            use_cse = infer_use_cse(ckpt, cli_use_cse=args.use_cse)
            model = model_fn(num_classes=args.num_classes, use_cse=use_cse)
            model.load_state_dict(ckpt['model'], strict=False)
            print(f'Loaded: {args.checkpoint} (Epoch {ckpt["epoch"]+1}, '
                  f'mIoU={ckpt.get("best_miou", "?"):.4f})')
        model_display = f'RepELA-Net-{args.model.split("_")[1].title()}'
    else:
        raise ValueError(f'Unknown model: {args.model}')

    model = model.to(device)
    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'Model: {model_display}, {params:.2f}M params')

    # Load split
    basenames = load_split(args.split_dir, args.split)
    img_dir = os.path.join(args.data_root, 'ori', 'MoS2')
    mask_dir = os.path.join(args.data_root, 'mask')
    print(f'\nEvaluating on {args.split} set: {len(basenames)} images')
    print(f'Inference: full-image (fallback: sliding window crop={args.crop_size}, stride={args.stride})')

    if args.output:
        os.makedirs(args.output, exist_ok=True)

    metrics = SegmentationMetrics(args.num_classes)
    per_image = []
    total_time = 0

    for i, bn in enumerate(basenames):
        img_path = os.path.join(img_dir, f'{bn}.jpg')
        mask_path = os.path.join(mask_dir, f'{bn}.png')

        img = Image.open(img_path).convert('RGB')
        gt = np.array(Image.open(mask_path))

        img_tensor = TF.normalize(TF.to_tensor(img), MEAN, STD)

        t0 = time.time()
        if args.tta:
            pred = predict_tta(model, img_tensor, device, is_smp=is_smp)
            method = 'tta'
        else:
            pred, method = smart_predict(model, img_tensor, args.crop_size, args.stride, device, is_smp=is_smp)
        elapsed = time.time() - t0
        total_time += elapsed

        # Per-image IoU
        img_metrics = SegmentationMetrics(args.num_classes)
        img_metrics.update(pred, gt)
        img_results = img_metrics.get_results()
        per_image.append((bn, img_results))

        # Global metrics
        metrics.update(pred, gt)

        iou_str = ' '.join([f'{CLASSES[c][:4]}={img_results["per_class_iou"][c]:.3f}'
                            for c in range(args.num_classes)])
        print(f'  [{i+1}/{len(basenames)}] {bn}: mIoU={img_results["mIoU"]:.4f} '
              f'{iou_str} ({elapsed:.2f}s)')

        # Save raw mask if output dir specified
        if args.output:
            raw_mask_path = os.path.join(args.output, f'{bn}_pred.png')
            Image.fromarray(pred.astype(np.uint8)).save(raw_mask_path)

    # Overall results
    results = metrics.get_results()
    print(f'\n{"=" * 60}')
    print(f'{model_display} | {args.split} set | {len(basenames)} images')
    print(f'Sliding window: crop={args.crop_size}, stride={args.stride}')
    print(f'{"=" * 60}')
    print(f'  mIoU:       {results["mIoU"]:.4f}')
    print(f'  Pixel Acc:  {results["pixel_acc"]:.4f}')
    print(f'  Mean F1:    {results["mean_f1"]:.4f}')
    for c in range(args.num_classes):
        print(f'  {CLASSES[c]:12s}  IoU={results["per_class_iou"][c]:.4f}  '
              f'F1={results["f1"][c]:.4f}  Acc={results["class_acc"][c]:.4f}')
    print(f'  Avg time:   {total_time/len(basenames):.3f}s/image')
    print(f'{"=" * 60}')

    # Save to file
    if args.output:
        results_path = os.path.join(args.output, f'{args.split}_metrics.txt')
        with open(results_path, 'w') as f:
            f.write(f'{model_display} | {args.split} set | {len(basenames)} images\n')
            f.write(f'Checkpoint: {args.checkpoint}\n')
            f.write(f'Sliding window: crop={args.crop_size}, stride={args.stride}\n\n')
            f.write(f'mIoU:       {results["mIoU"]:.4f}\n')
            f.write(f'Pixel Acc:  {results["pixel_acc"]:.4f}\n')
            f.write(f'Mean F1:    {results["mean_f1"]:.4f}\n\n')
            for c in range(args.num_classes):
                f.write(f'{CLASSES[c]:12s}  IoU={results["per_class_iou"][c]:.4f}  '
                        f'F1={results["f1"][c]:.4f}  Acc={results["class_acc"][c]:.4f}\n')
            f.write(f'\nPer-image results:\n')
            for bn, r in per_image:
                f.write(f'  {bn}: mIoU={r["mIoU"]:.4f}\n')
        print(f'\nResults saved to: {results_path}')

        # Plot confusion matrix
        cm_path = os.path.join(args.output, f'{args.split}_confusion_matrix.png')
        plot_confusion_matrix(metrics.confusion_matrix, CLASS_LABELS_SHORT, cm_path)


if __name__ == '__main__':
    main()
