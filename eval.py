"""
RepELA-Net Evaluation Script.

Dedicated evaluation: computes mIoU, per-class IoU, F1, pixel accuracy
on a specified split using deterministic sliding-window inference.

Usage:
    # Evaluate on val set
    python eval.py --data_root Mos2_data --split val \
        --checkpoint output/repela_small_*/best_model.pth

    # Evaluate on test set
    python eval.py --data_root Mos2_data --split test \
        --checkpoint output/repela_small_*/best_model.pth
"""

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

from models.repela_net import repela_net_tiny, repela_net_small, repela_net_base
from utils.metrics import SegmentationMetrics

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
def predict_full_image(model, img_tensor, device):
    """Full-image inference (no sliding window).

    Sends the entire image through the model at once.
    Returns: prediction [H,W] numpy int array, or None if OOM.
    """
    try:
        img = img_tensor.unsqueeze(0).to(device)
        logits = model(img)
        pred = logits.argmax(dim=1)[0].cpu().numpy()
        return pred
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None  # Caller should fallback to sliding window


def sliding_window_predict(model, img_tensor, crop_size, stride, device):
    """Full-coverage sliding window inference.

    Guarantees every pixel is covered at least once by explicitly
    adding boundary windows for the last row, column, and corner.

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
            with torch.no_grad():
                probs = F.softmax(model(crop), dim=1)[0]
            y_end = min(y + crop_size, H)
            x_end = min(x + crop_size, W)
            pred_sum[:, y:y_end, x:x_end] += probs[:, :y_end-y, :x_end-x]
            count[y:y_end, x:x_end] += 1

    count = count.clamp(min=1)
    return (pred_sum / count.unsqueeze(0)).argmax(dim=0).cpu().numpy()


def smart_predict(model, img_tensor, crop_size, stride, device, use_full=True):
    """Try full-image inference first, fallback to sliding window on OOM."""
    if use_full:
        pred = predict_full_image(model, img_tensor, device)
        if pred is not None:
            return pred, 'full'
    pred = sliding_window_predict(model, img_tensor, crop_size, stride, device)
    return pred, 'sliding'


@torch.no_grad()
def predict_tta(model, img_tensor, device):
    """Test-Time Augmentation: average predictions over 4 transforms.

    Transforms: original, horizontal flip, vertical flip, 180° rotation.
    Averages softmax probabilities then takes argmax.
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
            output = model(augmented.unsqueeze(0).to(device))
            logits = output[0] if isinstance(output, tuple) else output
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


def main():
    parser = argparse.ArgumentParser(description='RepELA-Net Evaluation')
    parser.add_argument('--data_root', type=str, default='Mos2_data')
    parser.add_argument('--split_dir', type=str, default='splits/')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test', 'train'])
    parser.add_argument('--model', type=str, default='small', choices=['tiny', 'small', 'base'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--deploy_model', type=str, default=None)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--stride', type=int, default=384)
    parser.add_argument('--output', type=str, default=None,
                        help='Output dir for per-image results (optional)')
    parser.add_argument('--tta', action='store_true', default=False,
                        help='Enable test-time augmentation (4x slower, ~+0.5-1%% mIoU)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load model
    model_fn = {'tiny': repela_net_tiny, 'small': repela_net_small,
                'base': repela_net_base}[args.model]

    if args.deploy_model:
        model = model_fn(num_classes=args.num_classes, deploy=True).to(device)
        sd = torch.load(args.deploy_model, map_location=device, weights_only=True)
        model.load_state_dict(sd, strict=False)
        print(f'Loaded deploy model: {args.deploy_model}')
    else:
        model = model_fn(num_classes=args.num_classes).to(device)
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'], strict=False)
        print(f'Loaded: {args.checkpoint} (Epoch {ckpt["epoch"]+1}, '
              f'mIoU={ckpt.get("best_miou", "?"):.4f})')

    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'Model: RepELA-Net-{args.model}, {params:.2f}M params')

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
            pred = predict_tta(model, img_tensor, device)
            method = 'tta'
        else:
            pred, method = smart_predict(model, img_tensor, args.crop_size, args.stride, device)
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
    print(f'RepELA-Net-{args.model} | {args.split} set | {len(basenames)} images')
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
            f.write(f'RepELA-Net-{args.model} | {args.split} set | {len(basenames)} images\n')
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
