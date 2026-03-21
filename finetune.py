"""
RepELA-Net Fine-tuning Script for Cross-Material Transfer.

Fine-tunes a pretrained MoS2 model on other 2D material datasets
(graphene, WS2, MoS2_extra). Supports:
  - Different num_classes (3 for graphene, 4 for WS2/MoS2)
  - Encoder weight transfer from MoS2 pretrained checkpoint
  - Sliding-window validation
  - Confusion matrix output

Usage:
    # Fine-tune on graphene (3 classes)
    python finetune.py \
        --data_root "other data/graphene" \
        --pretrained output/repela_small_*/best_model.pth \
        --num_classes 3 --epochs 100 \
        --name graphene

    # Fine-tune on WS2 (4 classes)
    python finetune.py \
        --data_root "other data/WS2_data" \
        --pretrained output/repela_small_*/best_model.pth \
        --num_classes 4 --epochs 100 \
        --name ws2

    # Fine-tune on MoS2_extra (4 classes)
    python finetune.py \
        --data_root "other data/MoS2_data" \
        --pretrained output/repela_small_*/best_model.pth \
        --num_classes 4 --epochs 100 \
        --name mos2_extra
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.repela_net import repela_net_tiny, repela_net_small, repela_net_base
from datasets.material_dataset import (MaterialDataset, collate_variable_size,
                                        compute_dataset_stats, get_auto_crop_size,
                                        IMAGENET_MEAN, IMAGENET_STD)
from utils.losses import HybridLoss
from utils.metrics import SegmentationMetrics


def get_args():
    parser = argparse.ArgumentParser(description='RepELA-Net Fine-tuning')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to dataset (e.g., "other data/graphene")')
    parser.add_argument('--name', type=str, required=True,
                        help='Name for this experiment (e.g., graphene, ws2)')
    parser.add_argument('--pretrained', type=str, required=True,
                        help='Path to MoS2 pretrained checkpoint')
    parser.add_argument('--model', type=str, default='small',
                        choices=['tiny', 'small', 'base'])
    parser.add_argument('--num_classes', type=int, required=True,
                        help='Number of classes (3 for graphene, 4 for WS2/MoS2)')
    parser.add_argument('--train_split', type=str, default='train')
    parser.add_argument('--val_split', type=str, default='val',
                        help='Validation split name (val or test)')

    # Training
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Lower LR for fine-tuning')
    parser.add_argument('--encoder_lr_scale', type=float, default=0.5,
                        help='Encoder LR = lr * encoder_lr_scale')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--freeze_encoder_epochs', type=int, default=20,
                        help='Freeze encoder for N epochs, then unfreeze')
    parser.add_argument('--early_stop_patience', type=int, default=30,
                        help='Stop if val mIoU does not improve for N epochs')
    parser.add_argument('--num_workers', type=int, default=4)

    # Eval
    parser.add_argument('--val_crop_size', type=int, default=512)
    parser.add_argument('--val_stride', type=int, default=384)

    # Output
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger('Finetune')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(output_dir, 'finetune.log'))
    ch = logging.StreamHandler()
    fmt = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger


def load_pretrained(model, checkpoint_path, num_classes_pretrained=4,
                    num_classes_target=4, logger=None):
    """Load pretrained weights, handling class count mismatch."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    pretrained_sd = ckpt['model']

    if num_classes_pretrained != num_classes_target:
        # Remove classification head weights that don't match
        keys_to_remove = []
        for k in list(pretrained_sd.keys()):
            if 'seg_head' in k or 'boundary_head' in k:
                shape = pretrained_sd[k].shape
                if len(shape) > 0 and (shape[0] == num_classes_pretrained or
                        (len(shape) > 1 and shape[1] == num_classes_pretrained)):
                    keys_to_remove.append(k)

        for k in keys_to_remove:
            del pretrained_sd[k]
            if logger:
                logger.info(f'  Skipped (class mismatch): {k}')

    # Load with strict=False so mismatched keys are ignored
    missing, unexpected = model.load_state_dict(pretrained_sd, strict=False)
    if logger:
        logger.info(f'  Loaded pretrained weights from: {checkpoint_path}')
        logger.info(f'  Pretrained epoch: {ckpt.get("epoch", "?")+1}, '
                    f'mIoU: {ckpt.get("best_miou", "?"):.4f}')
        if missing:
            logger.info(f'  Missing keys (will be randomly init): {len(missing)}')
            for k in missing:
                logger.info(f'    {k}')
        if unexpected:
            logger.info(f'  Unexpected keys (ignored): {len(unexpected)}')

    return model


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr):
    import math
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return max(min_lr / optimizer.defaults['lr'],
                   0.5 * (1.0 + math.cos(math.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def sliding_window_predict(model, img_tensor, crop_size, stride, device):
    _, H, W = img_tensor.shape
    num_classes = model.num_classes

    pred_sum = torch.zeros(num_classes, H, W, dtype=torch.float32, device=device)
    count = torch.zeros(H, W, dtype=torch.float32, device=device)

    pad_h = max(0, crop_size - H)
    pad_w = max(0, crop_size - W)
    if pad_h > 0 or pad_w > 0:
        img_tensor = F.pad(img_tensor, [0, pad_w, 0, pad_h], mode='reflect')
    _, pH, pW = img_tensor.shape

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


def plot_confusion_matrix(confusion, class_labels, save_path):
    row_sums = np.maximum(confusion.sum(axis=1, keepdims=True), 1)
    cm_norm = confusion / row_sums * 100

    n = len(class_labels)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues,
                   vmin=0, vmax=100)
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
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close()


def train_one_epoch(model, loader, criterion, optimizer, device, logger):
    model.train()
    metrics = SegmentationMetrics(model.num_classes)
    total_loss = 0
    n = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss, _, _ = criterion(logits, masks)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        preds = logits.argmax(dim=1)
        metrics.update(preds, masks)
        total_loss += loss.item()
        n += 1

    return total_loss / n, metrics.get_results()


@torch.no_grad()
def validate(model, val_dataset, device, args, logger, mean=None, std=None):
    model.eval()
    metrics = SegmentationMetrics(model.num_classes)
    import torchvision.transforms.functional as TF
    norm_mean = mean or IMAGENET_MEAN
    norm_std = std or IMAGENET_STD

    for idx in range(len(val_dataset)):
        img_path, mask_path = val_dataset.pairs[idx]
        img = Image.open(img_path).convert('RGB')
        gt = np.array(Image.open(mask_path))
        img_tensor = TF.normalize(TF.to_tensor(img), norm_mean, norm_std)

        pred = sliding_window_predict(model, img_tensor,
                                      args.val_crop_size, args.val_stride, device)
        metrics.update(pred, gt)

    return metrics


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    output_dir = os.path.join(args.output_dir, f'finetune_{args.name}')
    logger = setup_logger(output_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Fine-tuning: {args.name} ({args.num_classes} classes)')
    logger.info(f'Device: {device}')
    logger.info(f'Args: {args}')

    # Auto-compute per-dataset normalization stats
    logger.info('Computing dataset-specific normalization stats...')
    stats = compute_dataset_stats(args.data_root, split=args.train_split)
    ds_mean = stats['mean']
    ds_std = stats['std']
    logger.info(f'  Dataset mean: [{ds_mean[0]:.4f}, {ds_mean[1]:.4f}, {ds_mean[2]:.4f}]')
    logger.info(f'  Dataset std:  [{ds_std[0]:.4f}, {ds_std[1]:.4f}, {ds_std[2]:.4f}]')

    # Auto-adjust crop size for small images
    auto_crop = get_auto_crop_size(args.data_root, split=args.train_split,
                                    max_crop=args.crop_size)
    if auto_crop != args.crop_size:
        logger.info(f'  Auto crop_size: {args.crop_size} -> {auto_crop} '
                    f'(images smaller than {args.crop_size})')
        args.crop_size = auto_crop
        args.val_crop_size = auto_crop
        args.val_stride = int(auto_crop * 0.75)

    # Datasets with domain-adaptive normalization
    train_dataset = MaterialDataset(args.data_root, split=args.train_split,
                                     crop_size=args.crop_size, augment=True,
                                     mean=ds_mean, std=ds_std)
    val_dataset = MaterialDataset(args.data_root, split=args.val_split,
                                   crop_size=args.crop_size, augment=False,
                                   mean=ds_mean, std=ds_std)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)

    # Model
    model_fn = {'tiny': repela_net_tiny, 'small': repela_net_small,
                'base': repela_net_base}[args.model]
    model = model_fn(num_classes=args.num_classes).to(device)

    # Load pretrained
    logger.info('Loading pretrained MoS2 weights...')
    model = load_pretrained(model, args.pretrained,
                            num_classes_pretrained=4,
                            num_classes_target=args.num_classes,
                            logger=logger)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Model: RepELA-Net-{args.model}, {total_params/1e6:.2f}M params')

    # Compute class weights from training data
    logger.info('Computing class weights from training data...')
    class_counts = np.zeros(args.num_classes)
    for _, mask_path in train_dataset.pairs:
        mask = np.array(Image.open(mask_path))
        for c in range(args.num_classes):
            class_counts[c] += (mask == c).sum()
    total_px = class_counts.sum()
    class_freqs = class_counts / total_px
    # Inverse frequency weights, capped
    class_weights = np.clip(1.0 / (class_freqs * args.num_classes + 1e-6), 0.1, 10.0)
    logger.info(f'Class distribution: {[f"{f:.3f}" for f in class_freqs]}')
    logger.info(f'Class weights: {[f"{w:.2f}" for w in class_weights]}')

    criterion = HybridLoss(
        num_classes=args.num_classes,
        focal_alpha=class_weights.tolist(),
        focal_gamma=2.0,
    ).to(device)

    # Differential LR: encoder slower, decoder faster
    encoder_params = []
    decoder_params = []
    for name, p in model.named_parameters():
        if 'decoder' in name or 'seg_head' in name or 'boundary_head' in name:
            decoder_params.append(p)
        else:
            encoder_params.append(p)

    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': args.lr * args.encoder_lr_scale},
        {'params': decoder_params, 'lr': args.lr},
    ], weight_decay=args.weight_decay)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup_epochs, args.epochs, min_lr=1e-6
    )

    logger.info(f'Encoder LR: {args.lr * args.encoder_lr_scale:.6f}, '
                f'Decoder LR: {args.lr:.6f}')
    logger.info(f'Train: {len(train_dataset)}, Val: {len(val_dataset)}')
    logger.info('=' * 60)

    best_miou = 0.0
    class_names = {
        3: ['BG', '1L', '2L'],  # graphene
        4: ['BG', '1L', '2L', 'ML'],  # WS2/MoS2
    }.get(args.num_classes, [f'C{i}' for i in range(args.num_classes)])

    # Freeze encoder initially
    encoder_frozen = False
    if args.freeze_encoder_epochs > 0:
        for name, p in model.named_parameters():
            if 'decoder' not in name:
                p.requires_grad = False
        encoder_frozen = True
        logger.info(f'Encoder FROZEN for first {args.freeze_encoder_epochs} epochs')

    no_improve_count = 0
    logger.info(f'Early stopping patience: {args.early_stop_patience} epochs')

    for epoch in range(args.epochs):
        # Unfreeze encoder after N epochs
        if encoder_frozen and epoch >= args.freeze_encoder_epochs:
            for p in model.parameters():
                p.requires_grad = True
            encoder_frozen = False
            logger.info(f'  >>> Encoder UNFROZEN at epoch {epoch+1}')

        t0 = time.time()
        lr = optimizer.param_groups[1]['lr']
        logger.info(f'Epoch [{epoch+1}/{args.epochs}] LR(dec): {lr:.6f}'
                    f'{" [encoder frozen]" if encoder_frozen else ""}')

        train_loss, train_results = train_one_epoch(
            model, train_loader, criterion, optimizer, device, logger
        )

        val_metrics = validate(model, val_dataset, device, args, logger,
                               mean=ds_mean, std=ds_std)
        val_results = val_metrics.get_results()
        scheduler.step()

        elapsed = time.time() - t0

        iou_str = ' | '.join([
            f'{class_names[i]}: {val_results["per_class_iou"][i]:.4f}'
            for i in range(args.num_classes)
        ])
        logger.info(
            f'  Train Loss: {train_loss:.4f} mIoU: {train_results["mIoU"]:.4f}')
        logger.info(
            f'  Val mIoU: {val_results["mIoU"]:.4f} Acc: {val_results["pixel_acc"]:.4f} '
            f'F1: {val_results["mean_f1"]:.4f}')
        logger.info(f'  Per-class IoU: {iou_str}')
        logger.info(f'  Time: {elapsed:.1f}s')

        if val_results['mIoU'] > best_miou:
            best_miou = val_results['mIoU']
            no_improve_count = 0
            torch.save({
                'epoch': epoch, 'model': model.state_dict(),
                'best_miou': best_miou, 'args': vars(args),
            }, os.path.join(output_dir, 'best_model.pth'))
            logger.info(f'  ★ New best mIoU: {best_miou:.4f}')

            # Save confusion matrix at best epoch
            plot_confusion_matrix(
                val_metrics.confusion_matrix, class_names,
                os.path.join(output_dir, 'confusion_matrix.png')
            )
        else:
            no_improve_count += 1
            if no_improve_count >= args.early_stop_patience:
                logger.info(f'  Early stopping: no improvement for '
                            f'{args.early_stop_patience} epochs')
                break

        logger.info('-' * 60)

    logger.info('=' * 60)
    logger.info(f'Fine-tuning complete. Best val mIoU: {best_miou:.4f}')

    # Final evaluation with best model
    logger.info('\nFinal evaluation with best checkpoint:')
    ckpt = torch.load(os.path.join(output_dir, 'best_model.pth'),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    final_metrics = validate(model, val_dataset, device, args, logger,
                             mean=ds_mean, std=ds_std)
    final_results = final_metrics.get_results()

    logger.info(f'  mIoU:      {final_results["mIoU"]:.4f}')
    logger.info(f'  Pixel Acc: {final_results["pixel_acc"]:.4f}')
    logger.info(f'  Mean F1:   {final_results["mean_f1"]:.4f}')
    for i in range(args.num_classes):
        logger.info(f'  {class_names[i]:5s}  IoU={final_results["per_class_iou"][i]:.4f}  '
                    f'F1={final_results["f1"][i]:.4f}')

    # Save final confusion matrix
    plot_confusion_matrix(
        final_metrics.confusion_matrix, class_names,
        os.path.join(output_dir, 'confusion_matrix_final.png')
    )
    logger.info(f'Confusion matrix saved: {output_dir}/confusion_matrix_final.png')

    # Save metrics to file
    with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
        f.write(f'Dataset: {args.name}\n')
        f.write(f'Pretrained: {args.pretrained}\n')
        f.write(f'Best epoch: {ckpt["epoch"]+1}\n')
        f.write(f'mIoU: {final_results["mIoU"]:.4f}\n')
        f.write(f'Pixel Acc: {final_results["pixel_acc"]:.4f}\n')
        f.write(f'Mean F1: {final_results["mean_f1"]:.4f}\n\n')
        for i in range(args.num_classes):
            f.write(f'{class_names[i]:5s}  IoU={final_results["per_class_iou"][i]:.4f}  '
                    f'F1={final_results["f1"][i]:.4f}\n')


if __name__ == '__main__':
    main()
