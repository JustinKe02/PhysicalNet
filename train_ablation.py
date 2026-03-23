"""
Ablation Study Training Script for RepELA-Net.

Creates model variants with specific components disabled to measure
each module's contribution to overall performance.

Ablation variants:
  with_cse     + ColorSpaceEnhancement (use_cse=True)
  no_ela       Replace ELA stages with RepConv stages (no attention)
  no_rep       Replace RepConvBN with standard Conv+BN (no reparameterization)
  no_boundary  Remove BoundaryEnhancement from decoder
  no_dwmff     Replace DynamicWeightedFusion with simple addition

Usage:
    python train_ablation.py --ablation no_ela
    python train_ablation.py --ablation all  # Train all variants sequentially
"""

import os
import sys
import copy
import time
import argparse
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from datasets.mos2_dataset import get_dataloaders, MoS2Dataset
from utils.losses import HybridLoss
from utils.metrics import SegmentationMetrics

# Import model components
from models.repela_net import RepELANet, ColorSpaceEnhancement
from models.rep_conv import RepConvStage, RepConvBlock
from models.ela_block import ELAStage
from models.decoder import DWMFFDecoder, DynamicWeightedFusion, BoundaryEnhancement


# ─── Ablation Model Builders ─────────────────────────────────────────

def build_ablation_model(ablation, num_classes=4, deep_supervision=True):
    """Build a RepELA-Net-Small variant with one component disabled."""

    if ablation == 'with_cse':
        # RepELA-Net-Small with CSE enabled
        from models.repela_net import repela_net_small
        return repela_net_small(num_classes=num_classes, deep_supervision=deep_supervision,
                                use_cse=True)

    elif ablation == 'no_ela':
        # Replace ELA stages (3,4) with RepConv stages (no attention)
        model = RepELANet(
            num_classes=num_classes,
            channels=(32, 64, 128, 256),
            num_blocks=(2, 2, 4, 2),
            num_heads=(0, 0, 4, 8),
            decoder_channels=128,
            deep_supervision=deep_supervision,
        )
        # Replace ELA stages with RepConv stages
        model.stage3 = RepConvStage(64, 128, num_blocks=4, expand_ratio=2, use_se=True)
        model.stage4 = RepConvStage(128, 256, num_blocks=2, expand_ratio=2, use_se=True)
        return model

    elif ablation == 'no_rep':
        # Replace RepConvBN multi-branch with standard single Conv+BN
        model = RepELANet(
            num_classes=num_classes,
            channels=(32, 64, 128, 256),
            num_blocks=(2, 2, 4, 2),
            num_heads=(0, 0, 4, 8),
            decoder_channels=128,
            deep_supervision=deep_supervision,
        )
        # Walk through all RepConvBN modules and replace multi-branch with single Conv+BN
        from models.rep_conv import RepConvBN
        for name, module in model.named_modules():
            if isinstance(module, RepConvBN) and not module.deploy:
                # Create a standard Conv+BN replacement (no multi-branch)
                std_conv = nn.Sequential(
                    nn.Conv2d(module.in_channels, module.out_channels,
                              module.kernel_size, stride=module.stride,
                              padding=module.padding, groups=module.groups, bias=False),
                    nn.BatchNorm2d(module.out_channels)
                )
                # Set this replacement on the parent module
                parts = name.split('.')
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
                setattr(parent, parts[-1], std_conv)
        return model

    elif ablation == 'no_boundary':
        # Replace BoundaryEnhancement with equal-param Conv block
        model = RepELANet(
            num_classes=num_classes,
            channels=(32, 64, 128, 256),
            num_blocks=(2, 2, 4, 2),
            num_heads=(0, 0, 4, 8),
            decoder_channels=128,
            deep_supervision=deep_supervision,
        )
        dec_ch = 128
        # Replace with a simple Conv+BN+GELU block (similar param count)
        model.decoder.boundary = nn.Sequential(
            nn.Conv2d(dec_ch, dec_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(dec_ch),
            nn.GELU()
        )
        return model

    elif ablation == 'no_dwmff':
        # Replace DynamicWeightedFusion with simple addition
        model = RepELANet(
            num_classes=num_classes,
            channels=(32, 64, 128, 256),
            num_blocks=(2, 2, 4, 2),
            num_heads=(0, 0, 4, 8),
            decoder_channels=128,
            deep_supervision=deep_supervision,
        )

        # Replace fusion modules with simple addition
        class SimpleAdd(nn.Module):
            def __init__(self, channels, num_inputs=2):
                super().__init__()
            def forward(self, features):
                return sum(features)

        dec_ch = 128
        model.decoder.fuse_43 = SimpleAdd(dec_ch)
        model.decoder.fuse_32 = SimpleAdd(dec_ch)
        model.decoder.fuse_21 = SimpleAdd(dec_ch)
        return model

    else:
        raise ValueError(f'Unknown ablation: {ablation}')


ALL_ABLATIONS = ['with_cse', 'no_ela', 'no_rep', 'no_boundary', 'no_dwmff']

ABLATION_NAMES = {
    'with_cse': '+ ColorSpaceEnhancement',
    'no_ela': 'w/o ELA (Linear Attention)',
    'no_rep': 'w/o RepConv (Reparameterization)',
    'no_boundary': 'w/o BoundaryEnhancement',
    'no_dwmff': 'w/o DW-MFF (Dynamic Fusion)',
}


# ─── Training Infrastructure ─────────────────────────────────────────

def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger('ablation')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(os.path.join(output_dir, 'train.log'))
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
    import math
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return max(min_lr / optimizer.defaults['lr'],
                   0.5 * (1 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()
    metrics = SegmentationMetrics(4)
    total_loss = total_ce = total_dice = 0
    num_images = 0
    loss_images = 0  # Only count images where loss was computed

    for images_list, masks_list in val_loader:
        for img_tensor, mask_tensor in zip(images_list, masks_list):
            mask_np = mask_tensor.numpy() if isinstance(mask_tensor, torch.Tensor) else mask_tensor
            try:
                _, H, W = img_tensor.shape
                pad_h = (32 - H % 32) % 32
                pad_w = (32 - W % 32) % 32
                img_padded = F.pad(img_tensor, [0, pad_w, 0, pad_h], mode='reflect')
                img = img_padded.unsqueeze(0).to(device)
                logits = model(img)
                if isinstance(logits, tuple):
                    logits = logits[0]
                logits = logits[:, :, :H, :W]
                mask_dev = torch.from_numpy(mask_np).unsqueeze(0).long().to(device)
                loss, ce_loss, dice_loss = criterion(logits, mask_dev)
                total_loss += loss.item()
                total_ce += ce_loss.item()
                total_dice += dice_loss.item()
                loss_images += 1
                prediction = logits.argmax(dim=1)[0].cpu().numpy()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                # Fallback to sliding window instead of zeros
                from train import sliding_window_predict
                prediction = sliding_window_predict(
                    model, img_tensor, crop_size=512, stride=384, device=device
                )

            metrics.update(
                torch.from_numpy(prediction).unsqueeze(0),
                torch.from_numpy(mask_np).unsqueeze(0)
            )
            num_images += 1

    results = metrics.get_results()
    if loss_images > 0:
        results['val_loss'] = total_loss / loss_images
        results['val_ce'] = total_ce / loss_images
        results['val_dice'] = total_dice / loss_images
    return results


def train_one_epoch(model, loader, criterion, optimizer, device, logger,
                    deep_supervision=True, aux_weight=0.4, log_interval=10):
    model.train()
    metrics = SegmentationMetrics(4)
    total_loss = total_focal = total_dice = 0
    num_batches = 0

    for i, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)

        output = model(images)
        if isinstance(output, tuple) and deep_supervision:
            logits, aux_list = output
            loss, focal, dice = criterion(logits, masks)
            for aux in aux_list:
                aux_up = F.interpolate(aux, size=masks.shape[1:], mode='bilinear', align_corners=False)
                aux_loss, _, _ = criterion(aux_up, masks)
                loss = loss + aux_weight * aux_loss
        else:
            logits = output[0] if isinstance(output, tuple) else output
            loss, focal, dice = criterion(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_focal += focal.item()
        total_dice += dice.item()
        num_batches += 1

        pred = logits.detach().argmax(dim=1)
        metrics.update(pred, masks)

        if (i + 1) % log_interval == 0:
            logger.info(f'  Batch [{i+1}/{len(loader)}] Loss: {loss.item():.4f}')

    return (total_loss / num_batches, total_focal / num_batches,
            total_dice / num_batches, metrics.get_results())


# ─── Main ────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description='Ablation Study Training')
    parser.add_argument('--data_root', type=str, default='./Mos2_data')
    parser.add_argument('--split_dir', type=str, default='splits/')
    parser.add_argument('--ablation', type=str, default='with_cse',
                        help=f'Ablation variant or "all". Options: {", ".join(ALL_ABLATIONS)}')
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=6e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--early_stop_patience', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='./output/ablation')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def train_single_ablation(ablation, args):
    import random as _random, numpy as _np
    _random.seed(args.seed)
    _np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'{ablation}_{timestamp}')
    logger = setup_logger(output_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard'))

    train_loader, val_loader = get_dataloaders(
        args.data_root, split_dir=args.split_dir,
        crop_size=args.crop_size, batch_size=args.batch_size,
        num_workers=args.num_workers, copy_paste=True  # Enable CopyPaste for V3
    )

    model = build_ablation_model(ablation, args.num_classes, deep_supervision=True)
    model = model.to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f'Ablation: {ABLATION_NAMES[ablation]}')
    logger.info(f'Params: {params:.2f}M')

    criterion = HybridLoss(
        num_classes=args.num_classes,
        focal_alpha=MoS2Dataset.CLASS_WEIGHTS,
        focal_gamma=2.0,
        loss_weights=(1.0, 1.0)
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup_epochs, args.epochs, args.min_lr
    )

    best_miou = 0.0
    no_improve = 0
    class_names = MoS2Dataset.CLASSES

    logger.info(f'Training for {args.epochs} epochs (patience={args.early_stop_patience})')
    logger.info('=' * 80)

    for epoch in range(args.epochs):
        lr = optimizer.param_groups[0]['lr']
        logger.info(f'Epoch [{epoch+1}/{args.epochs}] LR: {lr:.6f}')

        train_loss, train_focal, train_dice, train_results = train_one_epoch(
            model, train_loader, criterion, optimizer, device, logger,
            deep_supervision=True
        )
        val_results = validate(model, val_loader, criterion, device)
        scheduler.step()

        val_loss = val_results.get('val_loss', 0)
        logger.info(
            f'  Train Loss: {train_loss:.4f} mIoU: {train_results["mIoU"]:.4f}'
        )
        logger.info(
            f'  Val Loss: {val_loss:.4f} mIoU: {val_results["mIoU"]:.4f} '
            f'Acc: {val_results["pixel_acc"]:.4f} F1: {val_results["mean_f1"]:.4f}'
        )
        iou_str = ' | '.join([
            f'{class_names[i]}: {val_results["per_class_iou"][i]:.4f}'
            for i in range(args.num_classes)
        ])
        logger.info(f'  Per-class IoU: {iou_str}')
        logger.info('-' * 80)

        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch + 1)
        writer.add_scalars('mIoU', {
            'train': train_results['mIoU'], 'val': val_results['mIoU']
        }, epoch + 1)
        writer.flush()

        if val_results['mIoU'] > best_miou:
            best_miou = val_results['mIoU']
            no_improve = 0
            torch.save({
                'epoch': epoch, 'model': model.state_dict(),
                'best_miou': best_miou, 'ablation': ablation,
            }, os.path.join(output_dir, 'best_model.pth'))
            logger.info(f'  ★ New best mIoU: {best_miou:.4f}')
        else:
            no_improve += 1
            if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
                logger.info(f'  Early stopping at epoch {epoch+1}')
                break

    logger.info('=' * 80)
    logger.info(f'{ABLATION_NAMES[ablation]} training complete. Best val mIoU: {best_miou:.4f}')
    writer.close()
    return ablation, ABLATION_NAMES[ablation], f'{params:.2f}M', best_miou


def main():
    args = get_args()

    if args.ablation == 'all':
        results = []
        for abl in ALL_ABLATIONS:
            print(f'\n{"="*80}')
            print(f'Ablation: {ABLATION_NAMES[abl]}')
            print(f'{"="*80}\n')
            key, name, params, miou = train_single_ablation(abl, args)
            results.append((key, name, params, miou))

        print('\n' + '=' * 70)
        print('ABLATION STUDY SUMMARY')
        print('=' * 70)
        print(f'{"Variant":<35} {"Params":<10} {"Val mIoU":<10} {"Δ mIoU":<10}')
        print('-' * 65)
        base_miou = results[0][3]
        for key, name, params, miou in results:
            delta = miou - base_miou
            sign = '+' if delta >= 0 else ''
            print(f'{name:<35} {params:<10} {miou:.4f}     {sign}{delta:.4f}')
        print('=' * 70)
    else:
        train_single_ablation(args.ablation, args)


if __name__ == '__main__':
    main()
