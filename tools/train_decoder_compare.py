"""
Decoder Comparison Training Script.

Trains RepELA-Small encoder with different decoders to compare
decoder effectiveness under a fixed decoder-comparison recipe.

Usage:
    python tools/train_decoder_compare.py --decoder unet
    python tools/train_decoder_compare.py --decoder fpn
    python tools/train_decoder_compare.py --decoder all
"""

import os
import sys
import argparse
import time
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.mos2_dataset import get_dataloaders, MoS2Dataset
from utils.losses import HybridLoss
from utils.metrics import SegmentationMetrics
from models.repela_net import RepELANet
from models.decoders_compare import DECODER_REGISTRY, DECODER_NAMES, build_decoder


# ─── Model Builder ───────────────────────────────────────────────────

def build_encoder_with_decoder(decoder_name, num_classes=4):
    """Build RepELA-Small encoder + replacement decoder."""
    model = RepELANet(
        num_classes=num_classes,
        channels=(32, 64, 128, 256),
        num_blocks=(2, 2, 4, 2),
        num_heads=(0, 0, 4, 8),
        decoder_channels=128,
        deep_supervision=False,
        use_cse=False,
    )

    if decoder_name != 'ours':
        # Replace decoder with comparison decoder
        in_channels_list = [32, 64, 128, 256]
        new_decoder = build_decoder(decoder_name, in_channels_list, num_classes)
        model.decoder = new_decoder
    # else: keep original DW-MFF + BoundaryEnhancement decoder

    return model


# ─── Training Infrastructure ────────────────────────────────────────

def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger('decoder_compare')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fh = logging.FileHandler(os.path.join(output_dir, 'train.log'))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
    import math
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return max(epoch / max(warmup_epochs, 1), 0.01)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return max(0.5 * (1 + math.cos(math.pi * progress)), min_lr / optimizer.defaults['lr'])
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def validate(model, val_loader, criterion, device):
    model.eval()
    metrics = SegmentationMetrics(num_classes=4)
    total_loss = 0
    count = 0

    with torch.no_grad():
        for images_list, masks_list in val_loader:
            # Val loader uses collate_variable_size → lists
            for img, mask in zip(images_list, masks_list):
                img = img.unsqueeze(0).to(device)
                mask = mask.unsqueeze(0).to(device)
                H, W = mask.shape[1:]

                output = model(img)
                if isinstance(output, tuple):
                    output = output[0]
                output = output[:, :, :H, :W]

                loss, _, _ = criterion(output, mask)
                total_loss += loss.item()
                count += 1

                pred = output.argmax(dim=1)
                metrics.update(pred, mask)

    results = metrics.get_results()
    results['val_loss'] = total_loss / max(count, 1)
    return results


def train_one_epoch(model, loader, criterion, optimizer, device, logger, log_interval=10):
    model.train()
    metrics = SegmentationMetrics(num_classes=4)
    total_loss = 0
    count = 0

    for i, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)
        H, W = masks.shape[1:]

        output = model(images)
        if isinstance(output, tuple):
            output = output[0]
        output = output[:, :, :H, :W]

        loss, _, _ = criterion(output, masks)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        count += 1
        pred = output.detach().argmax(dim=1)
        metrics.update(pred, masks)

        if (i + 1) % log_interval == 0:
            logger.info(f'    [{i+1}/{len(loader)}] loss: {loss.item():.4f}')

    results = metrics.get_results()
    return total_loss / max(count, 1), results


# ─── Main ────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description='Decoder Comparison')
    ALL_DECODERS = list(DECODER_REGISTRY.keys()) + ['ours']
    parser.add_argument('--decoder', type=str, required=True,
                        choices=ALL_DECODERS + ['all'],
                        help='Decoder to use')
    parser.add_argument('--data-root', type=str, default='Mos2_data')
    parser.add_argument('--split-dir', type=str, default='splits')
    parser.add_argument('--output-dir', type=str, default='output/decoder_compare')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=6e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-epochs', type=int, default=10)
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument('--crop-size', type=int, default=512)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--num-classes', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--early-stop-patience', type=int, default=30)
    return parser.parse_args()


def train_single_decoder(decoder_name, args):
    import random as _random, numpy as _np
    _random.seed(args.seed)
    _np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'{decoder_name}_{timestamp}')
    logger = setup_logger(output_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard'))

    train_loader, val_loader = get_dataloaders(
        args.data_root, split_dir=args.split_dir,
        crop_size=args.crop_size, batch_size=args.batch_size,
        num_workers=args.num_workers, copy_paste=False,
        seed=args.seed
    )

    model = build_encoder_with_decoder(decoder_name, args.num_classes)
    model = model.to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    decoder_params = sum(p.numel() for p in model.decoder.parameters()) / 1e6
    dec_display = DECODER_NAMES.get(decoder_name, 'Ours (DW-MFF + Boundary)')
    logger.info(f'Decoder: {dec_display}')
    logger.info(f'Total Params: {params:.2f}M (Decoder: {decoder_params:.2f}M)')

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

        train_loss, train_results = train_one_epoch(
            model, train_loader, criterion, optimizer, device, logger
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
                'best_miou': best_miou, 'decoder': decoder_name,
                'args': vars(args),
            }, os.path.join(output_dir, 'best_model.pth'))
            logger.info(f'  ★ New best mIoU: {best_miou:.4f}')
        else:
            no_improve += 1
            if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
                logger.info(f'  Early stopping at epoch {epoch+1}')
                break

    logger.info('=' * 80)
    logger.info(f'Done. Best mIoU: {best_miou:.4f}')
    writer.close()
    return decoder_name, DECODER_NAMES[decoder_name], f'{params:.2f}M', best_miou


def main():
    args = get_args()

    if args.decoder == 'all':
        decoders = list(DECODER_REGISTRY.keys())
    else:
        decoders = [args.decoder]

    results = []
    for dec in decoders:
        dec_display = DECODER_NAMES.get(dec, 'Ours (DW-MFF + Boundary)')
        print(f'\n{"=" * 80}')
        print(f'  Training: {dec_display}')
        print(f'{"=" * 80}\n')
        result = train_single_decoder(dec, args)
        results.append(result)

    # Summary
    print('\n' + '=' * 80)
    print('  DECODER COMPARISON SUMMARY')
    print('=' * 80)
    print(f'{"Decoder":<30} {"Params":<10} {"Best Val mIoU":<15}')
    print('-' * 55)
    for name, display, params, miou in results:
        print(f'{display:<30} {params:<10} {miou:.4f}')
    print('=' * 80)


if __name__ == '__main__':
    main()
