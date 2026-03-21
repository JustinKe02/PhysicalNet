"""
RepELA-Net Training Script (v2).

Changes from v1:
  - Uses fixed split files (splits/train.txt, val.txt)
  - Deterministic sliding-window validation (not random crop)
  - Proper full-image mIoU reporting

Usage:
    python train.py --data_root ./Mos2_data --model small --epochs 200
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from models.repela_net import repela_net_tiny, repela_net_small, repela_net_base
from datasets.mos2_dataset import get_dataloaders, MoS2Dataset
from utils.losses import HybridLoss
from utils.metrics import SegmentationMetrics


def get_args():
    parser = argparse.ArgumentParser(description='RepELA-Net Training')

    # Data
    parser.add_argument('--data_root', type=str, default='./Mos2_data')
    parser.add_argument('--split_dir', type=str, default='splits/')
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)

    # Model
    parser.add_argument('--model', type=str, default='small',
                        choices=['tiny', 'small', 'base'])
    parser.add_argument('--num_classes', type=int, default=4)

    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=6e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_epochs', type=int, default=10)

    # Loss
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--loss_weights', type=float, nargs='+', default=[1.0, 1.0])
    parser.add_argument('--boundary_weight', type=float, default=0.5,
                        help='Weight for boundary supervision loss')

    # Output
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--save_freq', type=int, default=10)

    # Eval
    parser.add_argument('--val_crop_size', type=int, default=512,
                        help='Crop size for sliding window validation')
    parser.add_argument('--val_stride', type=int, default=384,
                        help='Stride for sliding window validation')

    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=False,
                        help='Mixed precision training (--amp / --no-amp)')
    parser.add_argument('--deep_supervision', action='store_true', default=True,
                        help='Enable deep supervision auxiliary losses')
    parser.add_argument('--aux_loss_weight', type=float, default=0.4,
                        help='Weight for each auxiliary loss head')
    parser.add_argument('--early_stop_patience', type=int, default=20,
                        help='Stop if val mIoU does not improve for N epochs (0=disabled)')
    parser.add_argument('--resume', type=str, default=None)

    return parser.parse_args()


def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'train.log')
    logger = logging.getLogger('RepELANet')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr):
    import math
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return max(min_lr / optimizer.defaults['lr'],
                   0.5 * (1.0 + math.cos(math.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def sliding_window_predict(model, img_tensor, crop_size=512, stride=384, device='cuda'):
    """Deterministic sliding window inference on a single full-res image.

    Returns: prediction [H, W] class indices (numpy).
    """
    _, H, W = img_tensor.shape
    num_classes = 4

    pred_sum = torch.zeros(num_classes, H, W, dtype=torch.float32, device=device)
    count = torch.zeros(H, W, dtype=torch.float32, device=device)

    # Pad if needed
    pad_h = max(0, crop_size - H)
    pad_w = max(0, crop_size - W)
    if pad_h > 0 or pad_w > 0:
        img_tensor = F.pad(img_tensor, [0, pad_w, 0, pad_h], mode='reflect')

    _, pH, pW = img_tensor.shape

    # Window positions (guaranteed full coverage)
    y_positions = list(range(0, max(1, pH - crop_size + 1), stride))
    if len(y_positions) == 0 or y_positions[-1] + crop_size < pH:
        y_positions.append(max(0, pH - crop_size))
    x_positions = list(range(0, max(1, pW - crop_size + 1), stride))
    if len(x_positions) == 0 or x_positions[-1] + crop_size < pW:
        x_positions.append(max(0, pW - crop_size))
    y_positions = sorted(set(y_positions))
    x_positions = sorted(set(x_positions))

    for y in y_positions:
        for x in x_positions:
            crop = img_tensor[:, y:y+crop_size, x:x+crop_size].unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(crop)
                probs = F.softmax(logits, dim=1)[0]

            y_end = min(y + crop_size, H)
            x_end = min(x + crop_size, W)
            pred_sum[:, y:y_end, x:x_end] += probs[:, :y_end-y, :x_end-x]
            count[y:y_end, x:x_end] += 1

    count = count.clamp(min=1)
    pred_avg = pred_sum / count.unsqueeze(0)
    return pred_avg.argmax(dim=0).cpu().numpy()


def train_one_epoch(model, train_loader, criterion, optimizer, scaler,
                    device, epoch, args, logger):
    model.train()
    metrics = SegmentationMetrics(args.num_classes)
    total_loss = total_focal = total_dice = 0
    num_batches = 0

    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad()

        if args.amp:
            with autocast('cuda'):
                output = model(images)
                if isinstance(output, tuple):
                    logits, aux_list = output
                else:
                    logits, aux_list = output, []
                loss, focal_loss, dice_loss = criterion(logits, masks)
                for aux in aux_list:
                    aux_loss, _, _ = criterion(aux, masks)
                    loss = loss + args.aux_loss_weight * aux_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(images)
            if isinstance(output, tuple):
                logits, aux_list = output
            else:
                logits, aux_list = output, []
            loss, focal_loss, dice_loss = criterion(logits, masks)
            for aux in aux_list:
                aux_loss, _, _ = criterion(aux, masks)
                loss = loss + args.aux_loss_weight * aux_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        preds = logits.argmax(dim=1)
        metrics.update(preds, masks)
        total_loss += loss.item()
        total_focal += focal_loss.item()
        total_dice += dice_loss.item()
        num_batches += 1

        if (batch_idx + 1) % 10 == 0:
            logger.info(
                f'  Batch [{batch_idx+1}/{len(train_loader)}] '
                f'Loss: {loss.item():.4f} '
                f'(Focal: {focal_loss.item():.4f}, Dice: {dice_loss.item():.4f})'
            )

    return (total_loss / num_batches, total_focal / num_batches,
            total_dice / num_batches, metrics.get_results())


@torch.no_grad()
def predict_full_or_sliding(model, img_tensor, crop_size, stride, device):
    """Try full-image inference, fallback to sliding window on OOM."""
    try:
        img = img_tensor.unsqueeze(0).to(device)
        logits = model(img)
        pred = logits.argmax(dim=1)[0].cpu().numpy()
        return pred
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return sliding_window_predict(model, img_tensor, crop_size, stride, device)


@torch.no_grad()
def validate(model, val_loader, criterion, device, args, logger):
    """Full-image validation with loss computation and OOM fallback."""
    model.eval()
    metrics = SegmentationMetrics(args.num_classes)
    total_loss = total_ce = total_dice = 0
    num_images = 0

    for images_list, masks_list in val_loader:
        for img_tensor, mask_tensor in zip(images_list, masks_list):
            mask_np = mask_tensor.numpy() if isinstance(mask_tensor, torch.Tensor) else mask_tensor

            # Try full-image forward for both prediction and loss
            try:
                img = img_tensor.unsqueeze(0).to(device)
                mask_dev = torch.from_numpy(mask_np).unsqueeze(0).long().to(device)
                logits = model(img)
                loss, ce_loss, dice_loss = criterion(logits, mask_dev)
                total_loss += loss.item()
                total_ce += ce_loss.item()
                total_dice += dice_loss.item()
                prediction = logits.argmax(dim=1)[0].cpu().numpy()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                prediction = sliding_window_predict(
                    model, img_tensor,
                    crop_size=args.val_crop_size,
                    stride=args.val_stride,
                    device=device
                )
                # No loss for sliding window fallback

            metrics.update(
                torch.from_numpy(prediction).unsqueeze(0),
                torch.from_numpy(mask_np).unsqueeze(0)
            )
            num_images += 1

    results = metrics.get_results()
    if num_images > 0:
        results['val_loss'] = total_loss / num_images
        results['val_ce'] = total_ce / num_images
        results['val_dice'] = total_dice / num_images
    return results


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'repela_{args.model}_{timestamp}')
    logger = setup_logger(output_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')
    logger.info(f'Args: {args}')

    # TensorBoard
    tb_dir = os.path.join(output_dir, 'tensorboard')
    writer = SummaryWriter(log_dir=tb_dir)
    logger.info(f'TensorBoard: {tb_dir}')

    # Data
    train_loader, val_loader = get_dataloaders(
        args.data_root, split_dir=args.split_dir,
        crop_size=args.crop_size, batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Model
    model_fn = {'tiny': repela_net_tiny, 'small': repela_net_small,
                'base': repela_net_base}[args.model]
    model = model_fn(num_classes=args.num_classes,
                     deep_supervision=args.deep_supervision).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Model: RepELA-Net-{args.model}, Params: {total_params:,} ({total_params/1e6:.2f}M)')
    logger.info(f'Deep supervision: {args.deep_supervision}')

    # Loss
    criterion = HybridLoss(
        num_classes=args.num_classes,
        focal_alpha=MoS2Dataset.CLASS_WEIGHTS,
        focal_gamma=args.focal_gamma,
        loss_weights=tuple(args.loss_weights),
        boundary_weight=args.boundary_weight
    ).to(device)

    # Optimizer & scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup_epochs, args.epochs, args.min_lr
    )
    scaler = GradScaler('cuda', enabled=args.amp)

    # Resume
    start_epoch = 0
    best_miou = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_miou = ckpt.get('best_miou', 0.0)
        logger.info(f'Resumed from epoch {start_epoch}, best mIoU: {best_miou:.4f}')

    class_names = MoS2Dataset.CLASSES
    logger.info(f'Validation: sliding window (crop={args.val_crop_size}, stride={args.val_stride})')
    if args.early_stop_patience > 0:
        logger.info(f'Early stopping patience: {args.early_stop_patience} epochs')
    logger.info(f'Starting training for {args.epochs} epochs...')
    logger.info('=' * 80)

    no_improve_count = 0

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        lr = optimizer.param_groups[0]['lr']
        logger.info(f'Epoch [{epoch+1}/{args.epochs}] LR: {lr:.6f}')

        # Train (random crop)
        train_loss, train_focal, train_dice, train_results = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, args, logger
        )

        # Validate (sliding window on full images)
        val_results = validate(
            model, val_loader, criterion, device, args, logger
        )

        scheduler.step()
        epoch_time = time.time() - epoch_start

        logger.info(
            f'  Train Loss: {train_loss:.4f} '
            f'(CE: {train_focal:.4f}, Dice: {train_dice:.4f}) '
            f'mIoU: {train_results["mIoU"]:.4f} Acc: {train_results["pixel_acc"]:.4f}'
        )
        val_loss = val_results.get('val_loss', 0)
        val_ce = val_results.get('val_ce', 0)
        val_dice = val_results.get('val_dice', 0)
        logger.info(
            f'  Val Loss: {val_loss:.4f} '
            f'(CE: {val_ce:.4f}, Dice: {val_dice:.4f}) '
            f'mIoU: {val_results["mIoU"]:.4f} '
            f'Acc: {val_results["pixel_acc"]:.4f} '
            f'F1: {val_results["mean_f1"]:.4f}'
        )
        iou_str = ' | '.join([
            f'{class_names[i]}: {val_results["per_class_iou"][i]:.4f}'
            for i in range(args.num_classes)
        ])
        logger.info(f'  Per-class IoU: {iou_str}')
        logger.info(f'  Time: {epoch_time:.1f}s')
        logger.info('-' * 80)

        # TensorBoard logging
        writer.add_scalars('Loss/Total', {
            'train': train_loss,
            'val': val_loss,
        }, epoch + 1)
        writer.add_scalars('Loss/CE', {
            'train': train_focal,
            'val': val_ce,
        }, epoch + 1)
        writer.add_scalars('Loss/Dice', {
            'train': train_dice,
            'val': val_dice,
        }, epoch + 1)
        writer.add_scalars('mIoU', {
            'train': train_results['mIoU'],
            'val': val_results['mIoU'],
        }, epoch + 1)
        writer.add_scalars('Accuracy', {
            'train': train_results['pixel_acc'],
            'val': val_results['pixel_acc'],
        }, epoch + 1)
        writer.add_scalar('LR', lr, epoch + 1)
        for i, name in enumerate(class_names):
            writer.add_scalar(f'Val_IoU/{name}', val_results['per_class_iou'][i], epoch + 1)
        writer.flush()

        # Save best
        if val_results['mIoU'] > best_miou:
            best_miou = val_results['mIoU']
            no_improve_count = 0
            torch.save({
                'epoch': epoch, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_miou': best_miou, 'args': vars(args),
            }, os.path.join(output_dir, 'best_model.pth'))
            logger.info(f'  ★ New best mIoU: {best_miou:.4f}')
        else:
            no_improve_count += 1
            if args.early_stop_patience > 0 and no_improve_count >= args.early_stop_patience:
                logger.info(f'  Early stopping: no improvement for '
                            f'{args.early_stop_patience} epochs')
                break

        if (epoch + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_miou': best_miou, 'args': vars(args),
            }, os.path.join(output_dir, f'checkpoint_epoch{epoch+1}.pth'))

    logger.info('=' * 80)
    logger.info(f'Training complete. Best val mIoU (sliding-window): {best_miou:.4f}')

    model.switch_to_deploy()
    torch.save(model.state_dict(), os.path.join(output_dir, 'deploy_model.pth'))
    logger.info('Deploy model saved.')
    writer.close()


if __name__ == '__main__':
    main()
