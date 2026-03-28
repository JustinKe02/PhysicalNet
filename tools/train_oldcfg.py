"""
Old-config Baseline Training for RepELA-Net-Small.

Purpose:
    Reproduce the oldcfg baseline (Val mIoU ≈ 0.8333) by replicating the exact
    training loop used in train_ablation.py, fixed for the RepELA-Small w/o CSE
    baseline.

Why a separate file:
    1. The old 0.8333 was produced by a simpler training loop (train_ablation.py style),
       see train_before/nohup_train_oldcfg.log:6
    2. The unified train.py has additional logic (gradient clipping, NaN guard,
       GradScaler, etc.) that changes training dynamics
    3. Simply toggling CLI flags in train.py does not reproduce the old result
       (0.8156 vs 0.8333)

Key differences from train.py:
    - No gradient clipping (train.py clips at max_norm=1.0)
    - No early stopping (runs full 200 epochs)
    - Simpler training loop without extra guards

Recipe (matches oldcfg exactly):
    Focal(alpha=[0.15, 3.6, 4.56, 0.57]) + Dice, DS=False, boundary=False,
    EMA=False, CopyPaste=False, epochs=200, lr=6e-4, cosine warmup 10, seed=42

Usage:
    python tools/train_oldcfg.py
    python tools/train_oldcfg.py --output_dir ./output/baseline_oldcfg
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import os as _os
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
_os.chdir(_PROJECT_ROOT)

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
from torch.utils.tensorboard import SummaryWriter

from datasets.mos2_dataset import get_dataloaders, MoS2Dataset
from utils.losses import HybridLoss
from utils.metrics import SegmentationMetrics
from models.repela_net import repela_net_small


# ─── Training Infrastructure ─────────────────────────────────────────

def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger('oldcfg')
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


# ─── Training Loop (same as train_ablation.py) ───────────────────────
# NOTE: This is intentionally a simpler loop than train.py:
#   - No gradient clipping
#   - No NaN guard
#   - No AMP / GradScaler
#   - No EMA
# This matches the oldcfg training path that produced Val mIoU = 0.8333.

def train_one_epoch(model, loader, criterion, optimizer, device, logger,
                    log_interval=10):
    """Train for one epoch. Identical to train_ablation.py loop (DS=False)."""
    model.train()
    metrics = SegmentationMetrics(4)
    total_loss = total_focal = total_dice = 0
    num_batches = 0

    for i, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)

        output = model(images)
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
            logger.info(f'    Batch [{i+1}/{len(loader)}] Loss: {loss.item():.4f}')

    return (total_loss / num_batches, total_focal / num_batches,
            total_dice / num_batches, metrics.get_results())


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """Validate with full-image inference (pad to 32). Same as train_ablation.py."""
    model.eval()
    metrics = SegmentationMetrics(4)
    total_loss = total_ce = total_dice = 0
    loss_images = 0

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
                from tools.train import sliding_window_predict
                prediction = sliding_window_predict(
                    model, img_tensor, crop_size=512, stride=384, device=device
                )

            metrics.update(
                torch.from_numpy(prediction).unsqueeze(0),
                torch.from_numpy(mask_np).unsqueeze(0)
            )

    results = metrics.get_results()
    if loss_images > 0:
        results['val_loss'] = total_loss / loss_images
        results['val_ce'] = total_ce / loss_images
        results['val_dice'] = total_dice / loss_images
    return results


# ─── Deploy Model ────────────────────────────────────────────────────

def save_deploy_model(model, save_path, logger):
    """Save deploy (reparameterized) version of the model."""
    import copy
    deploy_model = copy.deepcopy(model).cpu()
    deploy_model.eval()
    if hasattr(deploy_model, 'switch_to_deploy'):
        deploy_model.switch_to_deploy()
    torch.save(deploy_model.state_dict(), save_path)
    logger.info('Deploy model saved.')


# ─── Main ────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(
        description='Old-config Baseline Training for RepELA-Small')
    parser.add_argument('--data_root', type=str, default='./Mos2_data')
    parser.add_argument('--split_dir', type=str, default='splits/')
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=6e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--val_crop_size', type=int, default=512)
    parser.add_argument('--val_stride', type=int, default=384)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='./output/baseline_oldcfg')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = get_args()

    # Strict reproducibility
    import random, numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'repela_small_{timestamp}')
    logger = setup_logger(output_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard'))

    # Model: RepELA-Small w/o CSE (use_cse=False, deep_supervision=False)
    model = repela_net_small(num_classes=args.num_classes,
                              deep_supervision=False, use_cse=False)
    model = model.to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6

    logger.info(f'Model: RepELA-Net-Small w/o CSE ({params:.2f}M params)')
    logger.info(f'Device: {device}')
    logger.info(f'Args: {args}')

    # Data (seed passed to get_dataloaders for generator + worker_init_fn)
    train_loader, val_loader = get_dataloaders(
        args.data_root, split_dir=args.split_dir,
        crop_size=args.crop_size, batch_size=args.batch_size,
        num_workers=args.num_workers, copy_paste=False,
        seed=args.seed
    )

    # Loss: Focal(class_weights) + Dice — matches oldcfg exactly
    criterion = HybridLoss(
        num_classes=args.num_classes,
        focal_alpha=MoS2Dataset.CLASS_WEIGHTS,
        focal_gamma=args.focal_gamma,
        loss_weights=(1.0, 1.0),
    ).to(device)
    logger.info(f'Loss: Focal(alpha={MoS2Dataset.CLASS_WEIGHTS}, gamma={args.focal_gamma}) + Dice')
    logger.info('NO: deep_supervision, boundary_loss, EMA, early_stopping, CopyPaste, gradient_clipping')
    logger.info(f'Validation: sliding window (crop={args.val_crop_size}, stride={args.val_stride})')

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup_epochs, args.epochs, args.min_lr)

    best_miou = 0.0
    class_names = MoS2Dataset.CLASSES

    logger.info(f'Training for {args.epochs} epochs (no early stopping)')
    logger.info('=' * 80)

    for epoch in range(args.epochs):
        lr = optimizer.param_groups[0]['lr']
        logger.info(f'Epoch [{epoch+1}/{args.epochs}] LR={lr:.6f}')

        t0 = time.time()
        train_loss, train_focal, train_dice, train_results = train_one_epoch(
            model, train_loader, criterion, optimizer, device, logger)
        val_results = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        val_loss = val_results.get('val_loss', 0)
        val_ce = val_results.get('val_ce', 0)
        val_dice = val_results.get('val_dice', 0)

        logger.info(f'  Train  Loss={train_loss:.4f} mIoU={train_results["mIoU"]:.4f}')
        logger.info(
            f'  Val    Loss={val_loss:.4f} (CE={val_ce:.4f} Dice={val_dice:.4f}) '
            f'mIoU={val_results["mIoU"]:.4f} F1={val_results["mean_f1"]:.4f}')
        iou_str = ' | '.join([
            f'{class_names[i]}: {val_results["per_class_iou"][i]:.4f}'
            for i in range(args.num_classes)
        ])
        logger.info(f'  IoU: {iou_str}')
        logger.info(f'  Time: {elapsed:.1f}s')

        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch + 1)
        writer.add_scalars('mIoU', {
            'train': train_results['mIoU'], 'val': val_results['mIoU']
        }, epoch + 1)
        writer.flush()

        if val_results['mIoU'] > best_miou:
            best_miou = val_results['mIoU']
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'best_miou': best_miou,
            }, os.path.join(output_dir, 'best_model.pth'))
            logger.info(f'  ★ New best mIoU: {best_miou:.4f}')

        # Periodic checkpoint
        if (epoch + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_miou': best_miou,
            }, os.path.join(output_dir, f'checkpoint_epoch{epoch+1}.pth'))

        logger.info('-' * 80)

    logger.info('=' * 80)
    logger.info(f'Done. Best mIoU: {best_miou:.4f}')

    # Save deploy model
    ckpt = torch.load(os.path.join(output_dir, 'best_model.pth'),
                      map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    save_deploy_model(model, os.path.join(output_dir, 'deploy_model.pth'), logger)

    writer.close()


if __name__ == '__main__':
    main()
