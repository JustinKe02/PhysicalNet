"""
Unified Training Script for RepELA-Net, Baselines, and Ablation Studies.

All models share the SAME training recipe (crop, loss, AMP, grad clip, eval)
for fair and reproducible comparison on 4-class MoS2 segmentation.

Usage (from project root):
    # Main model
    python tools/train.py --model repela_small

    # Baseline (requires segmentation_models_pytorch)
    python tools/train.py --model unet_r18

    # Ablation (only repela_small)
    python tools/train.py --model repela_small --ablation no_ela

    # All baselines sequentially
    python tools/train.py --model all_baselines

    # All ablations sequentially
    python tools/train.py --model repela_small --ablation all
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
import sys
import math
import copy
import time
import random
import logging
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from models.repela_net import repela_net_tiny, repela_net_small, repela_net_base
from train_ablation import build_ablation_model, ALL_ABLATIONS, ABLATION_NAMES
from datasets.mos2_dataset import get_dataloaders, MoS2Dataset
from utils.losses import HybridLoss
from utils.metrics import SegmentationMetrics


# ═══════════════════════════════════════════════════════════════════════
# Model Registry
# ═══════════════════════════════════════════════════════════════════════

REPELA_MODELS = {
    'repela_tiny': lambda nc, ds, cse: repela_net_tiny(num_classes=nc, deep_supervision=ds, use_cse=cse),
    'repela_small': lambda nc, ds, cse: repela_net_small(num_classes=nc, deep_supervision=ds, use_cse=cse),
    'repela_base': lambda nc, ds, cse: repela_net_base(num_classes=nc, deep_supervision=ds, use_cse=cse),
}

# smp baselines — built lazily to avoid hard dependency
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

ALL_BASELINE_NAMES = list(SMP_MODEL_SPECS.keys())
ALL_MODEL_NAMES = list(REPELA_MODELS.keys()) + ALL_BASELINE_NAMES


def _build_smp_model(model_name, num_classes, pretrained=True):
    """Lazily import smp and build a baseline model."""
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        raise ImportError(
            f'segmentation_models_pytorch is required for baseline "{model_name}". '
            f'Install with: pip install segmentation-models-pytorch')
    arch, encoder = SMP_MODEL_SPECS[model_name]
    cls = getattr(smp, arch)
    weights = 'imagenet' if pretrained else None
    return cls(encoder_name=encoder, encoder_weights=weights,
               in_channels=3, classes=num_classes)


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser(description='Unified Training')

    # Data
    p.add_argument('--data_root', type=str, default='./Mos2_data')
    p.add_argument('--split_dir', type=str, default='splits/')
    p.add_argument('--crop_size', type=int, default=512)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)

    # Model
    p.add_argument('--model', type=str, default='repela_small',
                   choices=ALL_MODEL_NAMES + ['all_baselines'],
                   help='Model name or "all_baselines"')
    p.add_argument('--num_classes', type=int, default=4,
                   help='Number of classes (this script is designed for 4-class MoS2)')
    p.add_argument('--ablation', type=str, default=None,
                   choices=ALL_ABLATIONS + ['all', None],
                   help='Ablation variant (only with repela_small)')

    # Training
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=6e-4)
    p.add_argument('--min_lr', type=float, default=1e-6)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--warmup_epochs', type=int, default=10)

    # Loss
    p.add_argument('--focal_gamma', type=float, default=2.0)
    p.add_argument('--loss_weights', type=float, nargs='+', default=[1.0, 1.0])
    p.add_argument('--boundary_weight', type=float, default=0.0,
                   help='Weight for boundary loss (default 0)')

    # Eval
    p.add_argument('--val_crop_size', type=int, default=512)
    p.add_argument('--val_stride', type=int, default=384)

    # Output
    p.add_argument('--output_dir', type=str, default='./output')
    p.add_argument('--save_freq', type=int, default=10)

    # Other
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=False,
                   help='Mixed precision training (--amp / --no-amp)')
    p.add_argument('--deep_supervision', action=argparse.BooleanOptionalAction,
                   default=False)
    p.add_argument('--aux_loss_weight', type=float, default=0.4)
    p.add_argument('--ema', action=argparse.BooleanOptionalAction, default=False,
                   help='Use EMA model for validation (--ema / --no-ema)')
    p.add_argument('--ema_decay', type=float, default=0.9995)
    p.add_argument('--early_stop_patience', type=int, default=30)
    p.add_argument('--use_cse', action=argparse.BooleanOptionalAction, default=False,
                   help='Enable ColorSpaceEnhancement (--use_cse / --no-use_cse)')
    p.add_argument('--copy_paste', action=argparse.BooleanOptionalAction, default=False,
                   help='Enable CopyPaste augmentation (--copy_paste / --no-copy_paste)')
    p.add_argument('--no_pretrain', action='store_true', default=False,
                   help='Train smp baselines from scratch (no ImageNet pretrain)')
    p.add_argument('--resume', type=str, default=None)

    return p.parse_args()


def validate_args(args):
    """Check argument legality and fix incompatible combos."""
    is_smp = args.model in SMP_MODEL_SPECS
    is_repela = args.model in REPELA_MODELS

    # This script is designed for 4-class MoS2 benchmark
    if args.num_classes != 4:
        raise ValueError(
            f'This script is designed for 4-class MoS2. '
            f'Got --num_classes={args.num_classes}. '
            f'Use finetune.py for other datasets.')

    # smp models cannot have ablation
    if is_smp and args.ablation:
        raise ValueError(f'--ablation is not allowed with smp model "{args.model}"')

    # ablation only allowed with repela_small
    if args.ablation and args.model != 'repela_small':
        raise ValueError(f'--ablation only works with repela_small, got "{args.model}"')

    # smp models: no deep supervision, force off
    if is_smp:
        args.deep_supervision = False


# ═══════════════════════════════════════════════════════════════════════
# Infrastructure
# ═══════════════════════════════════════════════════════════════════════

def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger('unified')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(os.path.join(output_dir, 'train.log'))
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    return logger


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return max(min_lr / optimizer.defaults['lr'],
                   0.5 * (1.0 + math.cos(math.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class ModelEMA:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of model weights updated as:
        ema_weight = decay * ema_weight + (1 - decay) * model_weight

    Using the EMA model for validation smooths out epoch-to-epoch
    fluctuations and typically yields better final performance.
    """
    def __init__(self, model, decay=0.9995):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            for ema_p, p in zip(self.ema.parameters(), model.parameters()):
                ema_p.data.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, state_dict):
        self.ema.load_state_dict(state_dict)


def build_model(args):
    """Build model from args. Returns (model, display_name, has_deep_sup)."""
    nc = args.num_classes

    # Ablation (e.g. with_cse, no_ela, etc.)
    if args.ablation:
        model = build_ablation_model(args.ablation, nc,
                                     deep_supervision=args.deep_supervision)
        name = ABLATION_NAMES.get(args.ablation, args.ablation)
        has_ds = args.deep_supervision
        return model, name, has_ds

    # RepELA-Net
    if args.model in REPELA_MODELS:
        model = REPELA_MODELS[args.model](nc, args.deep_supervision, args.use_cse)
        cse_tag = '+CSE' if args.use_cse else ''
        name = f'RepELA-Net-{args.model.split("_")[1].title()}{cse_tag}'
        has_ds = args.deep_supervision
        return model, name, has_ds

    # smp baseline (lazy import)
    if args.model in SMP_MODEL_SPECS:
        pretrained = not getattr(args, 'no_pretrain', False)
        model = _build_smp_model(args.model, nc, pretrained=pretrained)
        pt_tag = '' if pretrained else ' (scratch)'
        name = f'{args.model}{pt_tag}'
        return model, name, False

    raise ValueError(f'Unknown model: {args.model}')


# ═══════════════════════════════════════════════════════════════════════
# Sliding Window Predict
# ═══════════════════════════════════════════════════════════════════════

def sliding_window_predict(model, img_tensor, crop_size, stride, device,
                           num_classes=4, is_smp=False):
    """Sliding window inference on a single full-res image.

    For smp models, each crop is padded to a multiple of 32.
    """
    _, H, W = img_tensor.shape
    pred_sum = torch.zeros(num_classes, H, W, dtype=torch.float32, device=device)
    count = torch.zeros(H, W, dtype=torch.float32, device=device)

    pad_h = max(0, crop_size - H)
    pad_w = max(0, crop_size - W)
    if pad_h > 0 or pad_w > 0:
        img_tensor = F.pad(img_tensor, [0, pad_w, 0, pad_h], mode='reflect')

    _, pH, pW = img_tensor.shape
    y_pos = sorted(set(
        list(range(0, max(1, pH - crop_size + 1), stride)) +
        [max(0, pH - crop_size)]
    ))
    x_pos = sorted(set(
        list(range(0, max(1, pW - crop_size + 1), stride)) +
        [max(0, pW - crop_size)]
    ))

    for y in y_pos:
        for x in x_pos:
            crop = img_tensor[:, y:y+crop_size, x:x+crop_size]

            # smp models need input padded to multiple of 32
            if is_smp:
                _, cH, cW = crop.shape
                smp_pad_h = (32 - cH % 32) % 32
                smp_pad_w = (32 - cW % 32) % 32
                if smp_pad_h > 0 or smp_pad_w > 0:
                    crop = F.pad(crop, [0, smp_pad_w, 0, smp_pad_h], mode='reflect')

            with torch.no_grad():
                output = model(crop.unsqueeze(0).to(device))
                logits = output[0] if isinstance(output, tuple) else output
                # Crop back smp padding
                if is_smp and (smp_pad_h > 0 or smp_pad_w > 0):
                    logits = logits[:, :, :cH, :cW]
                probs = F.softmax(logits, dim=1)[0]

            y_end = min(y + crop_size, H)
            x_end = min(x + crop_size, W)
            pred_sum[:, y:y_end, x:x_end] += probs[:, :y_end-y, :x_end-x]
            count[y:y_end, x:x_end] += 1

    count = count.clamp(min=1)
    return (pred_sum / count.unsqueeze(0)).argmax(dim=0).cpu().numpy()


def predict_smp(model, img_tensor, device, num_classes=4):
    """Full-image prediction for smp models (pad to 32)."""
    _, H, W = img_tensor.shape
    pad_h = (32 - H % 32) % 32
    pad_w = (32 - W % 32) % 32
    img_padded = F.pad(img_tensor, [0, pad_w, 0, pad_h], mode='reflect')
    img = img_padded.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img)
    logits = logits[:, :, :H, :W]
    return logits


# ═══════════════════════════════════════════════════════════════════════
# Train & Validate
# ═══════════════════════════════════════════════════════════════════════

def train_one_epoch(model, train_loader, criterion, optimizer, scaler,
                    device, args, has_deep_sup, is_smp, logger=None, ema=None):
    model.train()
    metrics = SegmentationMetrics(args.num_classes)
    total_loss = total_ce = total_dice = 0
    num_batches = 0

    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        # smp models: pad to multiple of 32
        if is_smp:
            _, _, H, W = images.shape
            pad_h = (32 - H % 32) % 32
            pad_w = (32 - W % 32) % 32
            if pad_h > 0 or pad_w > 0:
                images = F.pad(images, [0, pad_w, 0, pad_h], mode='reflect')

        optimizer.zero_grad()

        if args.amp:
            with autocast('cuda'):
                output = model(images)
                if has_deep_sup and isinstance(output, tuple):
                    logits, aux_list = output
                else:
                    logits = output[0] if isinstance(output, tuple) else output
                    aux_list = []

                # Crop back for smp
                if is_smp and (pad_h > 0 or pad_w > 0):
                    logits = logits[:, :, :H, :W]

                loss, ce_loss, dice_loss = criterion(logits, masks)
                for aux in aux_list:
                    aux_l, _, _ = criterion(aux, masks)
                    loss = loss + args.aux_loss_weight * aux_l

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # NaN guard: skip step if gradients exploded
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if torch.isfinite(grad_norm):
                scaler.step(optimizer)
            else:
                if logger:
                    logger.warning(f'    NaN grad at batch {batch_idx+1}, skipping step')
            scaler.update()
        else:
            output = model(images)
            if has_deep_sup and isinstance(output, tuple):
                logits, aux_list = output
            else:
                logits = output[0] if isinstance(output, tuple) else output
                aux_list = []

            if is_smp and (pad_h > 0 or pad_w > 0):
                logits = logits[:, :, :H, :W]

            loss, ce_loss, dice_loss = criterion(logits, masks)
            for aux in aux_list:
                aux_l, _, _ = criterion(aux, masks)
                loss = loss + args.aux_loss_weight * aux_l

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # EMA update after each step
        if ema is not None:
            ema.update(model)

        # NaN guard for loss accumulation
        if not torch.isfinite(loss):
            if logger:
                logger.warning(f'    NaN loss at batch {batch_idx+1}, skipping accumulation')
            continue

        preds = logits.argmax(dim=1)
        metrics.update(preds, masks)
        total_loss += loss.item()
        total_ce += ce_loss.item()
        total_dice += dice_loss.item()
        num_batches += 1

        if (batch_idx + 1) % 10 == 0 and logger:
            logger.info(f'    Batch [{batch_idx+1}/{len(train_loader)}] '
                        f'Loss: {loss.item():.4f}')

    return (total_loss / max(1, num_batches),
            total_ce / max(1, num_batches),
            total_dice / max(1, num_batches),
            metrics.get_results())


@torch.no_grad()
def validate(model, val_loader, criterion, device, args, has_deep_sup, is_smp):
    model.eval()
    metrics = SegmentationMetrics(args.num_classes)
    total_loss = total_ce = total_dice = 0
    num_images = 0
    loss_images = 0

    for images_list, masks_list in val_loader:
        for img_tensor, mask_tensor in zip(images_list, masks_list):
            mask_np = mask_tensor.numpy() if isinstance(mask_tensor, torch.Tensor) else mask_tensor

            try:
                if is_smp:
                    logits = predict_smp(model, img_tensor, device, args.num_classes)
                else:
                    img = img_tensor.unsqueeze(0).to(device)
                    output = model(img)
                    logits = output[0] if isinstance(output, tuple) else output

                mask_dev = torch.from_numpy(mask_np).unsqueeze(0).long().to(device)
                loss, ce, dice = criterion(logits, mask_dev)
                total_loss += loss.item()
                total_ce += ce.item()
                total_dice += dice.item()
                loss_images += 1
                prediction = logits.argmax(dim=1)[0].cpu().numpy()

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                prediction = sliding_window_predict(
                    model, img_tensor,
                    crop_size=args.val_crop_size,
                    stride=args.val_stride,
                    device=device,
                    num_classes=args.num_classes,
                    is_smp=is_smp,
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


# ═══════════════════════════════════════════════════════════════════════
# Main Training Loop
# ═══════════════════════════════════════════════════════════════════════

def train_single(args):
    """Train a single model configuration."""
    # Seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Run name
    if args.ablation:
        run_name = f'ablation_{args.ablation}'
    else:
        run_name = args.model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'{run_name}_{timestamp}')

    logger = setup_logger(output_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model
    model, model_name, has_deep_sup = build_model(args)
    is_smp = args.model in SMP_MODEL_SPECS
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Model: {model_name} ({total_params/1e6:.2f}M params)')
    logger.info(f'Device: {device}')
    logger.info(f'Args: {args}')

    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard'))

    # Data
    train_loader, val_loader = get_dataloaders(
        args.data_root, split_dir=args.split_dir,
        crop_size=args.crop_size, batch_size=args.batch_size,
        num_workers=args.num_workers, copy_paste=args.copy_paste,
    )

    # Loss
    criterion = HybridLoss(
        num_classes=args.num_classes,
        focal_alpha=MoS2Dataset.CLASS_WEIGHTS,
        focal_gamma=args.focal_gamma,
        loss_weights=tuple(args.loss_weights),
        boundary_weight=args.boundary_weight,
    ).to(device)

    # Optimizer & scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup_epochs, args.epochs, args.min_lr)
    scaler = GradScaler('cuda', enabled=args.amp, init_scale=1024)

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
        logger.info(f'Resumed from epoch {start_epoch}, best mIoU={best_miou:.4f}')

    # EMA
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema else None
    if ema:
        logger.info(f'EMA: enabled (decay={args.ema_decay})')

    class_names = MoS2Dataset.CLASSES
    logger.info(f'AMP: {args.amp} | Deep supervision: {has_deep_sup} | '
                f'Boundary weight: {args.boundary_weight}')
    logger.info(f'Validation: sliding window (crop={args.val_crop_size}, '
                f'stride={args.val_stride})')
    logger.info(f'Early stopping patience: {args.early_stop_patience}')
    logger.info('=' * 80)

    no_improve = 0

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        lr = optimizer.param_groups[0]['lr']
        logger.info(f'Epoch [{epoch+1}/{args.epochs}] LR={lr:.6f}')

        # Train
        tr_loss, tr_ce, tr_dice, tr_res = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, args, has_deep_sup, is_smp, logger, ema)

        # Validate (always use original model — EMA weights are for deploy only)
        val_res = validate(model, val_loader, criterion, device,
                           args, has_deep_sup, is_smp)

        scheduler.step()
        dt = time.time() - t0

        # Log
        v_loss = val_res.get('val_loss', 0)
        v_ce = val_res.get('val_ce', 0)
        v_dice = val_res.get('val_dice', 0)
        logger.info(f'  Train  Loss={tr_loss:.4f} (CE={tr_ce:.4f} Dice={tr_dice:.4f}) '
                    f'mIoU={tr_res["mIoU"]:.4f}')
        logger.info(f'  Val    Loss={v_loss:.4f} (CE={v_ce:.4f} Dice={v_dice:.4f}) '
                    f'mIoU={val_res["mIoU"]:.4f} F1={val_res["mean_f1"]:.4f}')
        iou_str = ' | '.join(
            f'{class_names[i]}: {val_res["per_class_iou"][i]:.4f}'
            for i in range(args.num_classes))
        logger.info(f'  IoU: {iou_str}')
        logger.info(f'  Time: {dt:.1f}s')
        logger.info('-' * 80)

        # TensorBoard
        writer.add_scalars('Loss', {'train': tr_loss, 'val': v_loss}, epoch+1)
        writer.add_scalars('mIoU', {'train': tr_res['mIoU'],
                                    'val': val_res['mIoU']}, epoch+1)
        writer.add_scalar('LR', lr, epoch+1)
        for i, cn in enumerate(class_names):
            writer.add_scalar(f'Val_IoU/{cn}',
                              val_res['per_class_iou'][i], epoch+1)
        writer.flush()

        # Save best
        if val_res['mIoU'] > best_miou:
            best_miou = val_res['mIoU']
            no_improve = 0
            save_dict = {
                'epoch': epoch, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_miou': best_miou, 'args': vars(args),
            }
            if ema:
                save_dict['ema'] = ema.state_dict()
            torch.save(save_dict, os.path.join(output_dir, 'best_model.pth'))
            logger.info(f'  ★ New best mIoU: {best_miou:.4f}')
        else:
            no_improve += 1
            if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
                logger.info(f'  Early stopping ({args.early_stop_patience} epochs)')
                break

        if (epoch + 1) % args.save_freq == 0:
            save_dict = {
                'epoch': epoch, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_miou': best_miou, 'args': vars(args),
            }
            if ema:
                save_dict['ema'] = ema.state_dict()
            torch.save(save_dict, os.path.join(output_dir, f'checkpoint_epoch{epoch+1}.pth'))

    logger.info('=' * 80)
    logger.info(f'Done. Best mIoU: {best_miou:.4f}')

    # Deploy model (RepELA-Net only)
    if args.model in REPELA_MODELS and (not args.ablation or args.ablation == 'with_cse'):
        try:
            # Use EMA weights for deploy if available
            deploy_model = ema.ema if ema else model
            deploy_model.switch_to_deploy()
            torch.save(deploy_model.state_dict(),
                       os.path.join(output_dir, 'deploy_model.pth'))
            logger.info('Deploy model saved (EMA weights).' if ema else 'Deploy model saved.')
        except Exception as e:
            logger.warning(f'switch_to_deploy failed: {e}')

    writer.close()
    return model_name, f'{total_params/1e6:.2f}M', best_miou


def main():
    args = get_args()

    # Handle batch modes
    if args.model == 'all_baselines':
        results = []
        for m in ALL_BASELINE_NAMES:
            print(f'\n{"="*80}\n  Training baseline: {m}\n{"="*80}')
            args.model = m
            validate_args(args)
            name, params, miou = train_single(args)
            results.append((m, name, params, miou))
        print('\n\n' + '=' * 60)
        print('ALL BASELINES COMPLETE')
        print(f'{"Model":<20} {"Params":<10} {"mIoU":<10}')
        for m, name, params, miou in results:
            print(f'{m:<20} {params:<10} {miou:.4f}')
        return

    if args.ablation == 'all':
        results = []
        for abl in ALL_ABLATIONS:
            print(f'\n{"="*80}\n  Ablation: {ABLATION_NAMES[abl]}\n{"="*80}')
            args.ablation = abl
            validate_args(args)
            name, params, miou = train_single(args)
            results.append((abl, name, params, miou))
        print('\n\n' + '=' * 60)
        print('ALL ABLATIONS COMPLETE')
        print(f'{"Ablation":<20} {"Name":<35} {"Params":<10} {"mIoU":<10}')
        for abl, name, params, miou in results:
            print(f'{abl:<20} {name:<35} {params:<10} {miou:.4f}')
        return

    # Single model
    validate_args(args)
    train_single(args)


if __name__ == '__main__':
    main()
