"""
Baseline Model Training Script.

Trains standard segmentation models on MoS2 dataset using the SAME
data pipeline, loss function, metrics, and evaluation protocol
as RepELA-Net for fair horizontal comparison.

Supported models (via segmentation_models_pytorch):
  - unet_r18       U-Net with ResNet-18 encoder
  - unet_r34       U-Net with ResNet-34 encoder
  - deeplabv3p_mv2 DeepLabV3+ with MobileNetV2 encoder
  - deeplabv3p_r18 DeepLabV3+ with ResNet-18 encoder
  - pspnet_r18     PSPNet with ResNet-18 encoder
  - fpn_r18        FPN with ResNet-18 encoder
  - segformer_b0   SegFormer-B0 (mit_b0 encoder)
  - bisenetv2      BiSeNet V2 (custom lightweight)

Usage:
    python train_baseline.py --model unet_r18 --epochs 200
    python train_baseline.py --model deeplabv3p_mv2 --epochs 200
    python train_baseline.py --model all  # Train all baselines sequentially
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
from torch.utils.tensorboard import SummaryWriter

import segmentation_models_pytorch as smp

from datasets.mos2_dataset import get_dataloaders, MoS2Dataset
from utils.losses import HybridLoss
from utils.metrics import SegmentationMetrics


# ─── Model Registry ──────────────────────────────────────────────────

def build_model(model_name, num_classes=4):
    """Build a baseline segmentation model.

    All models use ImageNet-pretrained encoders for fair comparison.
    Returns: (model, param_count_str)
    """
    models = {
        'unet_r18': lambda: smp.Unet(
            encoder_name='resnet18', encoder_weights='imagenet',
            in_channels=3, classes=num_classes,
        ),
        'unet_r34': lambda: smp.Unet(
            encoder_name='resnet34', encoder_weights='imagenet',
            in_channels=3, classes=num_classes,
        ),
        'deeplabv3p_mv2': lambda: smp.DeepLabV3Plus(
            encoder_name='mobilenet_v2', encoder_weights='imagenet',
            in_channels=3, classes=num_classes,
        ),
        'deeplabv3p_r18': lambda: smp.DeepLabV3Plus(
            encoder_name='resnet18', encoder_weights='imagenet',
            in_channels=3, classes=num_classes,
        ),
        'pspnet_r18': lambda: smp.PSPNet(
            encoder_name='resnet18', encoder_weights='imagenet',
            in_channels=3, classes=num_classes,
        ),
        'fpn_r18': lambda: smp.FPN(
            encoder_name='resnet18', encoder_weights='imagenet',
            in_channels=3, classes=num_classes,
        ),
        'segformer_b0': lambda: smp.Unet(
            encoder_name='mit_b0', encoder_weights='imagenet',
            in_channels=3, classes=num_classes,
        ),
    }

    if model_name not in models:
        raise ValueError(
            f'Unknown model: {model_name}\n'
            f'Available: {", ".join(models.keys())}'
        )

    model = models[model_name]()
    params = sum(p.numel() for p in model.parameters())
    return model, f'{params/1e6:.2f}M'


ALL_MODELS = [
    'unet_r18', 'deeplabv3p_mv2', 'deeplabv3p_r18',
    'pspnet_r18', 'fpn_r18', 'segformer_b0',
]


# ─── Training Infrastructure ─────────────────────────────────────────

def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger('baseline')
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


# ─── Sliding Window Predict ──────────────────────────────────────────

@torch.no_grad()
def sliding_window_predict(model, img_tensor, crop_size, stride, device):
    _, H, W = img_tensor.shape
    num_classes = 4
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
            probs = F.softmax(model(crop), dim=1)[0]
            y_end = min(y + crop_size, H)
            x_end = min(x + crop_size, W)
            pred_sum[:, y:y_end, x:x_end] += probs[:, :y_end-y, :x_end-x]
            count[y:y_end, x:x_end] += 1

    count = count.clamp(min=1)
    return (pred_sum / count.unsqueeze(0)).argmax(dim=0).cpu().numpy()


# ─── Train & Validate ────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, logger, log_interval=10):
    model.train()
    metrics = SegmentationMetrics(4)
    total_loss = total_focal = total_dice = 0
    num_batches = 0

    for i, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
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
            logger.info(
                f'  Batch [{i+1}/{len(loader)}] '
                f'Loss: {loss.item():.4f} (Focal: {focal.item():.4f}, Dice: {dice.item():.4f})'
            )

    return (total_loss / num_batches, total_focal / num_batches,
            total_dice / num_batches, metrics.get_results())


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()
    metrics = SegmentationMetrics(4)
    total_loss = total_ce = total_dice = 0
    num_images = 0

    for images_list, masks_list in val_loader:
        for img_tensor, mask_tensor in zip(images_list, masks_list):
            mask_np = mask_tensor.numpy() if isinstance(mask_tensor, torch.Tensor) else mask_tensor

            try:
                # Pad to nearest multiple of 32 for encoder compatibility
                _, H, W = img_tensor.shape
                pad_h = (32 - H % 32) % 32
                pad_w = (32 - W % 32) % 32
                img_padded = F.pad(img_tensor, [0, pad_w, 0, pad_h], mode='reflect')

                img = img_padded.unsqueeze(0).to(device)
                logits = model(img)
                logits = logits[:, :, :H, :W]  # Crop back to original size

                mask_dev = torch.from_numpy(mask_np).unsqueeze(0).long().to(device)
                loss, ce_loss, dice_loss = criterion(logits, mask_dev)
                total_loss += loss.item()
                total_ce += ce_loss.item()
                total_dice += dice_loss.item()
                prediction = logits.argmax(dim=1)[0].cpu().numpy()
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                torch.cuda.empty_cache()
                prediction = sliding_window_predict(
                    model, img_tensor, crop_size=512, stride=384, device=device
                )

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


# ─── Main ────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description='Baseline Model Training')
    parser.add_argument('--data_root', type=str, default='./Mos2_data')
    parser.add_argument('--split_dir', type=str, default='splits/')
    parser.add_argument('--model', type=str, default='unet_r18',
                        help=f'Model name or "all". Options: {", ".join(ALL_MODELS)}')
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=768)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=6e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--early_stop_patience', type=int, default=20)
    parser.add_argument('--output_dir', type=str, default='./output/baselines')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def train_single_model(model_name, args):
    """Train a single baseline model."""
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'{model_name}_{timestamp}')
    logger = setup_logger(output_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard'))

    # Data (same as RepELA-Net)
    train_loader, val_loader = get_dataloaders(
        args.data_root, split_dir=args.split_dir,
        crop_size=args.crop_size, batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Model
    model, param_str = build_model(model_name, args.num_classes)
    model = model.to(device)
    logger.info(f'Model: {model_name}, Params: {param_str}')

    # Loss (same as RepELA-Net)
    criterion = HybridLoss(
        num_classes=args.num_classes,
        focal_alpha=MoS2Dataset.CLASS_WEIGHTS,
        focal_gamma=2.0,
        loss_weights=(1.0, 1.0)
    ).to(device)

    # Optimizer & scheduler (same as RepELA-Net)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup_epochs, args.epochs, args.min_lr
    )

    best_miou = 0.0
    no_improve_count = 0
    class_names = MoS2Dataset.CLASSES

    logger.info(f'Training {model_name} for {args.epochs} epochs...')
    logger.info(f'Crop: {args.crop_size}, Batch: {args.batch_size}, LR: {args.lr}')
    logger.info(f'Early stopping patience: {args.early_stop_patience}')
    logger.info('=' * 80)

    for epoch in range(args.epochs):
        epoch_start = time.time()
        lr = optimizer.param_groups[0]['lr']
        logger.info(f'Epoch [{epoch+1}/{args.epochs}] LR: {lr:.6f}')

        train_loss, train_focal, train_dice, train_results = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, logger
        )

        val_results = validate(model, val_loader, criterion, device)

        scheduler.step()
        epoch_time = time.time() - epoch_start

        val_loss = val_results.get('val_loss', 0)
        val_ce = val_results.get('val_ce', 0)
        val_dice = val_results.get('val_dice', 0)

        logger.info(
            f'  Train Loss: {train_loss:.4f} '
            f'(Focal: {train_focal:.4f}, Dice: {train_dice:.4f}) '
            f'mIoU: {train_results["mIoU"]:.4f} Acc: {train_results["pixel_acc"]:.4f}'
        )
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

        # TensorBoard
        writer.add_scalars('Loss/Total', {'train': train_loss, 'val': val_loss}, epoch + 1)
        writer.add_scalars('mIoU', {
            'train': train_results['mIoU'], 'val': val_results['mIoU']
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
                'best_miou': best_miou, 'model_name': model_name,
            }, os.path.join(output_dir, 'best_model.pth'))
            logger.info(f'  ★ New best mIoU: {best_miou:.4f}')
        else:
            no_improve_count += 1
            if args.early_stop_patience > 0 and no_improve_count >= args.early_stop_patience:
                logger.info(f'  Early stopping at epoch {epoch+1}')
                break

    logger.info('=' * 80)
    logger.info(f'{model_name} training complete. Best val mIoU: {best_miou:.4f}')
    writer.close()

    return model_name, param_str, best_miou


def main():
    args = get_args()

    if args.model == 'all':
        results = []
        for model_name in ALL_MODELS:
            print(f'\n{"="*80}')
            print(f'Training: {model_name}')
            print(f'{"="*80}\n')
            name, params, miou = train_single_model(model_name, args)
            results.append((name, params, miou))

        # Summary table
        print('\n' + '=' * 60)
        print('BASELINE COMPARISON SUMMARY')
        print('=' * 60)
        print(f'{"Model":<25} {"Params":<10} {"Val mIoU":<10}')
        print('-' * 45)
        for name, params, miou in results:
            print(f'{name:<25} {params:<10} {miou:.4f}')
        print('=' * 60)
    else:
        train_single_model(args.model, args)


if __name__ == '__main__':
    main()
