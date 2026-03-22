"""Reusable evaluation script for MoS2 checkpoints.

Examples:
    python tmp/test_eval.py \
        --model repela_small \
        --ablation no_color \
        --checkpoint outputv3/ablation_no_color_*/best_model.pth

    python tmp/test_eval.py \
        --model unet_mit_b0 \
        --checkpoint outputv3/unet_mit_b0_*/best_model.pth
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.mos2_dataset import MoS2Dataset
from models.repela_net import repela_net_tiny, repela_net_small, repela_net_base
from train_ablation import ALL_ABLATIONS, build_ablation_model
from utils.metrics import SegmentationMetrics


CLASS_NAMES = ['background', 'monolayer', 'fewlayer', 'multilayer']
REPELA_MODELS = {
    'repela_tiny': repela_net_tiny,
    'repela_small': repela_net_small,
    'repela_base': repela_net_base,
}
SMP_MODEL_SPECS = {
    'unet_r18': ('Unet', 'resnet18'),
    'unet_r34': ('Unet', 'resnet34'),
    'deeplabv3p_r18': ('DeepLabV3Plus', 'resnet18'),
    'deeplabv3p_mv2': ('DeepLabV3Plus', 'mobilenet_v2'),
    'pspnet_r18': ('PSPNet', 'resnet18'),
    'fpn_r18': ('FPN', 'resnet18'),
    'unet_mit_b0': ('Unet', 'mit_b0'),
}
ALL_MODELS = list(REPELA_MODELS) + list(SMP_MODEL_SPECS)


def parse_args():
    parser = argparse.ArgumentParser(description='Reusable test/val evaluation')
    parser.add_argument('--data_root', type=str, default=str(REPO_ROOT / 'Mos2_data'))
    parser.add_argument('--split_dir', type=str, default='splits/')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--model', type=str, required=True, choices=ALL_MODELS)
    parser.add_argument('--ablation', type=str, default=None,
                        choices=ALL_ABLATIONS,
                        help='Only valid with repela_small')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to best_model.pth or raw state_dict file')
    parser.add_argument('--use_ema', action='store_true',
                        help='Load ckpt[\"ema\"] instead of ckpt[\"model\"] when available')
    parser.add_argument('--deploy', action='store_true',
                        help='Load raw deploy state_dict for RepELA models')
    parser.add_argument('--deep_supervision', action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--stride', type=int, default=384)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--label', type=str, default=None)
    parser.add_argument('--output', type=str, default=None,
                        help='Optional path to save plain-text metrics')
    return parser.parse_args()


def validate_args(args):
    if args.ablation and args.model != 'repela_small':
        raise ValueError('--ablation only works with --model repela_small')
    if args.deploy and args.model in SMP_MODEL_SPECS:
        raise ValueError('--deploy only applies to RepELA checkpoints')
    if args.model in SMP_MODEL_SPECS:
        args.deep_supervision = False


def build_model(args, device):
    is_smp = args.model in SMP_MODEL_SPECS

    if args.ablation:
        model = build_ablation_model(
            args.ablation,
            args.num_classes,
            deep_supervision=args.deep_supervision,
        )
        return model.to(device), False

    if args.model in REPELA_MODELS:
        model = REPELA_MODELS[args.model](
            num_classes=args.num_classes,
            deep_supervision=args.deep_supervision,
            deploy=args.deploy,
        )
        return model.to(device), False

    arch, encoder = SMP_MODEL_SPECS[args.model]
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ImportError(
            f'segmentation_models_pytorch is required for {args.model}'
        ) from exc
    model_cls = getattr(smp, arch)
    model = model_cls(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=3,
        classes=args.num_classes,
    )
    return model.to(device), True


def load_state_dict_for_eval(args, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if args.deploy:
        if not isinstance(checkpoint, dict):
            return checkpoint
        if 'model' in checkpoint or 'ema' in checkpoint:
            raise ValueError('--deploy expects a raw deploy state_dict file, not best_model.pth')
        return checkpoint

    if isinstance(checkpoint, dict) and ('model' in checkpoint or 'ema' in checkpoint):
        if args.use_ema:
            if 'ema' not in checkpoint:
                raise ValueError('Checkpoint has no "ema" weights')
            return checkpoint['ema']
        return checkpoint['model']

    return checkpoint


def sliding_window_predict(model, img_tensor, device, crop_size, stride,
                           num_classes=4, is_smp=False):
    _, height, width = img_tensor.shape
    pred_sum = torch.zeros(num_classes, height, width, dtype=torch.float32, device=device)
    count = torch.zeros(height, width, dtype=torch.float32, device=device)

    pad_h = max(0, crop_size - height)
    pad_w = max(0, crop_size - width)
    if pad_h > 0 or pad_w > 0:
        img_tensor = F.pad(img_tensor, [0, pad_w, 0, pad_h], mode='reflect')

    _, padded_h, padded_w = img_tensor.shape
    y_positions = sorted(set(
        list(range(0, max(1, padded_h - crop_size + 1), stride)) +
        [max(0, padded_h - crop_size)]
    ))
    x_positions = sorted(set(
        list(range(0, max(1, padded_w - crop_size + 1), stride)) +
        [max(0, padded_w - crop_size)]
    ))

    for y in y_positions:
        for x in x_positions:
            crop = img_tensor[:, y:y+crop_size, x:x+crop_size]
            crop_h, crop_w = crop.shape[1], crop.shape[2]
            smp_pad_h = smp_pad_w = 0

            if is_smp:
                smp_pad_h = (32 - crop_h % 32) % 32
                smp_pad_w = (32 - crop_w % 32) % 32
                if smp_pad_h > 0 or smp_pad_w > 0:
                    crop = F.pad(crop, [0, smp_pad_w, 0, smp_pad_h], mode='reflect')

            with torch.no_grad():
                output = model(crop.unsqueeze(0).to(device))
                logits = output[0] if isinstance(output, tuple) else output
                if is_smp and (smp_pad_h > 0 or smp_pad_w > 0):
                    logits = logits[:, :, :crop_h, :crop_w]
                probs = F.softmax(logits, dim=1)[0]

            y_end = min(y + crop_size, height)
            x_end = min(x + crop_size, width)
            pred_sum[:, y:y_end, x:x_end] += probs[:, :y_end-y, :x_end-x]
            count[y:y_end, x:x_end] += 1

    count = count.clamp(min=1)
    return (pred_sum / count.unsqueeze(0)).argmax(dim=0)


def evaluate(model, loader, device, crop_size, stride, num_classes, is_smp):
    model.eval()
    metrics = SegmentationMetrics(num_classes)
    for images, masks in loader:
        image = images.squeeze(0)
        mask = masks.squeeze(0)
        pred = sliding_window_predict(
            model,
            image,
            device=device,
            crop_size=crop_size,
            stride=stride,
            num_classes=num_classes,
            is_smp=is_smp,
        )
        metrics.update(pred.unsqueeze(0), mask.unsqueeze(0).to(device))
    return metrics.get_results()


def format_results(label, args, results):
    lines = []
    lines.append('=' * 60)
    lines.append(f'  {label}')
    lines.append('=' * 60)
    lines.append(f'  split: {args.split}')
    lines.append(f'  checkpoint: {args.checkpoint}')
    lines.append(f'  mIoU:  {results["mIoU"]:.4f}')
    lines.append(f'  F1:    {results["mean_f1"]:.4f}')
    for idx, name in enumerate(CLASS_NAMES[:args.num_classes]):
        lines.append(
            f'  {name:>12}: IoU={results["per_class_iou"][idx]:.4f}  '
            f'F1={results["f1"][idx]:.4f}'
        )
    return '\n'.join(lines)


def main():
    args = parse_args()
    validate_args(args)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    dataset = MoS2Dataset(
        args.data_root,
        args.split,
        split_dir=args.split_dir,
        crop_size=None,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model, is_smp = build_model(args, device)
    state_dict = load_state_dict_for_eval(args, checkpoint_path)
    model.load_state_dict(state_dict, strict=True)

    label = args.label or (
        f'{args.model} ({args.ablation})' if args.ablation else args.model
    )
    print(f'{args.split} set: {len(dataset)} images')
    results = evaluate(
        model,
        loader,
        device=device,
        crop_size=args.crop_size,
        stride=args.stride,
        num_classes=args.num_classes,
        is_smp=is_smp,
    )
    text = format_results(label, args, results)
    print('\n' + text + '\n')

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + '\n')
        print(f'Saved metrics to: {output_path}')


if __name__ == '__main__':
    main()
