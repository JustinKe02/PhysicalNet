"""
Evaluate transfer-learning checkpoints on a held-out split.

This script reuses the transfer-learning inference protocol:
  - target-domain train-split normalization stats
  - deterministic sliding-window inference
  - class-aware confusion matrix and metric report

Outputs are written next to the checkpoint by default:
  - <split>_results.txt
  - <split>_confusion_matrix.png
  - <split>_preds/<basename>_pred.png
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.chdir(str(Path(__file__).resolve().parents[1]))

from models.repela_net import (
    repela_net_tiny,
    repela_net_small,
    repela_net_base,
    infer_use_cse,
)
from transfer.finetune import DATASET_CONFIGS, sliding_window_predict, plot_confusion_matrix
from transfer.material_dataset import MaterialDataset, compute_dataset_stats
from utils.metrics import SegmentationMetrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate transfer checkpoint on test split.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pth")
    parser.add_argument("--split", default="test", help="Split to evaluate, default=test")
    parser.add_argument("--data-root", default=None, help="Override dataset root")
    parser.add_argument("--name", default=None, help="Override experiment name")
    parser.add_argument("--num-classes", type=int, default=None, help="Override num classes")
    parser.add_argument("--model", default=None, choices=["tiny", "small", "base"], help="Override model size")
    parser.add_argument("--crop-size", type=int, default=None, help="Override eval crop size")
    parser.add_argument("--stride", type=int, default=None, help="Override eval stride")
    parser.add_argument("--output-dir", default=None, help="Directory to save results")
    parser.add_argument(
        "--save-preds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-image predictions",
    )
    return parser.parse_args()


def resolve_dataset_key(name: str):
    if not name:
        return None
    name_lower = name.lower()
    if name_lower in DATASET_CONFIGS:
        return name_lower
    matches = [k for k in DATASET_CONFIGS if name_lower.startswith(k)]
    if matches:
        return max(matches, key=len)
    return None


def load_model_from_checkpoint(checkpoint_path: str, num_classes: int, model_name: str,
                               norm_mean, norm_std, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    use_cse = infer_use_cse(ckpt, cli_use_cse=False)
    model_fn = {
        "tiny": repela_net_tiny,
        "small": repela_net_small,
        "base": repela_net_base,
    }[model_name]
    model = model_fn(
        num_classes=num_classes,
        use_cse=use_cse,
        norm_mean=norm_mean,
        norm_std=norm_std,
    ).to(device)

    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    for key in list(state_dict.keys()):
        if "color_enhance.mean" in key or "color_enhance.std" in key:
            del state_dict[key]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, ckpt, missing, unexpected


def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

    data_root = args.data_root or ckpt_args.get("data_root")
    if not data_root:
        raise ValueError("data_root is missing; pass --data-root explicitly")
    exp_name = args.name or ckpt_args.get("name") or Path(args.checkpoint).parent.name
    num_classes = args.num_classes or ckpt_args.get("num_classes")
    if num_classes is None:
        raise ValueError("num_classes is missing; pass --num-classes explicitly")
    model_name = args.model or ckpt_args.get("model", "small")
    crop_size = args.crop_size or ckpt_args.get("val_crop_size", 512)
    stride = args.stride or ckpt_args.get("val_stride", 384)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = output_dir / f"{args.split}_preds"
    if args.save_preds:
        pred_dir.mkdir(parents=True, exist_ok=True)

    ds_key = resolve_dataset_key(exp_name)
    class_names = (
        DATASET_CONFIGS[ds_key]["class_names"]
        if ds_key in DATASET_CONFIGS else [f"C{i}" for i in range(num_classes)]
    )

    stats = compute_dataset_stats(data_root, split=ckpt_args.get("train_split", "train"))
    ds_mean, ds_std = stats["mean"], stats["std"]

    dataset = MaterialDataset(
        data_root,
        split=args.split,
        crop_size=crop_size,
        augment=False,
        mean=ds_mean,
        std=ds_std,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No matched image/mask pairs found in {data_root} split={args.split}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt_loaded, missing, unexpected = load_model_from_checkpoint(
        args.checkpoint,
        num_classes=num_classes,
        model_name=model_name,
        norm_mean=ds_mean,
        norm_std=ds_std,
        device=device,
    )

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {data_root} | split={args.split} | pairs={len(dataset)}")
    print(f"Stats: mean={[round(x, 4) for x in ds_mean]}, std={[round(x, 4) for x in ds_std]}")
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")

    metrics = SegmentationMetrics(num_classes)
    per_image = []
    total_time = 0.0

    for idx, (img_path, mask_path) in enumerate(dataset.pairs, start=1):
        bn = Path(img_path).stem
        img = Image.open(img_path).convert("RGB")
        gt = np.array(Image.open(mask_path))
        img_tensor = TF.normalize(TF.to_tensor(img), ds_mean, ds_std)

        t0 = time.time()
        pred = sliding_window_predict(model, img_tensor, crop_size, stride, device)
        elapsed = time.time() - t0
        total_time += elapsed

        metrics.update(pred, gt)
        img_metrics = SegmentationMetrics(num_classes)
        img_metrics.update(pred, gt)
        per_image.append((bn, img_metrics.get_results()["mIoU"]))

        if args.save_preds:
            Image.fromarray(pred.astype(np.uint8)).save(pred_dir / f"{bn}_pred.png")

        print(f"  [{idx}/{len(dataset)}] {bn}: mIoU={per_image[-1][1]:.4f} ({elapsed:.2f}s)")

    results = metrics.get_results()
    cm_path = output_dir / f"{args.split}_confusion_matrix.png"
    plot_confusion_matrix(metrics.confusion_matrix, class_names, str(cm_path))

    results_path = output_dir / f"{args.split}_results.txt"
    with open(results_path, "w") as f:
        f.write(f"Dataset: {exp_name}\n")
        f.write(f"Data root: {data_root}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Checkpoint epoch: {ckpt_loaded.get('epoch', -1) + 1 if isinstance(ckpt_loaded, dict) and 'epoch' in ckpt_loaded else '?'}\n")
        f.write(f"Val-selected mIoU in checkpoint: {ckpt_loaded.get('best_miou', 0.0):.4f}\n")
        f.write(f"Eval crop/stride: {crop_size}/{stride}\n")
        f.write(f"Pairs: {len(dataset)}\n\n")
        f.write(f"mIoU: {results['mIoU']:.4f}\n")
        f.write(f"Pixel Acc: {results['pixel_acc']:.4f}\n")
        f.write(f"Mean F1: {results['mean_f1']:.4f}\n")
        f.write(f"Avg time: {total_time / len(dataset):.3f}s/image\n\n")
        for i, cname in enumerate(class_names):
            f.write(
                f"{cname:5s}  IoU={results['per_class_iou'][i]:.4f}  "
                f"F1={results['f1'][i]:.4f}  Acc={results['class_acc'][i]:.4f}\n"
            )
        f.write("\nPer-image mIoU:\n")
        for bn, miou in per_image:
            f.write(f"  {bn}: {miou:.4f}\n")

    print(f"Saved: {results_path}")
    print(f"Saved: {cm_path}")
    if args.save_preds:
        print(f"Saved predictions: {pred_dir}")


if __name__ == "__main__":
    main()
