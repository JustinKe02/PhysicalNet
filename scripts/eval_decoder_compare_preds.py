"""
Evaluate decoder-comparison checkpoints on MoS2 split and save per-image masks.

This fills output/eval_results/decoder_<name>/ with:
  - <split>_metrics.txt
  - <split>_confusion_matrix.png
  - <basename>_pred.png

The inference protocol matches the decoder comparison records:
deterministic sliding-window inference with crop=512, stride=384.
"""

import argparse
import glob
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

from models.decoders_compare import DECODER_NAMES
from tools.eval import (
    MEAN, STD, CLASSES, CLASS_LABELS_SHORT,
    load_split, sliding_window_predict, plot_confusion_matrix,
)
from tools.train_decoder_compare import build_encoder_with_decoder
from utils.metrics import SegmentationMetrics


ALL_DECODERS = ["unet", "fpn", "aspp", "segformer", "ppm", "hamburger", "ours"]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate decoder-compare checkpoints and save preds.")
    parser.add_argument("--decoders", nargs="+", default=["all"], help="Decoder names or 'all'")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--data-root", default="Mos2_data")
    parser.add_argument("--split-dir", default="splits")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--output-root", default="output/eval_results")
    return parser.parse_args()


def resolve_decoders(requested):
    if requested == ["all"] or requested == ["ALL"]:
        return ALL_DECODERS
    unknown = [d for d in requested if d not in ALL_DECODERS]
    if unknown:
        raise ValueError(f"Unknown decoders: {unknown}")
    return requested


def resolve_checkpoint(decoder_name: str) -> str:
    matches = sorted(glob.glob(f"output/decoder_compare/{decoder_name}_*/best_model.pth"))
    if not matches:
        raise FileNotFoundError(f"No best_model.pth found for decoder={decoder_name}")
    return matches[-1]


def load_model(decoder_name: str, checkpoint: str, device: torch.device):
    model = build_encoder_with_decoder(decoder_name, num_classes=4)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt)
    model.load_state_dict(sd, strict=False)
    model = model.to(device)
    model.eval()
    return model


def evaluate_one(decoder_name: str, args, device: torch.device):
    checkpoint = resolve_checkpoint(decoder_name)
    output_dir = Path(args.output_root) / f"decoder_{decoder_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(decoder_name, checkpoint, device)
    basenames = load_split(args.split_dir, args.split)
    img_dir = Path(args.data_root) / "ori" / "MoS2"
    mask_dir = Path(args.data_root) / "mask"
    metrics = SegmentationMetrics(4)
    per_image = []
    total_time = 0.0

    model_display = "Ours (DW-MFF + Boundary)" if decoder_name == "ours" else DECODER_NAMES[decoder_name]
    print(f"\n=== {decoder_name} ===")
    print(f"Checkpoint: {checkpoint}")
    print(f"Output: {output_dir}")

    for i, bn in enumerate(basenames, start=1):
        img = Image.open(img_dir / f"{bn}.jpg").convert("RGB")
        gt = np.array(Image.open(mask_dir / f"{bn}.png"))
        img_tensor = TF.normalize(TF.to_tensor(img), MEAN, STD)

        t0 = time.time()
        pred = sliding_window_predict(
            model, img_tensor,
            crop_size=args.crop_size,
            stride=args.stride,
            device=device,
            is_smp=False,
        )
        elapsed = time.time() - t0
        total_time += elapsed

        Image.fromarray(pred.astype(np.uint8)).save(output_dir / f"{bn}_pred.png")

        img_metrics = SegmentationMetrics(4)
        img_metrics.update(pred, gt)
        img_results = img_metrics.get_results()
        per_image.append((bn, img_results))
        metrics.update(pred, gt)

        print(f"  [{i}/{len(basenames)}] {bn}: mIoU={img_results['mIoU']:.4f} ({elapsed:.2f}s)")

    results = metrics.get_results()
    results_path = output_dir / f"{args.split}_metrics.txt"
    with open(results_path, "w") as f:
        f.write(f"{model_display} | {args.split} set | {len(basenames)} images\n")
        f.write(f"Checkpoint: {checkpoint}\n")
        f.write(f"Sliding window: crop={args.crop_size}, stride={args.stride}\n\n")
        f.write(f"mIoU:       {results['mIoU']:.4f}\n")
        f.write(f"Pixel Acc:  {results['pixel_acc']:.4f}\n")
        f.write(f"Mean F1:    {results['mean_f1']:.4f}\n\n")
        for c in range(4):
            f.write(
                f"{CLASSES[c]:12s}  IoU={results['per_class_iou'][c]:.4f}  "
                f"F1={results['f1'][c]:.4f}  Acc={results['class_acc'][c]:.4f}\n"
            )
        f.write(f"\nAvg time:   {total_time/len(basenames):.3f}s/image\n")
        f.write("\nPer-image results:\n")
        for bn, r in per_image:
            f.write(f"  {bn}: mIoU={r['mIoU']:.4f}\n")

    cm_path = output_dir / f"{args.split}_confusion_matrix.png"
    plot_confusion_matrix(metrics.confusion_matrix, CLASS_LABELS_SHORT, str(cm_path))
    print(f"Saved: {results_path}")
    print(f"Saved: {cm_path}")


def main():
    args = parse_args()
    decoders = resolve_decoders(args.decoders)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    for decoder_name in decoders:
        evaluate_one(decoder_name, args, device)


if __name__ == "__main__":
    main()
