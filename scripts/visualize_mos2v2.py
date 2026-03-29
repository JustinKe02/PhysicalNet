"""
Build consolidated visualizations for the other_datav2 control-transfer experiments.
Outputs:
  - output/transfer_vis/MOS2V2_training_curves.png
  - output/transfer_vis/MOS2V2_confusion_matrices.png
  - output/transfer_vis/MOS2V2_inference_grid.png
"""
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageDraw


ROOT = Path("/root/autodl-tmp/PhysicalNet")
OUT_DIR = ROOT / "output" / "transfer_vis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = [
    ("mos2v2_scratch", "Scratch", "#1E88E5"),
    ("mos2v2_ft_resethead", "FT+reset_head", "#D81B60"),
    ("mos2v2_ft_keephead", "FT+keep_head", "#FB8C00"),
]


def parse_log(log_path: Path):
    epochs = []
    val_mious = []
    cur_epoch = None
    with log_path.open() as f:
        for line in f:
            m = re.search(r"Epoch \[(\d+)/\d+\]", line)
            if m:
                cur_epoch = int(m.group(1))
            m = re.search(r"Val mIoU:\s*([\d.]+)", line)
            if m and cur_epoch is not None:
                epochs.append(cur_epoch)
                val_mious.append(float(m.group(1)))
    return epochs, val_mious


def plot_training_curves():
    fig, ax = plt.subplots(figsize=(8.5, 5))

    for exp_name, label, color in EXPERIMENTS:
        log_path = ROOT / "output" / f"finetune_{exp_name}" / "finetune.log"
        if not log_path.exists():
            continue
        epochs, val_mious = parse_log(log_path)
        if epochs:
            ax.plot(epochs, val_mious, label=label, color=color, linewidth=2)

    ax.set_title("MoS2v2 Transfer: Validation Curves", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val mIoU")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend()

    out_path = OUT_DIR / "MOS2V2_training_curves.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def add_title_band(image: Image.Image, title: str, band_height: int = 34):
    titled = Image.new("RGB", (image.width, image.height + band_height), "white")
    titled.paste(image, (0, band_height))
    draw = ImageDraw.Draw(titled)
    draw.text((10, 8), title, fill="black")
    return titled


def build_confusion_grid():
    images = []
    for exp_name, label, _ in EXPERIMENTS:
        img_path = ROOT / "output" / f"finetune_{exp_name}" / "confusion_matrix_final.png"
        if not img_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")
        img = ImageOps.contain(img, (420, 420))
        images.append(add_title_band(img, label))

    if not images:
        return None

    gap = 16
    width = sum(img.width for img in images) + gap * (len(images) - 1)
    height = max(img.height for img in images)
    canvas = Image.new("RGB", (width, height), "white")

    x = 0
    for img in images:
        canvas.paste(img, (x, 0))
        x += img.width + gap

    out_path = OUT_DIR / "MOS2V2_confusion_matrices.png"
    canvas.save(out_path)
    return out_path


def copy_inference_grid():
    src = ROOT / "output" / "individual_preds" / "mos2v2_test" / "comparison_main.png"
    if not src.exists():
        return None
    dst = OUT_DIR / "MOS2V2_inference_grid.png"
    Image.open(src).save(dst)
    return dst


def main():
    generated = [
        plot_training_curves(),
        build_confusion_grid(),
        copy_inference_grid(),
    ]
    for path in generated:
        if path:
            print(f"Saved: {path}")


if __name__ == "__main__":
    main()
