"""
Build WS2 official-test visualizations for transfer experiments.

Outputs:
  - output/transfer_vis/WS2_confusion_matrices.png
  - output/transfer_vis/WS2_inference_grid.png
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


ROOT = Path("/root/autodl-tmp/PhysicalNet")
OUT_DIR = ROOT / "output" / "transfer_vis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WS2_COLORS = {
    0: (210, 210, 210),
    1: (253, 174, 97),
    2: (116, 173, 209),
    3: (69, 117, 180),
}

EXPERIMENTS = [
    ("Scratch", ROOT / "output/finetune_ws2_scratch"),
    ("FT+reset_head", ROOT / "output/finetune_ws2_r2_resethead"),
    ("FT+keep_head", ROOT / "output/finetune_ws2_r2_keephead"),
]

SAMPLES = [
    "2025-02-12_16_05_07.267_7",
    "2025-02-12_16_16_37.218_6",
    "2025-02-12_16_07_38.530_10",
    "2025-02-12_16_17_01.449_1",
    "2025-02-12_16_03_15.030_1",
]


def colorize(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in WS2_COLORS.items():
        rgb[mask == cls] = color
    return rgb


def add_title_band(image: Image.Image, title: str, band_height: int = 34):
    canvas = Image.new("RGB", (image.width, image.height + band_height), "white")
    canvas.paste(image, (0, band_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), title, fill="black")
    return canvas


def build_confusion_grid():
    images = []
    for label, run_dir in EXPERIMENTS:
        img_path = run_dir / "test_confusion_matrix.png"
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

    out_path = OUT_DIR / "WS2_confusion_matrices.png"
    canvas.save(out_path)
    return out_path


def load_panel_images(sample_id: str):
    img_path = ROOT / "other data" / "WS2_data" / "img_dir" / "test" / f"{sample_id}.jpg"
    gt_path = ROOT / "other data" / "WS2_data" / "ann_dir" / "test" / f"{sample_id}.png"

    original = Image.open(img_path).convert("RGB")
    gt = Image.fromarray(colorize(np.array(Image.open(gt_path))))

    panels = [original, gt]
    for _, run_dir in EXPERIMENTS:
        pred_path = run_dir / "test_preds" / f"{sample_id}_pred.png"
        pred = Image.fromarray(colorize(np.array(Image.open(pred_path))))
        panels.append(pred)
    return panels


def build_inference_grid():
    col_titles = ["Original", "GT"] + [label for label, _ in EXPERIMENTS]
    rows = []

    target_h = 220
    gap = 10
    band_h = 32
    left_margin = 0

    for sample_id in SAMPLES:
        panels = load_panel_images(sample_id)
        resized = []
        for img in panels:
            w, h = img.size
            new_w = int(w * target_h / h)
            resized.append(img.resize((new_w, target_h), Image.Resampling.BILINEAR))
        rows.append(resized)

    col_widths = [max(row[c].width for row in rows) for c in range(len(col_titles))]
    total_width = sum(col_widths) + gap * (len(col_titles) - 1) + left_margin
    total_height = band_h + len(rows) * target_h + gap * (len(rows) - 1)
    canvas = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(canvas)

    x = left_margin
    for title, col_w in zip(col_titles, col_widths):
        draw.text((x + 8, 8), title, fill="black")
        x += col_w + gap

    y = band_h
    for row in rows:
        x = left_margin
        for img, col_w in zip(row, col_widths):
            x_off = x + (col_w - img.width) // 2
            canvas.paste(img, (x_off, y))
            x += col_w + gap
        y += target_h + gap

    out_path = OUT_DIR / "WS2_inference_grid.png"
    canvas.save(out_path)
    return out_path


def main():
    generated = [build_confusion_grid(), build_inference_grid()]
    for path in generated:
        if path:
            print(f"Saved: {path}")


if __name__ == "__main__":
    main()
