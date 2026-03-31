"""
Build consolidated visualizations for the other_datav2 control-transfer experiments.
Outputs:
  - output/transfer_vis/MOS2V2_training_curves.png
  - output/transfer_vis/MOS2V2_confusion_matrices.png
  - output/transfer_vis/MOS2V2_inference_grid.png
"""
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
    ("mos2v2_ft_full", "FT+full", "#6D4C41"),
    ("mos2v2_ft_resethead", "FT+reset_head", "#D81B60"),
    ("mos2v2_ft_keephead", "FT+keep_head", "#FB8C00"),
]

TEST_ROOT = ROOT / "other_datav2_prepared"
SAMPLE_IDS = ["00001", "00021", "00014"]
MASK_COLORS = {
    0: (210, 210, 210),  # BG
    1: (253, 174, 97),   # 1L
    2: (116, 173, 209),  # FL
    3: (69, 117, 180),   # ML
}


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


def add_frame(image: Image.Image, border: int = 2, color=(235, 235, 235)):
    return ImageOps.expand(image, border=border, fill=color)


def colorize_mask(mask_path: Path) -> Image.Image:
    mask = Image.open(mask_path)
    if mask.mode != "L":
        mask = mask.convert("L")
    colored = Image.new("RGB", mask.size)
    pixels = mask.load()
    out_pixels = colored.load()
    for y in range(mask.height):
        for x in range(mask.width):
            out_pixels[x, y] = MASK_COLORS.get(pixels[x, y], (0, 0, 0))
    return colored


def build_confusion_grid():
    images = []
    for exp_name, label, _ in EXPERIMENTS:
        test_img = ROOT / "output" / f"finetune_{exp_name}" / "test_confusion_matrix.png"
        final_img = ROOT / "output" / f"finetune_{exp_name}" / "confusion_matrix_final.png"
        img_path = test_img if test_img.exists() else final_img
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


def build_inference_grid():
    columns = [
        ("Original", None),
        ("GT", None),
        ("Scratch", "mos2v2_scratch"),
        ("FT+full", "mos2v2_ft_full"),
        ("FT+reset_head", "mos2v2_ft_resethead"),
        ("FT+keep_head", "mos2v2_ft_keephead"),
    ]

    sample_rows = []
    target_size = (260, 195)
    header_height = 36
    row_gap = 18
    col_gap = 14

    for sample_id in SAMPLE_IDS:
        row_images = []
        original = Image.open(TEST_ROOT / "img_dir" / "test" / f"{sample_id}.png").convert("RGB")
        row_images.append(add_frame(ImageOps.contain(original, target_size)))

        gt = colorize_mask(TEST_ROOT / "ann_dir" / "test" / f"{sample_id}.png")
        row_images.append(add_frame(ImageOps.contain(gt, target_size)))

        for _, exp_name in columns[2:]:
            pred = colorize_mask(ROOT / "output" / f"finetune_{exp_name}" / "test_preds" / f"{sample_id}_pred.png")
            row_images.append(add_frame(ImageOps.contain(pred, target_size)))
        sample_rows.append(row_images)

    cell_w = max(img.width for row in sample_rows for img in row)
    cell_h = max(img.height for row in sample_rows for img in row)
    width = len(columns) * cell_w + (len(columns) - 1) * col_gap
    height = header_height + len(sample_rows) * cell_h + (len(sample_rows) - 1) * row_gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    for idx, (title, _) in enumerate(columns):
        x = idx * (cell_w + col_gap)
        draw.text((x + 10, 8), title, fill="black")

    for row_idx, row in enumerate(sample_rows):
        y = header_height + row_idx * (cell_h + row_gap)
        for col_idx, img in enumerate(row):
            x = col_idx * (cell_w + col_gap)
            paste_x = x + (cell_w - img.width) // 2
            paste_y = y + (cell_h - img.height) // 2
            canvas.paste(img, (paste_x, paste_y))

    out_path = OUT_DIR / "MOS2V2_inference_grid.png"
    canvas.save(out_path)
    return out_path


def main():
    generated = [
        plot_training_curves(),
        build_confusion_grid(),
        build_inference_grid(),
    ]
    for path in generated:
        if path:
            print(f"Saved: {path}")


if __name__ == "__main__":
    main()
