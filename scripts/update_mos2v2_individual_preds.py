"""
Supplement individual MoS2v2 test visualizations with FT-full predictions.

Outputs:
  - output/individual_preds/mos2v2_test/{id}_FT_full.png
"""
from pathlib import Path

from PIL import Image


ROOT = Path("/root/autodl-tmp/PhysicalNet")
OUT_DIR = ROOT / "output" / "individual_preds" / "mos2v2_test"
PRED_DIR = ROOT / "output" / "finetune_mos2v2_ft_full" / "test_preds"

MASK_COLORS = {
    0: (210, 210, 210),  # BG
    1: (253, 174, 97),   # 1L
    2: (116, 173, 209),  # FL
    3: (69, 117, 180),   # ML
}


def colorize_mask(mask_path: Path) -> Image.Image:
    mask = Image.open(mask_path)
    if mask.mode != "L":
        mask = mask.convert("L")
    colored = Image.new("RGB", mask.size)
    src = mask.load()
    dst = colored.load()
    for y in range(mask.height):
        for x in range(mask.width):
            dst[x, y] = MASK_COLORS.get(src[x, y], (0, 0, 0))
    return colored


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    for pred_path in sorted(PRED_DIR.glob("*_pred.png")):
        sample_id = pred_path.stem.replace("_pred", "")
        out_path = OUT_DIR / f"{sample_id}_FT_full.png"
        colorize_mask(pred_path).save(out_path)
        count += 1
        print(f"Saved: {out_path}")
    print(f"Generated {count} FT-full visualizations.")


if __name__ == "__main__":
    main()
