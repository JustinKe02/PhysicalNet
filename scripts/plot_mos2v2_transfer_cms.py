"""
Replot MoS2v2 transfer confusion matrices in the same SCI style as gen_paper_figures.py.

Outputs:
  - output/paper_figures/cm_transfer_mos2v2_scratch.png
  - output/paper_figures/cm_transfer_mos2v2_ft_full.png
  - output/paper_figures/cm_transfer_mos2v2_ft_resethead.png
  - output/paper_figures/cm_transfer_mos2v2_ft_keephead.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image


ROOT = Path("/root/autodl-tmp/PhysicalNet")
OUT = ROOT / "output" / "paper_figures"
GT_DIR = ROOT / "other_datav2_prepared" / "ann_dir" / "test"
TEST_NAMES = [p.stem for p in sorted(GT_DIR.glob("*.png"))]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.linewidth": 0.8,
    "axes.labelsize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

JOBS = [
    ("cm_transfer_mos2v2_scratch", ROOT / "output" / "finetune_mos2v2_scratch" / "test_preds"),
    ("cm_transfer_mos2v2_ft_full", ROOT / "output" / "finetune_mos2v2_ft_full" / "test_preds"),
    ("cm_transfer_mos2v2_ft_resethead", ROOT / "output" / "finetune_mos2v2_ft_resethead" / "test_preds"),
    ("cm_transfer_mos2v2_ft_keephead", ROOT / "output" / "finetune_mos2v2_ft_keephead" / "test_preds"),
]


def cm_from_saved_preds(pred_dir: Path, gt_dir: Path, test_names, nc=4):
    cm = np.zeros((nc, nc), dtype=np.float64)
    found = 0
    for name in test_names:
        pred_path = pred_dir / f"{name}_pred.png"
        gt_path = gt_dir / f"{name}.png"
        if not pred_path.exists() or not gt_path.exists():
            continue
        pred = np.array(Image.open(pred_path))
        gt = np.array(Image.open(gt_path))
        for t in range(nc):
            for p in range(nc):
                cm[t, p] += ((gt == t) & (pred == p)).sum()
        found += 1
    if found == 0:
        raise RuntimeError(f"no pred masks found in {pred_dir}")
    row_sums = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, row_sums, where=row_sums > 0, out=np.zeros_like(cm))


def plot_confusion_matrix(cm, save_path: Path):
    fig = plt.figure(figsize=(4.5, 4.0))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 0.035], wspace=0.05)
    ax = fig.add_subplot(gs[0])
    cax = fig.add_subplot(gs[1])

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1, aspect="equal")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > 0.45 else "#1a1a1a"
            ax.text(
                j, i, f"{val:.3f}",
                ha="center", va="center",
                fontsize=10, fontweight="bold",
                color=color, fontfamily="serif",
            )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#333333")

    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    cbar.ax.tick_params(labelsize=9, length=3, width=0.5)
    cbar.outline.set_linewidth(0.5)

    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, pred_dir in JOBS:
        cm = cm_from_saved_preds(pred_dir, GT_DIR, TEST_NAMES, nc=4)
        plot_confusion_matrix(cm, OUT / f"{name}.png")


if __name__ == "__main__":
    main()
