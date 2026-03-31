"""
SCI-style params-vs-mIoU scatter for main-model comparison.

This figure summarizes:
1. Pretrained heavy baselines
2. Pretrained lightweight baselines
3. Scratch baselines
4. Ours (RepELA-Net, representative single run = seed_123)

Output:
    output/paper_figures/scatter_params_vs_miou.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT = Path("/root/autodl-tmp/PhysicalNet/output/paper_figures")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.linewidth": 0.8,
    "axes.labelsize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
})


GROUPS = {
    "Pretrained (Heavy)": {
        "color": "#8c84c6",
        "marker": "s",
        "points": [
            ("UNet-R18", 14.33, 87.39),
            ("DLV3+-R18", 12.33, 83.32),
            ("PSPNet-R18", 11.34, 86.57),
        ],
    },
    "Pretrained (Light)": {
        "color": "#4db6ac",
        "marker": "^",
        "points": [
            ("FPN-MV2", 4.22, 90.95),
            ("FPN-MNV3", 2.72, 89.53),
            ("DLV3+-EffB0", 4.91, 88.27),
            ("DLV3+-MV2", 4.38, 88.32),
            ("UNet-MNV3", 3.59, 88.34),
        ],
    },
    "Scratch": {
        "color": "#f0a35a",
        "marker": "D",
        "points": [
            ("DLV3+-EffB0*", 4.91, 86.89),
            ("UNet-MNV3*", 3.59, 85.87),
            ("FPN-MNV3*", 2.72, 83.79),
        ],
    },
    "Ours": {
        "color": "#d95f8d",
        "marker": "*",
        "points": [
            ("RepELA-Net", 2.12, 84.64),
        ],
    },
}


ANNOT_OFFSETS = {
    "UNet-R18": (-0.42, 0.18),
    "DLV3+-R18": (0.18, 0.22),
    "PSPNet-R18": (0.16, -0.42),
    "FPN-MV2": (0.16, 0.28),
    "FPN-MNV3": (0.14, 0.34),
    "DLV3+-EffB0": (0.22, 0.54),
    "DLV3+-MV2": (0.34, -0.26),
    "UNet-MNV3": (-0.36, -0.10),
    "DLV3+-EffB0*": (0.16, -0.18),
    "UNet-MNV3*": (0.16, -0.26),
    "FPN-MNV3*": (0.16, -0.44),
    "RepELA-Net": (0.15, 0.28),
}


def main():
    fig, ax = plt.subplots(figsize=(8.0, 5.1))

    for group_name, cfg in GROUPS.items():
        xs = [p[1] for p in cfg["points"]]
        ys = [p[2] for p in cfg["points"]]
        sizes = [150 if group_name != "Ours" else 320 for _ in cfg["points"]]
        ax.scatter(
            xs, ys,
            s=sizes,
            marker=cfg["marker"],
            color=cfg["color"],
            edgecolor="#2f2f2f",
            linewidth=0.8,
            alpha=0.95 if group_name == "Ours" else 0.90,
            label=group_name if group_name != "Ours" else "Ours (Scratch)",
            zorder=3 if group_name == "Ours" else 2,
        )

        for label, x, y in cfg["points"]:
            dx, dy = ANNOT_OFFSETS.get(label, (0.12, 0.20))
            ha = "left" if dx >= 0 else "right"
            va = "bottom" if dy >= 0 else "top"
            ax.text(
                x + dx, y + dy, label,
                fontsize=10.0 if label != "RepELA-Net" else 12,
                color=cfg["color"],
                fontweight="bold" if label == "RepELA-Net" else None,
                ha=ha, va=va,
            )

    ax.set_xlabel("Parameters (M)", fontweight="bold")
    ax.set_ylabel("Test mIoU (%)", fontweight="bold")
    ax.set_xlim(0.5, 15.5)
    ax.set_ylim(83.0, 92.2)
    ax.set_xticks([2, 4, 6, 8, 10, 12, 14])
    ax.set_yticks([84, 86, 88, 90, 92])
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.18, color="#888888")
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.8)

    legend = ax.legend(
        loc="upper right",
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        edgecolor="#999999",
        handletextpad=0.5,
        borderpad=0.4,
    )
    legend.get_frame().set_linewidth(0.8)

    save_path = OUT / "scatter_params_vs_miou.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
