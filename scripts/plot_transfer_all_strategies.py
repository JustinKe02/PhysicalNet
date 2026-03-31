"""
Compare all tested transfer strategies by target domain.

Output:
    output/paper_figures/bar_transfer_all_strategies.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/root/autodl-tmp/PhysicalNet")
OUT = ROOT / "output" / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
})


PANELS = [
    {
        "title": "WS2 (test)",
        "items": [
            ("Scratch", ROOT / "output/finetune_ws2_scratch/test_results.txt"),
            ("FT-full", ROOT / "output/finetune_ws2_finetune/test_results.txt"),
            ("FT-reset", ROOT / "output/finetune_ws2_r2_resethead/test_results.txt"),
            ("FT-keep", ROOT / "output/finetune_ws2_r2_keephead/test_results.txt"),
        ],
    },
    {
        "title": "Graphene (val)",
        "items": [
            ("Scratch", ROOT / "output/finetune_graphene_scratch/results.txt"),
            ("FT-full", ROOT / "output/finetune_graphene_finetune/results.txt"),
            ("FT-nomap", ROOT / "output/finetune_graphene_r2_nomap/results.txt"),
            ("FT-mapped", ROOT / "output/finetune_graphene_r2_mapped/results.txt"),
            ("FT-fullLR", ROOT / "output/finetune_graphene_r3_fullLR/results.txt"),
            ("FT-reset", ROOT / "output/finetune_graphene_r3_fullLR_resethead/results.txt"),
            ("FT-partial", ROOT / "output/finetune_graphene_r3_partial/results.txt"),
        ],
    },
    {
        "title": "MoS2v2 (test)",
        "items": [
            ("Scratch", ROOT / "output/finetune_mos2v2_scratch/test_results.txt"),
            ("FT-full", ROOT / "output/finetune_mos2v2_ft_full/test_results.txt"),
            ("FT-reset", ROOT / "output/finetune_mos2v2_ft_resethead/test_results.txt"),
            ("FT-keep", ROOT / "output/finetune_mos2v2_ft_keephead/test_results.txt"),
        ],
    },
]


COLORS = {
    "Scratch": "#8882c1",
    "FT-full": "#7eb6ad",
    "FT-nomap": "#e7c98b",
    "FT-mapped": "#d6a3ba",
    "FT-fullLR": "#7ebcc9",
    "FT-reset": "#3daa8a",
    "FT-keep": "#b7c67b",
    "FT-partial": "#ef9f6e",
}


def parse_miou(path: Path) -> float:
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("mIoU:"):
            return float(line.split()[-1]) * 100
        if line.startswith("Test mIoU:"):
            return float(line.split()[-1]) * 100
    raise ValueError(f"mIoU not found in {path}")


def main():
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.2), sharey=True)

    for ax, panel in zip(axes, PANELS):
        labels = [name for name, _ in panel["items"]]
        values = [parse_miou(path) for _, path in panel["items"]]
        x = np.arange(len(labels))
        colors = [COLORS[name] for name in labels]

        bars = ax.bar(
            x, values,
            color=colors,
            edgecolor="#4a4a4a",
            linewidth=0.45,
            alpha=0.95,
        )
        ax.set_title(panel["title"], fontweight="bold", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=28, ha="right")
        ax.grid(axis="y", alpha=0.18, linewidth=0.5, color="#999999")
        ax.set_axisbelow(True)
        ax.set_ylim(35, 100)

        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#333333")

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.7,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="#222222",
            )

    axes[0].set_ylabel("mIoU (%)", fontweight="bold")

    handles = []
    seen = set()
    for name, color in COLORS.items():
        if name in seen:
            continue
        patch = plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="#4a4a4a", linewidth=0.45)
        handles.append((patch, name))
        seen.add(name)
    fig.legend(
        [h for h, _ in handles], [n for _, n in handles],
        loc="upper center", bbox_to_anchor=(0.5, 1.05),
        ncol=8, frameon=False, columnspacing=0.9, handlelength=1.4,
    )

    fig.savefig(OUT / "bar_transfer_all_strategies.png")
    plt.close(fig)
    print(f"Saved: {OUT / 'bar_transfer_all_strategies.png'}")


if __name__ == "__main__":
    main()
