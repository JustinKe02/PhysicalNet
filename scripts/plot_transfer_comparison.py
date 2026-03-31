"""
SCI-style grouped transfer strategy comparison chart.

Output:
    output/paper_figures/bar_transfer_comparison.png
    output/paper_figures/bar_transfer_comparison_clean.png
"""
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path("/root/autodl-tmp/PhysicalNet")
OUT = PROJECT_ROOT / "output" / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.linewidth": 0.8,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
})


RUNS = [
    {
        "target": "WS2",
        "items": [
            ("Scratch", PROJECT_ROOT / "output/finetune_ws2_scratch/test_results.txt"),
            ("FT+keep", PROJECT_ROOT / "output/finetune_ws2_r2_keephead/test_results.txt"),
            ("FT+reset", PROJECT_ROOT / "output/finetune_ws2_r2_resethead/test_results.txt"),
        ],
    },
    {
        "target": "Graphene",
        "items": [
            ("Scratch", PROJECT_ROOT / "output/finetune_graphene_scratch/results.txt"),
            ("FT+reset", PROJECT_ROOT / "output/finetune_graphene_r3_fullLR_resethead/results.txt"),
            ("FT+partial", PROJECT_ROOT / "output/finetune_graphene_r3_partial/results.txt"),
        ],
    },
    {
        "target": "MoS2",
        "items": [
            ("Scratch", PROJECT_ROOT / "output/finetune_mos2v2_scratch/test_results.txt"),
            ("FT+keep", PROJECT_ROOT / "output/finetune_mos2v2_ft_keephead/test_results.txt"),
            ("FT+reset", PROJECT_ROOT / "output/finetune_mos2v2_ft_resethead/test_results.txt"),
        ],
    },
]

COLORS = {
    "Scratch": "#8882c1",
    "FT+keep": "#b7c67b",
    "FT+reset": "#3daa8a",
    "FT+partial": "#ef9f6e",
}


def parse_miou(path: Path) -> float:
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("mIoU:"):
            return float(line.split()[-1]) * 100
        if line.startswith("Test mIoU:"):
            return float(line.split()[-1]) * 100
    raise ValueError(f"mIoU not found in {path}")


def draw_chart(show_value_labels: bool, show_strategy_labels: bool, out_name: str):
    fig, ax = plt.subplots(figsize=(9.4, 4.6))

    group_centers = np.array([0.0, 1.6, 3.2])
    bar_width = 0.34
    offsets_3 = np.array([-bar_width, 0.0, bar_width])

    for center, run in zip(group_centers, RUNS):
        labels = [name for name, _ in run["items"]]
        values = [parse_miou(path) for _, path in run["items"]]
        colors = [COLORS[name] for name in labels]
        positions = center + offsets_3[: len(labels)]

        bars = ax.bar(
            positions,
            values,
            width=bar_width * 0.98,
            color=colors,
            edgecolor="#4a4a4a",
            linewidth=0.45,
            alpha=0.95,
        )

        if show_value_labels:
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.6,
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9.5,
                    fontweight="bold",
                    color="#222222",
                )

        if show_strategy_labels:
            for pos, label in zip(positions, labels):
                ax.text(
                    pos,
                    51.2,
                    label,
                    ha="center",
                    va="bottom",
                    rotation=18,
                    fontsize=9.2,
                    color="#4f4f4f",
                    style="italic" if label != "Scratch" else "normal",
                )

    ax.set_ylabel("mIoU (%)", fontweight="bold")
    ax.set_xlabel("Target Domain", fontweight="bold")
    ax.set_xticks(group_centers)
    ax.set_xticklabels([run["target"] for run in RUNS], fontsize=12)
    ax.set_ylim(50, 100)
    ax.set_yticks(np.arange(50, 101, 10))
    ax.grid(axis="y", alpha=0.18, linewidth=0.5, color="#999999")
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#333333")

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[name], edgecolor="#4a4a4a", linewidth=0.45)
        for name in ["Scratch", "FT+keep", "FT+reset", "FT+partial"]
    ]
    ax.legend(
        handles,
        ["Scratch", "FT+keep", "FT+reset", "FT+partial"],
        loc="upper right",
        frameon=True,
        edgecolor="#888888",
        fancybox=False,
        ncol=4,
        columnspacing=1.0,
        handlelength=1.2,
    )

    fig.savefig(OUT / out_name)
    plt.close(fig)
    print(f"Saved: {OUT / out_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true", help="hide per-bar value and strategy annotations")
    parser.add_argument(
        "--numbers-only",
        action="store_true",
        help="keep numeric labels but hide strategy text under bars",
    )
    args = parser.parse_args()

    if args.numbers_only:
        out_name = "bar_transfer_comparison_clean.png"
        draw_chart(show_value_labels=True, show_strategy_labels=False, out_name=out_name)
    elif args.clean:
        out_name = "bar_transfer_comparison_clean.png"
        draw_chart(show_value_labels=False, show_strategy_labels=False, out_name=out_name)
    else:
        out_name = "bar_transfer_comparison.png"
        draw_chart(show_value_labels=True, show_strategy_labels=True, out_name=out_name)


if __name__ == "__main__":
    main()
