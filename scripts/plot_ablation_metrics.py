"""
SCI-style ablation metrics figure using the final official test results.

Output:
    output/paper_figures/ablation_metrics_seed123.png
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.chdir("/root/autodl-tmp/PhysicalNet")

OUT = Path("output/paper_figures")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.linewidth": 0.8,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
})


RUNS = [
    ("Ours", "output/eval_results/seed_123/test_metrics.txt"),
    ("w/o RepConv", "output/eval_results/ablation_no_rep/test_metrics.txt"),
    ("w/o ELA", "output/eval_results/ablation_no_ela/test_metrics.txt"),
    ("w/o DW-MFF", "output/eval_results/ablation_no_dwmff/test_metrics.txt"),
    ("w/o Boundary", "output/eval_results/ablation_no_boundary/test_metrics.txt"),
]

METRIC_ORDER = ["mIoU", "monolayer", "fewlayer", "multilayer"]
METRIC_LABELS = ["mIoU", "1L IoU", "FL IoU", "ML IoU"]
# Slightly deeper SCI-style palette.
COLORS = {
    "Ours": "#3E6FA3",
    "w/o RepConv": "#7FA9D1",
    "w/o ELA": "#D6885C",
    "w/o DW-MFF": "#6FAA80",
    "w/o Boundary": "#A88DC5",
}


def parse_metrics(path: str):
    vals = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith("mIoU:"):
            vals["mIoU"] = float(line.split()[-1])
        elif "IoU=" in line and line.startswith(("background", "monolayer", "fewlayer", "multilayer")):
            cls = line.split()[0]
            for token in line.split():
                if token.startswith("IoU="):
                    vals[cls] = float(token.split("=")[1])
                    break
    return vals


def main():
    series = {name: parse_metrics(path) for name, path in RUNS}

    fig, ax = plt.subplots(figsize=(7.6, 4.1))
    x = np.arange(len(METRIC_ORDER))
    width = 0.16

    for i, (name, _) in enumerate(RUNS):
        y = [series[name][m] * 100 for m in METRIC_ORDER]
        offset = (i - 2) * width
        ax.bar(
            x + offset, y, width=width,
            color=COLORS[name],
            edgecolor="#4a4a4a",
            linewidth=0.4,
            label=name,
            alpha=0.94 if name == "Ours" else 0.88,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS)
    ax.set_ylabel("IoU / mIoU (%)", fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", alpha=0.18, linewidth=0.45, color="#a3a3a3")
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#333333")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.24),
        ncol=3,
        frameon=False,
        columnspacing=1.15,
        handlelength=1.45,
    )

    save_path = OUT / "ablation_metrics_seed123.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")
    for name, _ in RUNS:
        print(name, {k: round(series[name][k] * 100, 2) for k in METRIC_ORDER})


if __name__ == "__main__":
    main()
