"""
SCI-style decoder metrics figures using official test results.

Outputs:
    output/paper_figures/decoder_metrics_test.png
    output/paper_figures/decoder_params_vs_miou.png
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path("/root/autodl-tmp/PhysicalNet")
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from tools.train_decoder_compare import build_encoder_with_decoder

OUT = Path("output/paper_figures")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.linewidth": 0.8,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
})

RUNS = [
    ("UNet", "output/eval_results/decoder_unet/test_metrics.txt"),
    ("FPN", "output/eval_results/decoder_fpn/test_metrics.txt"),
    ("ASPP", "output/eval_results/decoder_aspp/test_metrics.txt"),
    ("SegFormer", "output/eval_results/decoder_segformer/test_metrics.txt"),
    ("PPM", "output/eval_results/decoder_ppm/test_metrics.txt"),
    ("Hamburger", "output/eval_results/decoder_hamburger/test_metrics.txt"),
    ("Ours", "output/eval_results/decoder_ours/test_metrics.txt"),
]

METRIC_ORDER = ["mIoU", "monolayer", "fewlayer", "multilayer"]
METRIC_LABELS = ["mIoU", "1L IoU", "FL IoU", "ML IoU"]

COLORS = {
    "UNet": "#cad7e5",
    "FPN": "#8fb9e3",
    "ASPP": "#f2d0a7",
    "SegFormer": "#b8d8ba",
    "PPM": "#d7c6e8",
    "Hamburger": "#f4b6c2",
    "Ours": "#9fc5b8",
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


def get_decoder_params_m(name: str) -> float:
    model = build_encoder_with_decoder(name.lower() if name != "SegFormer" else "segformer", num_classes=4)
    return sum(p.numel() for p in model.decoder.parameters()) / 1e6


def plot_efficiency_scatter(series):
    fig, ax = plt.subplots(figsize=(6.2, 4.4))

    label_to_decoder = {
        "UNet": "unet",
        "FPN": "fpn",
        "ASPP": "aspp",
        "SegFormer": "segformer",
        "PPM": "ppm",
        "Hamburger": "hamburger",
        "Ours": "ours",
    }

    for name, _ in RUNS:
        dec_name = label_to_decoder[name]
        x = get_decoder_params_m(dec_name)
        y = series[name]["mIoU"] * 100
        marker = "*" if name == "Ours" else "o"
        size = 180 if name == "Ours" else 90
        edge = "#2f2f2f"
        ax.scatter(
            x, y,
            s=size,
            marker=marker,
            color=COLORS[name],
            edgecolor=edge,
            linewidth=0.8,
            alpha=0.96,
            zorder=3,
        )
        dx = 0.03 if name != "Ours" else 0.05
        dy = 0.12 if name != "Ours" else 0.18
        ax.text(x + dx, y + dy, name, fontsize=10, color="#333333")

    ax.set_xlabel("Decoder Params (M)", fontweight="bold")
    ax.set_ylabel("Test mIoU (%)", fontweight="bold")
    ax.grid(True, alpha=0.20, linewidth=0.55, color="#999999")
    ax.set_axisbelow(True)
    ax.set_xlim(0, 1.55)
    ax.set_ylim(85, 93)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#333333")

    save_path = OUT / "decoder_params_vs_miou.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    series = {name: parse_metrics(path) for name, path in RUNS}

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    x = np.arange(len(METRIC_ORDER))
    width = 0.11

    for i, (name, _) in enumerate(RUNS):
        y = [series[name][m] * 100 for m in METRIC_ORDER]
        offset = (i - (len(RUNS) - 1) / 2) * width
        bars = ax.bar(
            x + offset, y, width=width,
            color=COLORS[name],
            edgecolor="#4a4a4a",
            linewidth=0.42,
            label=name,
            alpha=0.94 if name == "Ours" else 0.90,
        )
        if name == "Ours":
            for b in bars:
                b.set_hatch("//")

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS)
    ax.set_ylabel("IoU / mIoU (%)", fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", alpha=0.18, linewidth=0.5, color="#999999")
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#333333")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=4,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.4,
    )

    save_path = OUT / "decoder_metrics_test.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")
    plot_efficiency_scatter(series)
    for name, _ in RUNS:
        print(name, {k: round(series[name][k] * 100, 2) for k in METRIC_ORDER})


if __name__ == "__main__":
    main()
