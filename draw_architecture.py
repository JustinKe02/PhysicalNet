import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import os

# Create figure
fig = plt.figure(figsize=(22, 16), facecolor="#f8f9fa")
fig.suptitle("RepELA-Net Architecture for 2D Material Segmentation", fontsize=28, fontweight="bold", y=0.96)

# Provide fallback if image doesn't exist
img_path = "output/repela_base_20260320_083155/sample_images/image_6.png"
mask_path = "output/repela_base_20260320_083155/sample_images/prediction_6.png"
try:
    img = plt.imread(img_path)
except:
    img = np.random.rand(512, 512, 3)
try:
    mask = plt.imread(mask_path)
except:
    mask = np.random.rand(512, 512, 3)

# Define generic block drawing function
def draw_block(ax, x, y, w, h, title, subtitle, color, edge_color, zorder=10):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=0.5", 
                                  linewidth=2, edgecolor=edge_color, facecolor=color, zorder=zorder)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h/2+1.5, title, ha="center", va="center", fontsize=14, fontweight="bold", zorder=zorder+1)
    if subtitle:
        ax.text(x+w/2, y+h/2-1.5, subtitle, ha="center", va="center", fontsize=12, zorder=zorder+1)
    return rect

# ================= TOP PANEL: MAIN ARCHITECTURE =================
ax1 = fig.add_axes([0.05, 0.60, 0.9, 0.3])
ax1.axis("off")
ax1.add_patch(patches.FancyBboxPatch((0, 0), 100, 20, boxstyle="round,pad=1,rounding_size=2", 
                                    linewidth=3, edgecolor="#aaaaaa", facecolor="#ffffff", zorder=0))
ax1.text(2, 18, "(a) Main Network (Encoder-Decoder)", fontsize=18, fontweight="bold")

# Draw Encoder Blocks
ax1.imshow(img, extent=[3, 11, 4, 12], zorder=10)
ax1.text(7, 2, "Input Image\n512x512", ha="center", va="center", fontsize=12)

draw_block(ax1, 15, 6, 6, 8, "Stem", "C=32, 1/4", "#bbdefb", "#1976d2")
ax1.annotate("", xy=(15, 10), xytext=(11, 10), arrowprops=dict(arrowstyle="->", lw=2))

draw_block(ax1, 25, 5, 8, 10, "RepConv\nStage 1", "x2 blocks\nC=32, 1/8", "#fff9c4", "#fbc02d")
ax1.annotate("", xy=(25, 10), xytext=(21, 10), arrowprops=dict(arrowstyle="->", lw=2))

draw_block(ax1, 36, 4, 8, 12, "RepConv\nStage 2", "x2 blocks\nC=64, 1/16", "#fff9c4", "#fbc02d")
ax1.annotate("", xy=(36, 10), xytext=(33, 10), arrowprops=dict(arrowstyle="->", lw=2))

draw_block(ax1, 47, 3, 8, 14, "ELA\nStage 3", "x4 blocks\nC=128, 1/32", "#c8e6c9", "#388e3c")
ax1.annotate("", xy=(47, 10), xytext=(44, 10), arrowprops=dict(arrowstyle="->", lw=2))

draw_block(ax1, 58, 2, 8, 16, "ELA\nStage 4", "x2 blocks\nC=256, 1/64", "#c8e6c9", "#388e3c")
ax1.annotate("", xy=(58, 10), xytext=(55, 10), arrowprops=dict(arrowstyle="->", lw=2))

# Draw Decoder
draw_block(ax1, 72, 2, 14, 16, "DW-MFF\nDecoder", "Dynamic Weighted\nFusion", "#e1bee7", "#8e24aa")
ax1.annotate("", xy=(72, 10), xytext=(66, 10), arrowprops=dict(arrowstyle="->", lw=2))

draw_block(ax1, 88, 7, 3, 6, "BE", "Bndry\nEnh.", "#ffccbc", "#d84315")
ax1.annotate("", xy=(88, 10), xytext=(86, 10), arrowprops=dict(arrowstyle="->", lw=2))

# Skip connections
for x, y_start in [(18, 14), (29, 15), (40, 16), (51, 17)]:
    ax1.plot([x, x, 72], [y_start, 18, 18], color="#555555", lw=2, linestyle="--", zorder=5)
    ax1.annotate("", xy=(72, 18), xytext=(71.9, 18), arrowprops=dict(arrowstyle="->", color="#555555", lw=2))

# Output
ax1.imshow(mask, extent=[95, 103, 4, 12], zorder=10)
ax1.annotate("", xy=(95, 10), xytext=(91, 10), arrowprops=dict(arrowstyle="->", lw=2))
ax1.text(99, 2, "Segmentation\nMask", ha="center", va="center", fontsize=12)

ax1.set_xlim(0, 105)
ax1.set_ylim(-2, 22)


# ================= BOTTOM LEFT: REPCONV =================
ax2 = fig.add_axes([0.05, 0.1, 0.42, 0.45])
ax2.axis("off")
ax2.add_patch(patches.FancyBboxPatch((0, 0), 100, 100, boxstyle="round,pad=2,rounding_size=2", 
                                    linewidth=3, edgecolor="#aaaaaa", facecolor="#ffffff", zorder=0))
ax2.text(5, 92, "(b) RepConv Block", fontsize=18, fontweight="bold")

# Titles
ax2.text(30, 85, "Training", ha="center", fontsize=16, fontweight="bold")
ax2.text(80, 85, "Inference", ha="center", fontsize=16, fontweight="bold")
ax2.plot([60, 60], [10, 88], lw=3, color="#bbbbbb", linestyle="--")

def draw_mini(ax, x, y, text, color="#f3e5f5", edge="#8e24aa"):
    r = patches.Rectangle((x, y), 14, 8, facecolor=color, edgecolor=edge, lw=2, zorder=10)
    ax.add_patch(r)
    ax.text(x+7, y+4, text, ha="center", va="center", fontsize=12, fontweight="bold")

# Training logic
ax2.plot([30, 30], [82, 75], lw=2, color="black")
ax2.plot([12, 48], [75, 75], lw=2, color="black")

ax2.plot([12, 12], [75, 68], lw=2, color="black")
ax2.plot([30, 30], [75, 68], lw=2, color="black")
ax2.plot([48, 48], [75, 68], lw=2, color="black")

draw_mini(ax2, 5, 60, "Conv 3x3")
draw_mini(ax2, 5, 50, "BN", "#bbdefb", "#1976d2")
ax2.plot([12, 12], [60, 58], lw=2, color="black")

draw_mini(ax2, 23, 60, "Conv 1x1")
draw_mini(ax2, 23, 50, "BN", "#bbdefb", "#1976d2")
ax2.plot([30, 30], [60, 58], lw=2, color="black")

draw_mini(ax2, 41, 50, "Identity\n(BN)", "#bbdefb", "#1976d2")

ax2.plot([12, 12], [50, 42], lw=2, color="black")
ax2.plot([30, 30], [50, 42], lw=2, color="black")
ax2.plot([48, 48], [50, 42], lw=2, color="black")

ax2.plot([12, 48], [42, 42], lw=2, color="black")
ax2.plot([30, 30], [42, 38], lw=2, color="black")
ax2.scatter([30], [42], s=200, facecolor="white", edgecolor="black", zorder=20)
ax2.text(30, 42, "+", ha="center", va="center", fontsize=16, zorder=21)

draw_mini(ax2, 23, 30, "GELU", "#e8f5e9", "#2e7d32")
ax2.plot([30, 30], [30, 20], lw=2, color="black")

# Inference logic
ax2.annotate("", xy=(65, 55), xytext=(55, 55), arrowprops=dict(arrowstyle="simple", color="#ff9800", lw=2, shrinkA=5, shrinkB=5))
ax2.text(60, 60, "Structural\nRe-parameterization", ha="center", fontsize=12, color="#ef6c00", fontweight="bold")

ax2.plot([80, 80], [82, 68], lw=2, color="black")
draw_mini(ax2, 73, 60, "Conv 3x3")
ax2.plot([80, 80], [60, 38], lw=2, color="black")
draw_mini(ax2, 73, 30, "GELU", "#e8f5e9", "#2e7d32")
ax2.plot([80, 80], [30, 20], lw=2, color="black")
ax2.annotate("", xy=(80, 20), xytext=(80, 25), arrowprops=dict(arrowstyle="->", lw=2))
ax2.annotate("", xy=(30, 20), xytext=(30, 25), arrowprops=dict(arrowstyle="->", lw=2))

ax2.set_xlim(0, 100)
ax2.set_ylim(0, 100)


# ================= BOTTOM RIGHT: ELA BLOCK =================
ax3 = fig.add_axes([0.53, 0.1, 0.42, 0.45])
ax3.axis("off")
ax3.add_patch(patches.FancyBboxPatch((0, 0), 100, 100, boxstyle="round,pad=2,rounding_size=2", 
                                    linewidth=3, edgecolor="#aaaaaa", facecolor="#ffffff", zorder=0))
ax3.text(5, 92, "(c) ELA Block", fontsize=18, fontweight="bold")

# Draw architecture
ax3.plot([50, 50], [90, 82], lw=2, color="black")

# Norm & DWConv
draw_mini(ax3, 43, 74, "LayerNorm", "#b2dfdb", "#00796b")
ax3.plot([50, 50], [82, 74], lw=2, color="black")
draw_mini(ax3, 43, 62, "DWConv 3x3", "#bbdefb", "#1976d2")
ax3.plot([50, 50], [62, 54], lw=2, color="black")

# Multi-scale Linear Attention Box
r_attn = patches.Rectangle((20, 30), 60, 24, facecolor="#f1f8e9", edgecolor="#689f38", lw=2, zorder=5)
ax3.add_patch(r_attn)
ax3.text(50, 50, "Multi-scale Linear Attention", ha="center", fontsize=14, fontweight="bold")

draw_mini(ax3, 25, 35, "Q (Query)", "#ffccbc", "#d84315")
draw_mini(ax3, 43, 35, "K (Key)", "#ffccbc", "#d84315")
draw_mini(ax3, 61, 35, "V (Value)", "#ffccbc", "#d84315")

ax3.plot([50, 32], [54, 43], lw=2, color="black")
ax3.plot([50, 50], [54, 43], lw=2, color="black")
ax3.plot([50, 68], [54, 43], lw=2, color="black")

ax3.plot([50, 50], [30, 22], lw=2, color="black")
ax3.plot([10, 10], [86, 18], lw=2, color="black")
ax3.plot([10, 50], [86, 86], lw=2, color="black")
ax3.plot([10, 50], [18, 18], lw=2, color="black")

ax3.scatter([50], [18], s=200, facecolor="white", edgecolor="black", zorder=20)
ax3.text(50, 18, "+", ha="center", va="center", fontsize=16, zorder=21)

ax3.plot([50, 50], [18, 10], lw=2, color="black")
ax3.annotate("", xy=(50, 10), xytext=(50, 15), arrowprops=dict(arrowstyle="->", lw=2))

# FFN branch (simplified)
ax3.text(80, 50, "FFN\nBranch", ha="center", fontsize=14, color="#555")

ax3.set_xlim(0, 100)
ax3.set_ylim(0, 100)

plt.savefig("output/repela_architecture_mos2.png", dpi=300, bbox_inches="tight")
plt.close()
