"""
Generate confusion matrices for all 4 models on the same 27-image test set.
"""
import os, numpy as np, torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from train_ablation import build_ablation_model
import segmentation_models_pytorch as smp

MEAN, STD = [0.485,0.456,0.406], [0.229,0.224,0.225]
CROP, STRIDE = 512, 384
CLASSES = ['Background', 'Monolayer', 'Fewlayer', 'Multilayer']
N_CLS = 4
OUTPUT_DIR = 'outputv4_plots'

MODELS = [
    {
        'name': 'RepELA-Small (Old)\nFocal α, 2.12M',
        'short': 'repela_small_old',
        'ckpt': 'outputv4_oldcfg/nocolor_oldcfg_20260322_233322/best_model.pth',
        'type': 'repela', 'ds': False,
    },
    {
        'name': 'RepELA-Small (New)\nDS+CopyPaste, 2.12M',
        'short': 'repela_small_new',
        'ckpt': 'outputv4/ablation_no_color_20260323_005458/best_model.pth',
        'type': 'repela', 'ds': True,
    },
    {
        'name': 'RepELA-Base\nFocal+Dice, 5.34M',
        'short': 'repela_base',
        'ckpt': 'output/repela_base_20260320_083155/best_model.pth',
        'type': 'repela_base', 'ds': True,
    },
    {
        'name': 'UNet-MiT-B0\nImageNet, 5.55M',
        'short': 'unet_mitb0',
        'ckpt': 'outputv3/unet_mit_b0_20260322_193307/best_model.pth',
        'type': 'unet', 'ds': False,
    },
]


def sliding_predict(model, img_t, dev, is_smp=False):
    _, H, W = img_t.shape
    ps = torch.zeros(N_CLS, H, W, device=dev)
    cnt = torch.zeros(H, W, device=dev)
    ph, pw = max(0, CROP-H), max(0, CROP-W)
    pad = F.pad(img_t, [0,pw,0,ph], mode='reflect') if ph>0 or pw>0 else img_t
    _, pH, pW = pad.shape
    ys = sorted(set(list(range(0, max(1, pH-CROP+1), STRIDE)) + [max(0, pH-CROP)]))
    xs = sorted(set(list(range(0, max(1, pW-CROP+1), STRIDE)) + [max(0, pW-CROP)]))
    for y in ys:
        for x in xs:
            c = pad[:, y:y+CROP, x:x+CROP]
            cH, cW = c.shape[1], c.shape[2]
            if is_smp:
                sph = (32 - cH%32)%32; spw = (32 - cW%32)%32
                if sph>0 or spw>0: c = F.pad(c, [0,spw,0,sph], mode='reflect')
            with torch.no_grad():
                o = model(c.unsqueeze(0).to(dev))
                l = o[0] if isinstance(o, tuple) else o
                if is_smp and (sph>0 or spw>0): l = l[:,:,:cH,:cW]
                p = F.softmax(l, 1)[0]
            ye, xe = min(y+CROP, H), min(x+CROP, W)
            ps[:, y:ye, x:xe] += p[:, :ye-y, :xe-x]
            cnt[y:ye, x:xe] += 1
    return (ps / cnt.clamp(min=1).unsqueeze(0)).argmax(0).cpu().numpy()


def load_model(cfg, dev):
    if cfg['type'] == 'unet':
        model = smp.Unet(encoder_name='mit_b0', encoder_weights=None, in_channels=3, classes=4).to(dev)
    elif cfg['type'] == 'repela_base':
        from models.repela_net import repela_net_base
        model = repela_net_base(num_classes=4, deep_supervision=cfg['ds']).to(dev)
    else:
        model = build_ablation_model('no_color', num_classes=4, deep_supervision=cfg['ds']).to(dev)
    ck = torch.load(cfg['ckpt'], map_location=dev, weights_only=False)
    # RepELA-Base checkpoint may miss color_enhance buffers (mean/std) added later
    strict = (cfg['type'] != 'repela_base')
    model.load_state_dict(ck['model'], strict=strict)
    model.eval()
    return model


def plot_confusion_matrix(cm, title, save_path):
    """Plot a single confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)

    for i in range(N_CLS):
        for j in range(N_CLS):
            val = cm_norm[i, j]
            count = cm[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.1%}\n({count:,})',
                    ha='center', va='center', fontsize=9, color=color)

    ax.set_xticks(range(N_CLS))
    ax.set_yticks(range(N_CLS))
    ax.set_xticklabels(CLASSES, fontsize=10, rotation=30, ha='right')
    ax.set_yticklabels(CLASSES, fontsize=10)
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ground Truth', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Ratio', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('splits/test.txt') as f:
        names = [l.strip() for l in f if l.strip()]

    print(f'Test set: {len(names)} images\n')

    # Collect all confusion matrices
    all_cms = {}

    for cfg in MODELS:
        print(f'Processing: {cfg["short"]}...')
        model = load_model(cfg, dev)
        is_smp = cfg['type'] == 'unet'

        cm = np.zeros((N_CLS, N_CLS), dtype=np.int64)

        for i, bn in enumerate(names):
            ip = f'Mos2_data/ori/MoS2/{bn}.jpg'
            if not os.path.exists(ip): ip = f'Mos2_data/ori/MoS2/{bn}.png'
            gp = f'Mos2_data/mask/{bn}.png'
            img = Image.open(ip).convert('RGB')
            gt = np.array(Image.open(gp))
            it = TF.normalize(TF.to_tensor(img), MEAN, STD)
            pred = sliding_predict(model, it, dev, is_smp)

            for r in range(N_CLS):
                for c in range(N_CLS):
                    cm[r, c] += ((gt == r) & (pred == c)).sum()

            print(f'  [{i+1}/{len(names)}] {bn}')

        all_cms[cfg['short']] = cm

        # Individual plot
        save_path = os.path.join(OUTPUT_DIR, f'confusion_matrix_{cfg["short"]}.png')
        plot_confusion_matrix(cm, cfg['name'], save_path)
        print(f'  Saved: {save_path}\n')

        del model
        torch.cuda.empty_cache()

    # Combined 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for idx, cfg in enumerate(MODELS):
        ax = axes[idx // 2][idx % 2]
        cm = all_cms[cfg['short']]
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)

        for i in range(N_CLS):
            for j in range(N_CLS):
                val = cm_norm[i, j]
                color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.1%}', ha='center', va='center',
                        fontsize=11, fontweight='bold', color=color)

        ax.set_xticks(range(N_CLS))
        ax.set_yticks(range(N_CLS))
        ax.set_xticklabels(CLASSES, fontsize=9, rotation=30, ha='right')
        ax.set_yticklabels(CLASSES, fontsize=9)
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('Ground Truth', fontsize=10)
        ax.set_title(cfg['name'], fontsize=11, fontweight='bold', pad=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle('Confusion Matrices — Test Set (27 images)', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    combined_path = os.path.join(OUTPUT_DIR, 'confusion_matrices_all.png')
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✅ Combined: {combined_path}')


if __name__ == '__main__':
    main()
