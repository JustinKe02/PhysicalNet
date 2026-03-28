"""
Qualitative inference comparison for Ablation (Section 4) and Decoder (Section 6).
Generates 2 publication-ready figures.
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torch, torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms as T

sys.path.insert(0, '/root/autodl-tmp/PhysicalNet')
os.chdir('/root/autodl-tmp/PhysicalNet')

from models.repela_net import RepELANet
from tools.train_ablation import build_ablation_model
from tools.train_decoder_compare import build_encoder_with_decoder
from datasets.mos2_dataset import MoS2Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR = 'output/transfer_vis'

# MoS2 color palette — V: Nature/ColorBrewer colorblind-safe
COLORS = {0: (210,210,210), 1: (253,174,97), 2: (116,173,209), 3: (69,117,180)}
CLASS_NAMES = ['BG', 'Mono', 'Few', 'Multi']

def colorize(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in COLORS.items():
        rgb[mask == cls] = color
    return rgb

# MoS2 mean/std (pre-computed from training set)
mean = [0.4914, 0.5107, 0.5445]
std = [0.2102, 0.2041, 0.2015]

def load_model_weights(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('model', ckpt)
    model.load_state_dict(sd, strict=False)
    return model.to(device).eval()

def sliding_window_pred(model, img_t, crop=512, stride=384, nc=4):
    _, H, W = img_t.shape
    pred_sum = torch.zeros(nc, H, W, device=device)
    count = torch.zeros(H, W, device=device)
    pad_h = max(0, crop - H); pad_w = max(0, crop - W)
    if pad_h > 0 or pad_w > 0:
        img_t = F.pad(img_t, [0, pad_w, 0, pad_h], mode='reflect')
    _, pH, pW = img_t.shape
    ys = sorted(set(list(range(0, max(1, pH-crop+1), stride)) + [max(0, pH-crop)]))
    xs = sorted(set(list(range(0, max(1, pW-crop+1), stride)) + [max(0, pW-crop)]))
    for y in ys:
        for x in xs:
            c = img_t[:, y:y+crop, x:x+crop].unsqueeze(0).to(device)
            with torch.no_grad():
                o = model(c)
                o = o[0] if isinstance(o, tuple) else o
                p = F.softmax(o, dim=1)[0]
            ye = min(y+crop, H); xe = min(x+crop, W)
            pred_sum[:, y:ye, x:xe] += p[:, :ye-y, :xe-x]
            count[y:ye, x:xe] += 1
    return (pred_sum / count.clamp(min=1).unsqueeze(0)).argmax(0).cpu().numpy()

def read_image(name):
    """Read image and GT mask by test sample name."""
    img_path = f'Mos2_data/ori/MoS2/{name}.jpg'
    gt_path = f'Mos2_data/mask/{name}.png'
    img = Image.open(img_path).convert('RGB')
    gt = np.array(Image.open(gt_path))
    img_t = TF.normalize(T.ToTensor()(img), mean, std)
    return np.array(img), gt, img_t

# Ablation: samples where Ours advantage is largest
ABLATION_SAMPLES = ['m155', 'm106', 'm10']
# Decoder: samples where Ours performs competitively
DECODER_SAMPLES = ['m124', 'm80', 'm40']

# ══════════════════════════════════════════════════════════
# Figure 1: Ablation Comparison
# ══════════════════════════════════════════════════════════
def make_ablation_figure():
    print('=== Ablation Comparison Figure ===')
    variants = [
        ('Ours', 'seed42', None),
        ('w/o ELA', 'no_ela', 'no_ela'),
        ('w/o DW-MFF', 'no_dwmff', 'no_dwmff'),
        ('w/o Boundary', 'no_boundary', 'no_boundary'),
    ]

    ckpt_map = {
        'seed42': 'output/seed_test/seed_42/repela_small_20260324_080734/best_model.pth',
        'no_ela': glob.glob('output/ablation/no_ela_*/best_model.pth')[0],
        'no_dwmff': glob.glob('output/ablation/no_dwmff_*/best_model.pth')[0],
        'no_boundary': glob.glob('output/ablation/no_boundary_*/best_model.pth')[0],
    }

    # Load predictions
    preds = {}
    for label, ckpt_key, ablation in variants:
        print(f'  Loading {label}...')
        if ablation is None:
            model = RepELANet(num_classes=4, channels=(32,64,128,256),
                              num_blocks=(2,2,4,2), num_heads=(0,0,4,8),
                              decoder_channels=128, deep_supervision=False)
        else:
            model = build_ablation_model(ablation, num_classes=4, deep_supervision=False)
        model = load_model_weights(model, ckpt_map[ckpt_key])
        preds[label] = {}
        for name in ABLATION_SAMPLES:
            _, _, img_t = read_image(name)
            preds[label][name] = sliding_window_pred(model, img_t)
        del model; torch.cuda.empty_cache()

    # Build figure: rows=samples, cols=Image GT Ours w/oELA w/oDW-MFF w/oBoundary
    cols = ['Image', 'GT'] + [v[0] for v in variants]
    nrows = len(ABLATION_SAMPLES)
    ncols = len(cols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5*ncols, 2.5*nrows))

    for i, name in enumerate(ABLATION_SAMPLES):
        img_np, gt, _ = read_image(name)
        h = 256; w = int(img_np.shape[1] * h / img_np.shape[0])
        img_r = np.array(Image.fromarray(img_np).resize((w, h)))
        gt_r = np.array(Image.fromarray(colorize(gt)).resize((w, h), Image.NEAREST))

        axes[i][0].imshow(img_r); axes[i][0].axis('off')
        axes[i][1].imshow(gt_r); axes[i][1].axis('off')

        for j, (label, _, _) in enumerate(variants):
            pred_r = np.array(Image.fromarray(colorize(preds[label][name])).resize((w, h), Image.NEAREST))
            axes[i][j+2].imshow(pred_r); axes[i][j+2].axis('off')

    for j, title in enumerate(cols):
        axes[0][j].set_title(title, fontsize=10, fontweight='bold')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=np.array(c)/255., label=n) for n, c in zip(CLASS_NAMES, COLORS.values())]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    plt.suptitle('Ablation Study — Qualitative Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = f'{OUT_DIR}/ablation_qualitative.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')

# ══════════════════════════════════════════════════════════
# Figure 2: Decoder Comparison
# ══════════════════════════════════════════════════════════
def make_decoder_figure():
    print('=== Decoder Comparison Figure ===')
    decoders = [
        ('FPN', 'fpn'),
        ('PPM', 'ppm'),
        ('SegFormer', 'segformer'),
        ('Ours', 'ours'),
    ]

    ckpt_map = {}
    for label, name in decoders:
        paths = glob.glob(f'output/decoder_compare/{name}_*/best_model.pth')
        if paths:
            ckpt_map[name] = paths[0]
        else:
            print(f'  WARNING: No checkpoint for {name}')

    preds = {}
    for label, name in decoders:
        if name not in ckpt_map: continue
        print(f'  Loading {label}...')
        model = build_encoder_with_decoder(name, num_classes=4)
        model = load_model_weights(model, ckpt_map[name])
        preds[label] = {}
        for sname in DECODER_SAMPLES:
            _, _, img_t = read_image(sname)
            preds[label][sname] = sliding_window_pred(model, img_t)
        del model; torch.cuda.empty_cache()

    cols = ['Image', 'GT'] + [d[0] for d in decoders if d[0] in preds]
    nrows = len(DECODER_SAMPLES)
    ncols = len(cols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5*ncols, 2.5*nrows))

    for i, name in enumerate(DECODER_SAMPLES):
        img_np, gt, _ = read_image(name)
        h = 256; w = int(img_np.shape[1] * h / img_np.shape[0])
        img_r = np.array(Image.fromarray(img_np).resize((w, h)))
        gt_r = np.array(Image.fromarray(colorize(gt)).resize((w, h), Image.NEAREST))

        axes[i][0].imshow(img_r); axes[i][0].axis('off')
        axes[i][1].imshow(gt_r); axes[i][1].axis('off')

        for j, (label, _) in enumerate([(l,n) for l,n in decoders if l in preds]):
            pred_r = np.array(Image.fromarray(colorize(preds[label][name])).resize((w, h), Image.NEAREST))
            axes[i][j+2].imshow(pred_r); axes[i][j+2].axis('off')

    for j, title in enumerate(cols):
        axes[0][j].set_title(title, fontsize=10, fontweight='bold')

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=np.array(c)/255., label=n) for n, c in zip(CLASS_NAMES, COLORS.values())]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    plt.suptitle('Decoder Comparison — Qualitative Results', fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = f'{OUT_DIR}/decoder_qualitative.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')


if __name__ == '__main__':
    make_ablation_figure()
    make_decoder_figure()
    print('\n✅ Both figures complete!')
