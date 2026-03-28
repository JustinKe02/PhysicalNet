"""
Generate individual prediction images for ablation and decoder comparison.
Output: output/individual_preds/{ablation,decoder}/{sample}_{model}.png
"""
import os, sys, glob
import numpy as np
from PIL import Image
import torch, torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms as T

sys.path.insert(0, '/root/autodl-tmp/PhysicalNet')
os.chdir('/root/autodl-tmp/PhysicalNet')

from models.repela_net import RepELANet
from tools.train_ablation import build_ablation_model
from tools.train_decoder_compare import build_encoder_with_decoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# V palette — colorblind-safe
COLORS_4 = {0: (210,210,210), 1: (253,174,97), 2: (116,173,209), 3: (69,117,180)}
mean = [0.4914, 0.5107, 0.5445]
std = [0.2102, 0.2041, 0.2015]

def colorize(mask, palette):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in palette.items():
        rgb[mask == cls] = color
    return rgb

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
                o = model(c); o = o[0] if isinstance(o, tuple) else o
                p = F.softmax(o, dim=1)[0]
            ye = min(y+crop, H); xe = min(x+crop, W)
            pred_sum[:, y:ye, x:xe] += p[:, :ye-y, :xe-x]
            count[y:ye, x:xe] += 1
    return (pred_sum / count.clamp(min=1).unsqueeze(0)).argmax(0).cpu().numpy()

# ── Ablation ──
ABL_SAMPLES = ['m155', 'm106', 'm10']
ABL_DIR = 'output/individual_preds/ablation'
os.makedirs(ABL_DIR, exist_ok=True)

ablation_models = [
    ('Ours', None, 'output/seed_test/seed_42/repela_small_20260324_080734/best_model.pth'),
    ('wo_ELA', 'no_ela', glob.glob('output/ablation/no_ela_*/best_model.pth')[0]),
    ('wo_DWMFF', 'no_dwmff', glob.glob('output/ablation/no_dwmff_*/best_model.pth')[0]),
    ('wo_Boundary', 'no_boundary', glob.glob('output/ablation/no_boundary_*/best_model.pth')[0]),
]

# ── Decoder ──
DEC_SAMPLES = ['m124', 'm80', 'm40']
DEC_DIR = 'output/individual_preds/decoder'
os.makedirs(DEC_DIR, exist_ok=True)

decoder_models = [
    ('FPN', 'fpn'),
    ('PPM', 'ppm'),
    ('SegFormer', 'segformer'),
    ('Ours', 'ours'),
]

def save_originals(samples, out_dir):
    for name in samples:
        # Original image
        img = Image.open(f'Mos2_data/ori/MoS2/{name}.jpg').convert('RGB')
        img.save(f'{out_dir}/{name}_original.png')
        # GT mask
        gt = np.array(Image.open(f'Mos2_data/mask/{name}.png'))
        gt_rgb = colorize(gt, COLORS_4)
        Image.fromarray(gt_rgb).save(f'{out_dir}/{name}_GT.png')
        print(f'  {name}: original + GT saved')

print('=== Saving originals ===')
save_originals(ABL_SAMPLES, ABL_DIR)
save_originals(DEC_SAMPLES, DEC_DIR)

print('\n=== Ablation predictions ===')
for label, ablation, ckpt in ablation_models:
    print(f'  Loading {label}...')
    if ablation is None:
        model = RepELANet(num_classes=4, channels=(32,64,128,256),
                          num_blocks=(2,2,4,2), num_heads=(0,0,4,8),
                          decoder_channels=128, deep_supervision=False)
    else:
        model = build_ablation_model(ablation, num_classes=4, deep_supervision=False)
    model = load_model_weights(model, ckpt)
    for name in ABL_SAMPLES:
        img = Image.open(f'Mos2_data/ori/MoS2/{name}.jpg').convert('RGB')
        img_t = TF.normalize(T.ToTensor()(img), mean, std)
        pred = sliding_window_pred(model, img_t)
        pred_rgb = colorize(pred, COLORS_4)
        out_path = f'{ABL_DIR}/{name}_{label}.png'
        Image.fromarray(pred_rgb).save(out_path)
        print(f'    saved: {out_path}')
    del model; torch.cuda.empty_cache()

print('\n=== Decoder predictions ===')
for label, dname in decoder_models:
    print(f'  Loading {label}...')
    ckpt = glob.glob(f'output/decoder_compare/{dname}_*/best_model.pth')[0]
    model = build_encoder_with_decoder(dname, num_classes=4)
    model = load_model_weights(model, ckpt)
    for name in DEC_SAMPLES:
        img = Image.open(f'Mos2_data/ori/MoS2/{name}.jpg').convert('RGB')
        img_t = TF.normalize(T.ToTensor()(img), mean, std)
        pred = sliding_window_pred(model, img_t)
        pred_rgb = colorize(pred, COLORS_4)
        out_path = f'{DEC_DIR}/{name}_{label}.png'
        Image.fromarray(pred_rgb).save(out_path)
        print(f'    saved: {out_path}')
    del model; torch.cuda.empty_cache()

print('\n✅ All individual images saved!')
print(f'  Ablation: {ABL_DIR}/')
print(f'  Decoder:  {DEC_DIR}/')
