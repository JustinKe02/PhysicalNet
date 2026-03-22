"""Quick eval: compare best_model vs EMA weights."""
import sys, os
sys.path.insert(0, '/root/autodl-tmp/PhysicalNet')
import torch
import numpy as np
from models.repela_net import repela_net_small
from datasets.mos2_dataset import MoS2Dataset
from torch.utils.data import DataLoader
from utils.metrics import SegmentationMetrics

device = torch.device('cuda')
run_dir = '/root/autodl-tmp/PhysicalNet/outputv2_ema/repela_small_20260322_174740'

# Build val dataset directly
val_ds = MoS2Dataset('/root/autodl-tmp/PhysicalNet/Mos2_data', 'val',
                     split_dir='splits/', crop_size=None)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

def eval_weights(label, state_dict):
    model = repela_net_small(num_classes=4, deep_supervision=True).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    metrics = SegmentationMetrics(4)
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            out = model(images)
            logits = out[0] if isinstance(out, tuple) else out
            preds = logits.argmax(dim=1)
            metrics.update(preds, masks)
    res = metrics.get_results()
    print(f'{label}: mIoU={res["mIoU"]:.4f} | '
          f'BG={res["per_class_iou"][0]:.3f} 1L={res["per_class_iou"][1]:.3f} '
          f'FL={res["per_class_iou"][2]:.3f} ML={res["per_class_iou"][3]:.3f}')

ckpt = torch.load(os.path.join(run_dir, 'best_model.pth'), map_location=device, weights_only=False)
eval_weights('best_model', ckpt['model'])

if 'ema' in ckpt:
    eval_weights('EMA (ckpt)', ckpt['ema'])
else:
    print('No EMA weights in checkpoint')
