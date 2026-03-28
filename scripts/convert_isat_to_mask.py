"""
ISAT JSON → PNG Mask Converter for supplementary_data.

Converts ISAT-format polygon annotations to pixel-level PNG masks
and unifies image formats (.tif → .jpg).

Usage:
    python scripts/convert_isat_to_mask.py [--preview N]

Output structure per dataset:
    supplementary_data/{Material}/
        mask/           ← generated PNG masks
        ori_jpg/        ← unified .jpg images (tif converted)
"""
import os, sys, json, glob, argparse
import numpy as np
from PIL import Image
from pathlib import Path

# ── Class mappings (consistent with existing datasets) ────────────
# MoS2 & WS2: 4 classes — same as Mos2_data
CLASS_MAP_4 = {
    '__background__': 0,
    'background': 0,
    'monolayer': 1,
    'fewlayer': 2,
    'multilayer': 3,
}

# Graphene: 3 classes (no monolayer in this dataset)
# Map to match existing graphene data: BG=0, fewlayer=1, multilayer=2
CLASS_MAP_GR = {
    '__background__': 0,
    'background': 0,
    'fewlayer': 1,
    'multilayer': 2,
}

DATASET_CONFIGS = {
    'MoS2': {'class_map': CLASS_MAP_4, 'classes': ['BG','1L','FL','ML']},
    'WS2':  {'class_map': CLASS_MAP_4, 'classes': ['BG','1L','FL','ML']},
    'Gr':   {'class_map': CLASS_MAP_GR, 'classes': ['BG','FL','ML']},
}


def polygon_to_mask(segmentation, height, width):
    """Convert ISAT polygon (list of [x,y]) to binary mask using PIL."""
    from PIL import ImageDraw
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    # segmentation is list of [x,y] pairs
    if len(segmentation) > 0 and isinstance(segmentation[0], list):
        pts = [(p[0], p[1]) for p in segmentation]
    else:
        # flat list
        pts = [(segmentation[i], segmentation[i+1])
               for i in range(0, len(segmentation), 2)]
    if len(pts) >= 3:
        draw.polygon(pts, fill=1)
    return np.array(mask, dtype=np.uint8)


def convert_json_to_mask(json_path, class_map):
    """Convert one ISAT JSON file to a pixel-level mask."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    info = data['info']
    w, h = info['width'], info['height']
    mask = np.zeros((h, w), dtype=np.uint8)  # 0 = background

    objects = data.get('objects', [])
    # Sort by area descending so smaller objects get drawn on top
    for obj in sorted(objects, key=lambda x: x.get('area', 0), reverse=True):
        category = obj['category']
        if category not in class_map:
            print(f'  Warning: unknown category "{category}" in {json_path}, skipping')
            continue
        cls_id = class_map[category]
        if cls_id == 0:
            continue  # skip background objects

        seg = obj['segmentation']
        obj_mask = polygon_to_mask(seg, h, w)
        mask[obj_mask > 0] = cls_id

    return mask


def convert_image_to_jpg(src_path, dst_path):
    """Convert any image format to .jpg."""
    img = Image.open(src_path).convert('RGB')
    img.save(dst_path, 'JPEG', quality=95)


def process_dataset(base_dir, material, class_map, class_names, preview=0):
    """Process one dataset: convert all JSONs → masks, unify images."""
    ori_dir = os.path.join(base_dir, material, 'ori')
    label_dir = os.path.join(base_dir, material, 'label')
    mask_dir = os.path.join(base_dir, material, 'mask')
    jpg_dir = os.path.join(base_dir, material, 'ori_jpg')

    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(jpg_dir, exist_ok=True)

    jsons = sorted(glob.glob(os.path.join(label_dir, '*.json')))
    print(f'\n{"="*60}')
    print(f'  {material}: {len(jsons)} annotations')
    print(f'  Classes: {class_names}')
    print(f'{"="*60}')

    stats = {name: 0 for name in class_names}
    skipped = 0
    converted = 0

    for jf in jsons:
        basename = Path(jf).stem  # e.g., 's1', 'g1', 'w1'

        # Find matching image
        img_path = None
        for ext in ['.jpg', '.tif', '.tiff', '.png', '.bmp']:
            candidate = os.path.join(ori_dir, basename + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            print(f'  [SKIP] No image for {basename}')
            skipped += 1
            continue

        # Convert JSON → mask
        mask = convert_json_to_mask(jf, class_map)

        # Save mask as PNG
        mask_path = os.path.join(mask_dir, basename + '.png')
        Image.fromarray(mask).save(mask_path)

        # Convert image to jpg if needed
        jpg_path = os.path.join(jpg_dir, basename + '.jpg')
        if img_path.lower().endswith('.jpg') or img_path.lower().endswith('.jpeg'):
            # Already jpg, just copy or symlink
            if not os.path.exists(jpg_path):
                import shutil
                shutil.copy2(img_path, jpg_path)
        else:
            convert_image_to_jpg(img_path, jpg_path)

        # Count class pixels
        for i, name in enumerate(class_names):
            stats[name] += np.sum(mask == i)

        converted += 1

    # Print summary
    total_px = sum(stats.values())
    print(f'\n  Converted: {converted}, Skipped: {skipped}')
    print(f'  Class pixel distribution:')
    for name, count in stats.items():
        pct = count / max(total_px, 1) * 100
        print(f'    {name:12s}: {count:>12,} px ({pct:5.1f}%)')

    # Verify a few masks
    print(f'\n  Mask unique values check (first 5):')
    masks = sorted(glob.glob(os.path.join(mask_dir, '*.png')))[:5]
    for mp in masks:
        m = np.array(Image.open(mp))
        print(f'    {os.path.basename(mp)}: shape={m.shape}, unique={np.unique(m).tolist()}')

    # Preview
    if preview > 0:
        _generate_preview(material, base_dir, masks[:preview], class_map, class_names)


def _generate_preview(material, base_dir, mask_paths, class_map, class_names):
    """Generate a visual preview of converted masks."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    n_classes = len(class_names)
    # Color palette
    palette = np.array([
        [0, 0, 0],       # BG - black
        [255, 0, 0],     # class 1 - red
        [0, 255, 0],     # class 2 - green
        [0, 0, 255],     # class 3 - blue
    ], dtype=np.uint8)[:n_classes]

    n = len(mask_paths)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4*n))
    if n == 1:
        axes = axes[np.newaxis, :]

    jpg_dir = os.path.join(base_dir, material, 'ori_jpg')

    for row, mp in enumerate(mask_paths):
        bn = Path(mp).stem
        jp = os.path.join(jpg_dir, bn + '.jpg')
        img = Image.open(jp).convert('RGB')
        mask = np.array(Image.open(mp))

        # Colorize mask
        h, w = mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for c in range(n_classes):
            color_mask[mask == c] = palette[c]

        # Overlay
        overlay = (np.array(img) * 0.6 + color_mask * 0.4).astype(np.uint8)

        axes[row, 0].imshow(np.array(img))
        axes[row, 0].set_title('Image' if row == 0 else '')
        axes[row, 0].set_ylabel(bn, fontsize=9, rotation=0, labelpad=40)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(color_mask)
        axes[row, 1].set_title('Mask' if row == 0 else '')
        axes[row, 1].axis('off')

        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title('Overlay' if row == 0 else '')
        axes[row, 2].axis('off')

    legend = [Patch(facecolor=palette[i]/255.0, label=class_names[i])
              for i in range(n_classes)]
    fig.legend(handles=legend, loc='lower center', ncol=n_classes, fontsize=10)
    plt.suptitle(f'{material} — Converted Mask Preview', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(base_dir, material, f'preview_{material}.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  Preview saved: {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='supplementary_data',
                        help='Path to supplementary_data directory')
    parser.add_argument('--preview', type=int, default=5,
                        help='Number of preview images to generate per dataset (0=none)')
    parser.add_argument('--materials', nargs='+', default=['MoS2', 'Gr', 'WS2'],
                        help='Which materials to process')
    args = parser.parse_args()

    for material in args.materials:
        cfg = DATASET_CONFIGS[material]
        process_dataset(args.data_dir, material, cfg['class_map'],
                        cfg['classes'], preview=args.preview)

    print('\n✅ All conversions complete!')
