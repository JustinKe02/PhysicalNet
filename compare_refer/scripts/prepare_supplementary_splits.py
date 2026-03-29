"""
Prepare supplementary data splits for Line 2 and Line 3 experiments.
Creates train/val/test directory structures for WS2_supp, Gr_supp, and WS2_combined.
"""
import os, glob, shutil, random
import numpy as np
from pathlib import Path

BASE = '/root/autodl-tmp/PhysicalNet'
SUPP = os.path.join(BASE, 'supplementary_data')
PREP = os.path.join(BASE, 'supplementary_prepared')
SEED = 42


def split_dataset(names, ratios=(0.70, 0.15, 0.15)):
    """Split list into train/val/test."""
    random.seed(SEED)
    random.shuffle(names)
    n = len(names)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return names[:n_train], names[n_train:n_train+n_val], names[n_train+n_val:]


def copy_files(names, src_img_dir, src_mask_dir, dst_img_dir, dst_mask_dir):
    """Copy images and masks for a list of basenames."""
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_mask_dir, exist_ok=True)
    for name in names:
        # Image (from ori_jpg)
        src_img = os.path.join(src_img_dir, name + '.jpg')
        if os.path.exists(src_img):
            shutil.copy2(src_img, os.path.join(dst_img_dir, name + '.jpg'))
        # Mask
        src_mask = os.path.join(src_mask_dir, name + '.png')
        if os.path.exists(src_mask):
            shutil.copy2(src_mask, os.path.join(dst_mask_dir, name + '.png'))


def prepare_supp_dataset(material, supp_subdir, out_name):
    """Prepare one supplementary dataset with train/val/test splits."""
    src_img = os.path.join(SUPP, supp_subdir, 'ori_jpg')
    src_mask = os.path.join(SUPP, supp_subdir, 'mask')

    # Get all basenames
    masks = sorted(glob.glob(os.path.join(src_mask, '*.png')))
    names = [Path(m).stem for m in masks]

    train, val, test = split_dataset(names)
    out_dir = os.path.join(PREP, out_name)

    print(f'\n{"="*60}')
    print(f'  {out_name}: {len(names)} images → train={len(train)}, val={len(val)}, test={len(test)}')
    print(f'  Output: {out_dir}')
    print(f'{"="*60}')

    for split_name, split_list in [('train', train), ('val', val), ('test', test)]:
        copy_files(split_list,
                   src_img, src_mask,
                   os.path.join(out_dir, 'img_dir', split_name),
                   os.path.join(out_dir, 'ann_dir', split_name))
        print(f'  {split_name}: {len(split_list)} files copied')

    return train, val, test


def prepare_ws2_combined(ws2_supp_train):
    """Create WS2_combined: train = old + supp, val/test = old only."""
    old_ws2 = os.path.join(BASE, 'other data', 'WS2_data')
    out_dir = os.path.join(PREP, 'WS2_combined')

    print(f'\n{"="*60}')
    print(f'  WS2_combined')
    print(f'{"="*60}')

    # Train: old + supp
    dst_train_img = os.path.join(out_dir, 'img_dir', 'train')
    dst_train_ann = os.path.join(out_dir, 'ann_dir', 'train')
    os.makedirs(dst_train_img, exist_ok=True)
    os.makedirs(dst_train_ann, exist_ok=True)

    # Copy old WS2 train
    old_train_imgs = glob.glob(os.path.join(old_ws2, 'img_dir', 'train', '*'))
    old_train_anns = glob.glob(os.path.join(old_ws2, 'ann_dir', 'train', '*'))
    for f in old_train_imgs:
        shutil.copy2(f, dst_train_img)
    for f in old_train_anns:
        shutil.copy2(f, dst_train_ann)
    print(f'  train (old): {len(old_train_imgs)} images copied')

    # Copy supp WS2 train
    supp_img_dir = os.path.join(SUPP, 'WS2', 'ori_jpg')
    supp_mask_dir = os.path.join(SUPP, 'WS2', 'mask')
    for name in ws2_supp_train:
        src_img = os.path.join(supp_img_dir, name + '.jpg')
        src_mask = os.path.join(supp_mask_dir, name + '.png')
        if os.path.exists(src_img):
            shutil.copy2(src_img, os.path.join(dst_train_img, name + '.jpg'))
        if os.path.exists(src_mask):
            shutil.copy2(src_mask, os.path.join(dst_train_ann, name + '.png'))
    print(f'  train (supp): {len(ws2_supp_train)} images copied')
    print(f'  train (combined): {len(os.listdir(dst_train_img))} total')

    # Val: old only
    for split in ['val', 'test']:
        dst_img = os.path.join(out_dir, 'img_dir', split)
        dst_ann = os.path.join(out_dir, 'ann_dir', split)
        os.makedirs(dst_img, exist_ok=True)
        os.makedirs(dst_ann, exist_ok=True)
        old_imgs = glob.glob(os.path.join(old_ws2, 'img_dir', split, '*'))
        old_anns = glob.glob(os.path.join(old_ws2, 'ann_dir', split, '*'))
        for f in old_imgs:
            shutil.copy2(f, dst_img)
        for f in old_anns:
            shutil.copy2(f, dst_ann)
        print(f'  {split} (old only): {len(old_imgs)} images')


if __name__ == '__main__':
    # Line 2: Supplementary datasets
    ws2_supp_train, _, _ = prepare_supp_dataset('WS2', 'WS2', 'WS2_supp')
    prepare_supp_dataset('Gr', 'Gr', 'Gr_supp')

    # Line 3: WS2 combined
    prepare_ws2_combined(ws2_supp_train)

    # Summary
    print(f'\n{"="*60}')
    print('  Summary')
    print(f'{"="*60}')
    for d in ['WS2_supp', 'Gr_supp', 'WS2_combined']:
        for split in ['train', 'val', 'test']:
            p = os.path.join(PREP, d, 'img_dir', split)
            n = len(os.listdir(p)) if os.path.exists(p) else 0
            print(f'  {d}/{split}: {n}')
    print('\n✅ All data prepared!')
