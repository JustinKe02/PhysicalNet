"""
Generate fixed train/val/test splits for MoS2 dataset.

Split ratio: 70% train / 15% val / 15% test (default)
Saves to splits/train.txt, splits/val.txt, splits/test.txt

Usage:
    python scripts/generate_splits.py --data_root Mos2_data --output splits/
"""

import os
import glob
import random
import argparse
from pathlib import Path

# Ensure cwd = project root
os.chdir(str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='Mos2_data')
    parser.add_argument('--output', type=str, default='splits/')
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    img_dir = os.path.join(args.data_root, 'ori', 'MoS2')
    mask_dir = os.path.join(args.data_root, 'mask')

    # Discover all valid image-mask pairs
    all_images = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
    basenames = []
    for img_path in all_images:
        bn = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(mask_dir, f'{bn}.png')
        if os.path.exists(mask_path):
            basenames.append(bn)

    print(f'Total image-mask pairs: {len(basenames)}')

    # Shuffle with fixed seed
    rng = random.Random(args.seed)
    indices = list(range(len(basenames)))
    rng.shuffle(indices)

    # Split
    n = len(basenames)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    # n_test = remaining

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_names = sorted([basenames[i] for i in train_idx])
    val_names = sorted([basenames[i] for i in val_idx])
    test_names = sorted([basenames[i] for i in test_idx])

    print(f'Train: {len(train_names)}, Val: {len(val_names)}, Test: {len(test_names)}')

    # Save
    os.makedirs(args.output, exist_ok=True)
    for split_name, names in [('train', train_names), ('val', val_names), ('test', test_names)]:
        path = os.path.join(args.output, f'{split_name}.txt')
        with open(path, 'w') as f:
            for name in names:
                f.write(name + '\n')
        print(f'Saved: {path} ({len(names)} samples)')


if __name__ == '__main__':
    main()
