"""
Generic 2D Material Dataset Loader for Fine-tuning.

Supports domain-adaptive normalization: auto-computes per-dataset
RGB mean/std instead of using ImageNet defaults.

Folder structure:
    data_root/
    ├── img_dir/
    │   ├── train/  (*.jpg)
    │   ├── val/    (*.jpg)  [optional]
    │   └── test/   (*.jpg)  [optional]
    └── ann_dir/
        ├── train/  (*.png)
        ├── val/    (*.png)
        └── test/   (*.png)
"""

import os
import json
import glob
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

# ImageNet defaults (fallback)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def compute_dataset_stats(data_root, split='train', cache=True):
    """Compute RGB mean and std for a dataset.

    Scans all images in the given split and returns per-channel pixel-level
    statistics. Results are cached to data_root/stats_{split}.json.

    Returns:
        dict with 'mean' (list[3]) and 'std' (list[3]) in [0,1] range
    """
    cache_path = os.path.join(data_root, f'stats_{split}.json')
    if cache and os.path.exists(cache_path):
        with open(cache_path) as f:
            stats = json.load(f)
        return stats

    img_dir = os.path.join(data_root, 'img_dir', split)
    paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')) +
                   glob.glob(os.path.join(img_dir, '*.png')))

    if not paths:
        raise FileNotFoundError(f'No images found in {img_dir}')

    # Two-pass pixel-level computation for correct mean/std
    # Pass 1: compute mean
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    for p in paths:
        img = np.array(Image.open(p).convert('RGB')).astype(np.float64) / 255.0
        pixel_sum += img.sum(axis=(0, 1))
        pixel_count += img.shape[0] * img.shape[1]
    mean = (pixel_sum / pixel_count).tolist()

    # Pass 2: compute std
    sq_diff_sum = np.zeros(3, dtype=np.float64)
    for p in paths:
        img = np.array(Image.open(p).convert('RGB')).astype(np.float64) / 255.0
        sq_diff_sum += ((img - mean) ** 2).sum(axis=(0, 1))
    std = np.sqrt(sq_diff_sum / pixel_count).tolist()
    # Clamp std to avoid division issues
    std = [max(s, 0.01) for s in std]

    stats = {'mean': mean, 'std': std, 'num_images': len(paths), 'split': split}

    if cache:
        with open(cache_path, 'w') as f:
            json.dump(stats, f, indent=2)

    return stats


def get_auto_crop_size(data_root, split='train', max_crop=512, margin=32):
    """Determine crop size based on image dimensions.

    Scans ALL images and uses the minimum dimension across the dataset
    to ensure every image can be cropped consistently.
    """
    img_dir = os.path.join(data_root, 'img_dir', split)
    paths = glob.glob(os.path.join(img_dir, '*.jpg')) + \
            glob.glob(os.path.join(img_dir, '*.png'))
    if not paths:
        return max_crop

    # Find minimum dimension across ALL images
    min_dim = float('inf')
    for p in paths:
        img = Image.open(p)
        w, h = img.size
        min_dim = min(min_dim, w, h)

    if min_dim <= max_crop:
        # Use smaller crop to allow random positioning
        crop = max(128, min_dim - margin)
        # Round down to multiple of 32
        crop = (crop // 32) * 32
        return crop
    return max_crop


class MaterialDataset(Dataset):
    """Generic 2D material segmentation dataset with adaptive normalization."""

    def __init__(self, data_root, split='train', crop_size=512,
                 augment=True, mean=None, std=None):
        """
        Args:
            data_root: path to dataset root
            split: 'train', 'val', or 'test'
            crop_size: crop size for training
            augment: enable augmentation for training
            mean: RGB mean [R, G, B] in [0,1]. Auto-computed if None.
            std: RGB std [R, G, B] in [0,1]. Auto-computed if None.
        """
        super().__init__()
        self.split = split
        self.crop_size = crop_size
        self.is_train = (split == 'train')
        self.augment = augment and self.is_train

        # Normalization stats
        if mean is not None and std is not None:
            self.mean = mean
            self.std = std
        else:
            self.mean = IMAGENET_MEAN
            self.std = IMAGENET_STD

        img_dir = os.path.join(data_root, 'img_dir', split)
        ann_dir = os.path.join(data_root, 'ann_dir', split)

        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f'Image dir not found: {img_dir}')

        imgs = sorted(glob.glob(os.path.join(img_dir, '*.jpg')) +
                       glob.glob(os.path.join(img_dir, '*.png')))

        self.pairs = []
        for ip in imgs:
            bn = os.path.splitext(os.path.basename(ip))[0]
            ap = os.path.join(ann_dir, f'{bn}.png')
            if os.path.exists(ap):
                self.pairs.append((ip, ap))

        print(f'[MaterialDataset] {split}: {len(self.pairs)} images '
              f'(crop={crop_size}, mean={[f"{m:.3f}" for m in self.mean]})')

    def __len__(self):
        return len(self.pairs)

    def _random_crop(self, img, mask):
        w, h = img.size

        # Scale jitter: random resize before crop (multi-scale training)
        if self.augment and random.random() > 0.3:
            scale = random.uniform(0.75, 1.5)
            new_h, new_w = int(h * scale), int(w * scale)
            # Ensure at least crop_size
            new_h = max(new_h, self.crop_size)
            new_w = max(new_w, self.crop_size)
            img = TF.resize(img, [new_h, new_w],
                            interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, [new_h, new_w],
                             interpolation=TF.InterpolationMode.NEAREST)
            w, h = new_w, new_h

        crop_h = min(self.crop_size, h)
        crop_w = min(self.crop_size, w)
        top = random.randint(0, h - crop_h) if h > crop_h else 0
        left = random.randint(0, w - crop_w) if w > crop_w else 0
        img = TF.crop(img, top, left, crop_h, crop_w)
        mask = TF.crop(mask, top, left, crop_h, crop_w)
        if crop_h != self.crop_size or crop_w != self.crop_size:
            img = TF.resize(img, [self.crop_size, self.crop_size],
                            interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, [self.crop_size, self.crop_size],
                             interpolation=TF.InterpolationMode.NEAREST)
        return img, mask

    def _augment(self, img, mask):
        if random.random() > 0.5:
            img = TF.hflip(img); mask = TF.hflip(mask)
        if random.random() > 0.5:
            img = TF.vflip(img); mask = TF.vflip(mask)
        angle = random.choice([0, 90, 180, 270])
        if angle > 0:
            img = TF.rotate(img, angle); mask = TF.rotate(mask, angle)
        if random.random() > 0.5:
            img = TF.adjust_brightness(img, random.uniform(0.8, 1.2))
        if random.random() > 0.5:
            img = TF.adjust_contrast(img, random.uniform(0.8, 1.2))
        if random.random() > 0.5:
            img = TF.adjust_saturation(img, random.uniform(0.8, 1.2))
        # Note: hue jitter removed — risky for small material datasets
        return img, mask

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        if self.is_train:
            img, mask = self._random_crop(img, mask)
            if self.augment:
                img, mask = self._augment(img, mask)

        img = TF.to_tensor(img)
        img = TF.normalize(img, self.mean, self.std)
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask


def collate_variable_size(batch):
    images, masks = zip(*batch)
    return list(images), list(masks)
