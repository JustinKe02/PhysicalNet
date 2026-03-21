"""
MoS2 Dataset Loader for RepELA-Net (v2).

Changes from v1:
  - Uses fixed split files (splits/train.txt, val.txt, test.txt)
  - Training: random crop + augmentation
  - Validation/Test: full image (no crop, no augmentation)
  - Sliding window evaluation handled externally (eval.py / train.py)
"""

import os
import glob
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


class MoS2Dataset(Dataset):
    """MoS2 2D Material Segmentation Dataset.

    Structure:
        data_root/
        ├── ori/MoS2/    (RGB images, *.jpg)
        ├── mask/         (segmentation masks, *.png, values 0-3)
        └── splits/       (train.txt, val.txt, test.txt — basenames)

    Classes:
        0: background, 1: monolayer, 2: fewlayer, 3: multilayer
    """

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    CLASSES = ['background', 'monolayer', 'fewlayer', 'multilayer']
    # Inverse-frequency weights (bg:74.86%, mono:3.12%, few:2.46%, multi:19.56%)
    CLASS_WEIGHTS = [0.15, 3.60, 4.56, 0.57]

    def __init__(self, data_root, split='train', split_dir='splits/',
                 crop_size=512, augment=True, copy_paste=True):
        """
        Args:
            data_root: path to Mos2_data directory
            split: 'train', 'val', or 'test'
            split_dir: directory containing train.txt / val.txt / test.txt
            crop_size: crop size for training (ignored for val/test)
            augment: whether to apply augmentation (auto-disabled for val/test)
            copy_paste: whether to apply CopyPaste augmentation (train only)
        """
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.crop_size = crop_size
        self.is_train = (split == 'train')
        self.augment = augment and self.is_train
        self.copy_paste_enabled = copy_paste and self.is_train

        img_dir = os.path.join(data_root, 'ori', 'MoS2')
        mask_dir = os.path.join(data_root, 'mask')

        # Load split file
        split_file = os.path.join(split_dir, f'{split}.txt')
        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f'Split file not found: {split_file}\n'
                f'Run: python generate_splits.py --data_root {data_root} --output {split_dir}'
            )

        with open(split_file, 'r') as f:
            basenames = [line.strip() for line in f if line.strip()]

        self.pairs = []
        for bn in basenames:
            img_path = os.path.join(img_dir, f'{bn}.jpg')
            mask_path = os.path.join(mask_dir, f'{bn}.png')
            if os.path.exists(img_path) and os.path.exists(mask_path):
                self.pairs.append((img_path, mask_path))
            else:
                print(f'  Warning: missing {bn}, skipped')

        print(f'[MoS2Dataset] {split}: {len(self.pairs)} images '
              f'(from {split_file})')

    def __len__(self):
        return len(self.pairs)

    def _random_crop(self, img, mask):
        """Random crop for training."""
        w, h = img.size
        crop_h = min(self.crop_size, h)
        crop_w = min(self.crop_size, w)

        top = random.randint(0, h - crop_h) if h > crop_h else 0
        left = random.randint(0, w - crop_w) if w > crop_w else 0

        img = TF.crop(img, top, left, crop_h, crop_w)
        mask = TF.crop(mask, top, left, crop_h, crop_w)

        # Resize to exact crop_size if needed
        if crop_h != self.crop_size or crop_w != self.crop_size:
            img = TF.resize(img, [self.crop_size, self.crop_size],
                            interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, [self.crop_size, self.crop_size],
                             interpolation=TF.InterpolationMode.NEAREST)
        return img, mask

    def _augment(self, img, mask):
        """Data augmentation (train only)."""
        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        if random.random() > 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)

        angle = random.choice([0, 90, 180, 270])
        if angle > 0:
            img = TF.rotate(img, angle)
            mask = TF.rotate(mask, angle)

        if random.random() > 0.5:
            img = TF.adjust_brightness(img, random.uniform(0.8, 1.2))
        if random.random() > 0.5:
            img = TF.adjust_contrast(img, random.uniform(0.8, 1.2))
        if random.random() > 0.5:
            img = TF.adjust_saturation(img, random.uniform(0.8, 1.2))
        if random.random() > 0.7:
            img = TF.gaussian_blur(img, kernel_size=3)

        return img, mask

    def _copy_paste(self, img, mask):
        """CopyPaste augmentation for minority classes (monolayer=1, fewlayer=2).

        Strategy:
        1. Pick a random image from the training set
        2. Find connected regions of minority classes in that image
        3. Crop a region containing minority pixels
        4. Paste it onto the current image at a random position

        This directly increases minority class pixel frequency per batch.
        """
        if random.random() > 0.4:  # 60% probability
            return img, mask

        # Pick a random donor image
        donor_idx = random.randint(0, len(self.pairs) - 1)
        donor_img_path, donor_mask_path = self.pairs[donor_idx]
        donor_img = Image.open(donor_img_path).convert('RGB')
        donor_mask = np.array(Image.open(donor_mask_path))

        # Bias toward fewlayer (70%) since it's the weakest class
        target_class = 2 if random.random() < 0.7 else 1
        class_mask = (donor_mask == target_class)

        if class_mask.sum() < 100:  # Too few pixels, skip
            return img, mask

        # Find bounding box of the minority class region
        rows = np.any(class_mask, axis=1)
        cols = np.any(class_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Limit patch size to crop_size // 2
        max_sz = self.crop_size // 2
        if rmax - rmin > max_sz:
            mid = (rmin + rmax) // 2
            rmin, rmax = mid - max_sz // 2, mid + max_sz // 2
        if cmax - cmin > max_sz:
            mid = (cmin + cmax) // 2
            cmin, cmax = mid - max_sz // 2, mid + max_sz // 2
        rmin, cmin = max(0, rmin), max(0, cmin)
        rmax = min(donor_mask.shape[0], rmax + 1)
        cmax = min(donor_mask.shape[1], cmax + 1)

        # Extract patch
        patch_img = np.array(donor_img)[rmin:rmax, cmin:cmax]
        patch_mask = donor_mask[rmin:rmax, cmin:cmax]
        patch_binary = (patch_mask == target_class)

        # Convert current img/mask to numpy
        img_np = np.array(img)
        mask_np = np.array(mask)
        h, w = img_np.shape[:2]
        ph, pw = patch_img.shape[:2]

        if ph == 0 or pw == 0 or ph > h or pw > w:
            return img, mask

        # Random paste position
        top = random.randint(0, h - ph)
        left = random.randint(0, w - pw)

        # Paste only minority class pixels (not background)
        img_np[top:top+ph, left:left+pw][patch_binary] = patch_img[patch_binary]
        mask_np[top:top+ph, left:left+pw][patch_binary] = patch_mask[patch_binary]

        return Image.fromarray(img_np), Image.fromarray(mask_np)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        if self.is_train:
            # Training: random crop + CopyPaste + augmentation
            img, mask = self._random_crop(img, mask)
            if self.augment:
                if self.copy_paste_enabled:
                    img, mask = self._copy_paste(img, mask)
                img, mask = self._augment(img, mask)
        # Val/Test: return full image, no crop, no augmentation

        img = TF.to_tensor(img)
        img = TF.normalize(img, self.MEAN, self.STD)
        mask = torch.from_numpy(np.array(mask)).long()

        return img, mask


def collate_variable_size(batch):
    """Collate function for variable-size images (val/test).

    Returns list of (img, mask) instead of stacked tensors,
    since val/test images are full-resolution and may differ in size.
    """
    images, masks = zip(*batch)
    return list(images), list(masks)


def get_dataloaders(data_root, split_dir='splits/', crop_size=512,
                    batch_size=8, num_workers=4, copy_paste=True):
    """Create train and val dataloaders."""
    train_dataset = MoS2Dataset(
        data_root, split='train', split_dir=split_dir,
        crop_size=crop_size, augment=True, copy_paste=copy_paste
    )
    val_dataset = MoS2Dataset(
        data_root, split='val', split_dir=split_dir,
        crop_size=crop_size, augment=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    # Val: batch_size=1 because images are full-resolution
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
        collate_fn=collate_variable_size
    )
    return train_loader, val_loader
