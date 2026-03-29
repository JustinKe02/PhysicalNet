"""
Generate 3-fold CV directories for other_datav2.
Fixed fold splits on official train=28, official test=7 held out.
"""
import os, shutil

SRC_IMG = 'other_datav2_prepared/img_dir'
SRC_ANN = 'other_datav2_prepared/ann_dir'
OUT_ROOT = 'other_datav2_cv'

# Fixed fold assignments
FOLDS = {
    0: ['00000', '00002', '00003', '00004', '00005', '00006', '00007', '00008', '00010'],
    1: ['00011', '00012', '00013', '00015', '00016', '00017', '00019', '00020', '00023'],
    2: ['00024', '00026', '00027', '00028', '00029', '00030', '00031', '00032', '00033', '00034'],
}

ALL_TRAIN = []
for ids in FOLDS.values():
    ALL_TRAIN.extend(ids)
ALL_TRAIN = sorted(set(ALL_TRAIN))
print(f'Total non-test images: {len(ALL_TRAIN)}')

# Find actual file extension
def find_img(name):
    for ext in ['.png', '.jpg']:
        for split in ['train', 'val']:
            p = os.path.join(SRC_IMG, split, name + ext)
            if os.path.exists(p):
                return p, ext
    return None, None

for fold_id, val_ids in FOLDS.items():
    train_ids = [x for x in ALL_TRAIN if x not in val_ids]
    fold_dir = os.path.join(OUT_ROOT, f'fold{fold_id}')

    for split, ids in [('train', train_ids), ('val', val_ids)]:
        img_out = os.path.join(fold_dir, 'img_dir', split)
        ann_out = os.path.join(fold_dir, 'ann_dir', split)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(ann_out, exist_ok=True)

        for name in ids:
            img_src, ext = find_img(name)
            if img_src is None:
                print(f'  WARNING: {name} not found!')
                continue
            shutil.copy2(img_src, os.path.join(img_out, name + ext))

            ann_src = None
            for s in ['train', 'val']:
                p = os.path.join(SRC_ANN, s, name + '.png')
                if os.path.exists(p):
                    ann_src = p
                    break
            if ann_src:
                shutil.copy2(ann_src, os.path.join(ann_out, name + '.png'))

    print(f'Fold {fold_id}: train={len(train_ids)}, val={len(val_ids)}')

print('\nDone!')
