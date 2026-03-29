from __future__ import annotations

import random
import shutil
from pathlib import Path


SEED = 42
VAL_COUNT = 6


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_split(ids: list[str], split: str, src_root: Path, dst_root: Path) -> None:
    img_dir = dst_root / "img_dir" / split
    ann_dir = dst_root / "ann_dir" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    for sample_id in ids:
        src_img = src_root / "ori" / f"or{sample_id}.png"
        src_ann = src_root / "label_012" / f"col{sample_id}.png"
        dst_img = img_dir / f"{sample_id}.png"
        dst_ann = ann_dir / f"{sample_id}.png"

        if not src_img.exists():
            raise FileNotFoundError(f"Missing image: {src_img}")
        if not src_ann.exists():
            raise FileNotFoundError(f"Missing mask: {src_ann}")

        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_ann, dst_ann)


def main() -> None:
    src_root = Path("other_datav2")
    dst_root = Path("other_datav2_prepared")

    official_train = read_ids(src_root / "train.txt")
    official_test = read_ids(src_root / "test.txt")

    rng = random.Random(SEED)
    shuffled = official_train[:]
    rng.shuffle(shuffled)

    val_ids = sorted(shuffled[:VAL_COUNT])
    train_ids = sorted(shuffled[VAL_COUNT:])
    test_ids = sorted(official_test)

    ensure_clean_dir(dst_root)

    copy_split(train_ids, "train", src_root, dst_root)
    copy_split(val_ids, "val", src_root, dst_root)
    copy_split(test_ids, "test", src_root, dst_root)

    (dst_root / "train.txt").write_text("\n".join(train_ids) + "\n")
    (dst_root / "val.txt").write_text("\n".join(val_ids) + "\n")
    (dst_root / "test.txt").write_text("\n".join(test_ids) + "\n")

    print(f"Prepared dataset at: {dst_root}")
    print(f"train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    print("val ids:", ", ".join(val_ids))


if __name__ == "__main__":
    main()
