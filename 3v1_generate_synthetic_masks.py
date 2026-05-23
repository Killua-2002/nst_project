from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from config import CFG
from utils_image import (
    crop_object,
    ensure_dir,
    list_images,
    otsu_foreground_mask,
    paste_mask,
    paste_object,
    read_gray,
    resize_object,
    rotate_image_and_mask,
    save_label_and_masks,
    seed_everything,
)


def reset_split_dirs(dataset_dir: Path, split: str) -> None:
    for sub in ["images", "labels", "masks_A", "masks_B", "masks_C"]:
        d = dataset_dir / split / sub
        if d.exists():
            shutil.rmtree(d)
        ensure_dir(d)


def prepare_object(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    gray = read_gray(path)
    mask = otsu_foreground_mask(gray)
    obj, obj_mask = crop_object(gray, mask, pad=5)
    return obj, obj_mask


def create_pair_sample(obj_a: Tuple[np.ndarray, np.ndarray], obj_b: Tuple[np.ndarray, np.ndarray], image_size: int, rng: random.Random) -> Tuple[np.ndarray, np.ndarray]:
    base_a, mask_a0 = obj_a
    base_b, mask_b0 = obj_b

    for _ in range(80):
        target_a = rng.randint(int(image_size * 0.38), int(image_size * 0.68))
        target_b = rng.randint(int(image_size * 0.38), int(image_size * 0.68))
        a_img, a_mask = resize_object(base_a, mask_a0, target_a)
        b_img, b_mask = resize_object(base_b, mask_b0, target_b)

        # Define the semantic order: A is mostly horizontal, B is mostly vertical.
        a_angle = rng.uniform(-25, 25)
        b_angle = rng.uniform(65, 115)
        a_img, a_mask = rotate_image_and_mask(a_img, a_mask, a_angle, bg_value=255)
        b_img, b_mask = rotate_image_and_mask(b_img, b_mask, b_angle, bg_value=255)

        center = image_size // 2
        cxa = center + rng.randint(-18, 18)
        cya = center + rng.randint(-18, 18)
        cxb = center + rng.randint(-18, 18)
        cyb = center + rng.randint(-18, 18)

        mA = np.zeros((image_size, image_size), dtype=np.uint8)
        mB = np.zeros_like(mA)
        mA = paste_mask(mA, a_mask, (cxa, cya))
        mB = paste_mask(mB, b_mask, (cxb, cyb))
        overlap = (mA > 0) & (mB > 0)

        area_a = int(mA.sum())
        area_b = int(mB.sum())
        area_c = int(overlap.sum())
        min_area = max(30, int(image_size * image_size * 0.001))
        if area_a > min_area and area_b > min_area and area_c > min_area:
            canvas = np.full((image_size, image_size), 255, dtype=np.uint8)
            canvas = paste_object(canvas, a_img, a_mask, (cxa, cya))
            canvas = paste_object(canvas, b_img, b_mask, (cxb, cyb))
            # Slight blur/noise to imitate real raw data.
            if rng.random() < 0.5:
                canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
            noise = rng.normalvariate(0, 1)
            if abs(noise) > 0:
                arr = canvas.astype(np.float32) + np.random.normal(0, rng.uniform(0.5, 2.5), canvas.shape)
                canvas = np.clip(arr, 0, 255).astype(np.uint8)

            label = np.zeros((image_size, image_size), dtype=np.uint8)
            label[(mA > 0) & ~(mB > 0)] = 1
            label[(mB > 0) & ~(mA > 0)] = 2
            label[overlap] = 3
            return canvas, label

    # Fallback: force centers equal if overlap was hard to create.
    canvas = np.full((image_size, image_size), 255, dtype=np.uint8)
    a_img, a_mask = resize_object(base_a, mask_a0, int(image_size * 0.55))
    b_img, b_mask = resize_object(base_b, mask_b0, int(image_size * 0.55))
    a_img, a_mask = rotate_image_and_mask(a_img, a_mask, 0, bg_value=255)
    b_img, b_mask = rotate_image_and_mask(b_img, b_mask, 90, bg_value=255)
    center_xy = (image_size // 2, image_size // 2)
    mA = paste_mask(np.zeros((image_size, image_size), dtype=np.uint8), a_mask, center_xy)
    mB = paste_mask(np.zeros((image_size, image_size), dtype=np.uint8), b_mask, center_xy)
    canvas = paste_object(canvas, a_img, a_mask, center_xy)
    canvas = paste_object(canvas, b_img, b_mask, center_xy)
    label = np.zeros((image_size, image_size), dtype=np.uint8)
    label[(mA > 0) & ~(mB > 0)] = 1
    label[(mB > 0) & ~(mA > 0)] = 2
    label[(mA > 0) & (mB > 0)] = 3
    return canvas, label


def generate_split(split: str, count: int, objects: List[Tuple[np.ndarray, np.ndarray]], cfg: CFG, image_size: int, rng: random.Random) -> pd.DataFrame:
    reset_split_dirs(cfg.DATASET_DIR, split)
    rows = []
    for i in tqdm(range(count), desc=f"generate {split}"):
        obj_a = rng.choice(objects)
        obj_b = rng.choice(objects)
        image, label = create_pair_sample(obj_a, obj_b, image_size=image_size, rng=rng)
        name = f"{split}_{i:06d}.png"
        img_path = cfg.DATASET_DIR / split / "images" / name
        lab_path = cfg.DATASET_DIR / split / "labels" / name
        a_path = cfg.DATASET_DIR / split / "masks_A" / name
        b_path = cfg.DATASET_DIR / split / "masks_B" / name
        c_path = cfg.DATASET_DIR / split / "masks_C" / name
        Image.fromarray(image).save(img_path)
        save_label_and_masks(label, lab_path, a_path, b_path, c_path)
        rows.append({
            "split": split,
            "filename": name,
            "image": str(img_path),
            "label": str(lab_path),
            "mask_A": str(a_path),
            "mask_B": str(b_path),
            "mask_C": str(c_path),
            "pixels_A": int(((label == 1) | (label == 3)).sum()),
            "pixels_B": int(((label == 2) | (label == 3)).sum()),
            "pixels_C": int((label == 3).sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic segmentation masks A/B/C from single chromosome images.")
    parser.add_argument("--image-size", type=int, default=CFG.IMAGE_SIZE)
    parser.add_argument("--train", type=int, default=CFG.SYNTHETIC_TRAIN)
    parser.add_argument("--val", type=int, default=CFG.SYNTHETIC_VAL)
    parser.add_argument("--test", type=int, default=CFG.SYNTHETIC_TEST)
    parser.add_argument("--seed", type=int, default=CFG.SEED)
    args = parser.parse_args()

    seed_everything(args.seed)
    rng = random.Random(args.seed)
    cfg = CFG()
    resized_single = cfg.RESIZED_DIR / "single_chromosomes"
    paths = list_images(resized_single)
    if not paths:
        raise FileNotFoundError(
            f"No preprocessed single chromosomes found in {resized_single}. "
            "Run 2v1_preprocess_resize_skeleton.py first."
        )

    print("Preparing single chromosome objects...")
    objects = [prepare_object(p) for p in tqdm(paths)]
    objects = [(img, mask) for img, mask in objects if int(mask.sum()) > 30]
    if len(objects) < 2:
        raise RuntimeError("Need at least 2 valid single chromosome objects to generate synthetic overlaps.")

    all_rows = []
    for split, count in [("train", args.train), ("val", args.val), ("test", args.test)]:
        all_rows.append(generate_split(split, count, objects, cfg, args.image_size, rng))
    df = pd.concat(all_rows, ignore_index=True)
    ensure_dir(cfg.RESULT_DIR)
    csv_path = cfg.RESULT_DIR / "synthetic_dataset_manifest.csv"
    df.to_csv(csv_path, index=False)
    print("Saved:", csv_path)
    print(df.groupby("split").size())


if __name__ == "__main__":
    main()
