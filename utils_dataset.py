from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from utils_image import clean_mask, foreground_mask, list_images, read_gray, save_gray, save_label, zhang_suen_skeleton, analyze_skeleton, ensure_dir


def _random_transform_mask(mask: np.ndarray, image_size: int, rng: random.Random) -> np.ndarray:
    """Scale, rotate and translate one single chromosome mask to canvas."""
    mask = mask.astype(np.uint8) * 255
    h, w = mask.shape

    scale = rng.uniform(0.65, 1.05)
    new_size = max(16, int(image_size * scale))
    resized = cv2.resize(mask, (new_size, new_size), interpolation=cv2.INTER_NEAREST)

    angle = rng.uniform(-70, 70)
    M = cv2.getRotationMatrix2D((new_size / 2, new_size / 2), angle, 1.0)
    rotated = cv2.warpAffine(resized, M, (new_size, new_size), flags=cv2.INTER_NEAREST, borderValue=0)

    canvas = np.zeros((image_size, image_size), dtype=np.uint8)

    # Place near center to increase overlap probability.
    max_shift = image_size // 7
    center_x = image_size // 2 + rng.randint(-max_shift, max_shift)
    center_y = image_size // 2 + rng.randint(-max_shift, max_shift)

    x0 = int(center_x - new_size // 2)
    y0 = int(center_y - new_size // 2)

    src_x0 = max(0, -x0)
    src_y0 = max(0, -y0)
    dst_x0 = max(0, x0)
    dst_y0 = max(0, y0)

    copy_w = min(new_size - src_x0, image_size - dst_x0)
    copy_h = min(new_size - src_y0, image_size - dst_y0)
    if copy_w > 0 and copy_h > 0:
        canvas[dst_y0:dst_y0 + copy_h, dst_x0:dst_x0 + copy_w] = rotated[src_y0:src_y0 + copy_h, src_x0:src_x0 + copy_w]

    return canvas > 0


def _make_synthetic_image(mask_a: np.ndarray, mask_b: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = mask_a.shape
    img = np.full((h, w), rng.randint(230, 255), dtype=np.float32)
    noise = np.random.default_rng(rng.randint(0, 10**9)).normal(0, 5, size=(h, w))
    img += noise

    val_a = rng.randint(35, 90)
    val_b = rng.randint(45, 100)
    img[mask_a] = val_a + noise[mask_a]
    img[mask_b] = val_b + noise[mask_b]

    overlap = mask_a & mask_b
    img[overlap] = min(val_a, val_b) - rng.randint(5, 25) + noise[overlap]

    # Slight blur similar to microscopy crop.
    img = cv2.GaussianBlur(img.clip(0, 255).astype(np.uint8), (3, 3), 0)
    return img


def _label_from_masks(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    label = np.zeros(mask_a.shape, dtype=np.uint8)
    c = mask_a & mask_b
    label[np.logical_and(mask_a, ~mask_b)] = 1
    label[np.logical_and(mask_b, ~mask_a)] = 2
    label[c] = 3
    return label


def _valid_single_mask(mask: np.ndarray, strict_skeleton: bool) -> bool:
    area = int(mask.sum())
    if area < 30:
        return False
    if not strict_skeleton:
        return True
    # For a clean single chromosome, skeleton should be one unbranched path.
    info = analyze_skeleton(mask, already_skeleton=False)
    return info["status"] == "valid_single_path"


def generate_synthetic_split(
    single_dir: Path,
    out_images: Path,
    out_labels: Path,
    count: int,
    image_size: int,
    seed: int,
    strict_skeleton: bool = False,
    max_trials_multiplier: int = 80,
) -> int:
    ensure_dir(out_images)
    ensure_dir(out_labels)

    src_images = list_images(single_dir)
    if len(src_images) < 2:
        raise RuntimeError(
            f"Need at least 2 single chromosome images in {single_dir}. "
            "Put data in source_data/single_chromosomes first."
        )

    rng = random.Random(seed)
    made = 0
    trials = 0
    max_trials = max(count * max_trials_multiplier, count + 100)

    while made < count and trials < max_trials:
        trials += 1
        p1, p2 = rng.sample(src_images, 2)
        g1 = read_gray(p1, image_size=image_size)
        g2 = read_gray(p2, image_size=image_size)

        m1 = clean_mask(foreground_mask(g1), min_size=max(10, image_size // 12))
        m2 = clean_mask(foreground_mask(g2), min_size=max(10, image_size // 12))
        if not _valid_single_mask(m1, strict_skeleton) or not _valid_single_mask(m2, strict_skeleton):
            continue

        a = _random_transform_mask(m1, image_size, rng)
        b = _random_transform_mask(m2, image_size, rng)

        area_a = int(a.sum())
        area_b = int(b.sum())
        overlap_area = int((a & b).sum())
        if min(area_a, area_b) < 30:
            continue

        overlap_ratio = overlap_area / max(1, min(area_a, area_b))
        # Need true overlapping for A/B/C segmentation.
        if overlap_ratio < 0.03 or overlap_ratio > 0.65:
            continue

        img = _make_synthetic_image(a, b, rng)
        label = _label_from_masks(a, b)

        name = f"synthetic_{made:06d}.png"
        save_gray(out_images / name, img)
        save_label(out_labels / name, label)
        made += 1

    if made < count:
        print(f"[WARN] Requested {count} synthetic images, generated {made}. Try strict_skeleton=False or lower count.")
    return made
