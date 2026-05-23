from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import config
from src.skeleton_utils import analyze_skeleton, binarize_chromosome, read_grayscale, skeletonize_with_padding
from src.utils import clean_dir, list_images, set_seed


def crop_foreground(gray: np.ndarray) -> Image.Image:
    binary = binarize_chromosome(gray)
    ys, xs = np.where(binary)
    if len(xs) == 0 or len(ys) == 0:
        return Image.fromarray(gray).convert("L")
    margin = 4
    x1, x2 = max(0, xs.min() - margin), min(gray.shape[1], xs.max() + margin + 1)
    y1, y2 = max(0, ys.min() - margin), min(gray.shape[0], ys.max() + margin + 1)
    crop = gray[y1:y2, x1:x2]
    return Image.fromarray(crop).convert("L")


def load_single_chromosomes(
    single_dir: Path,
    image_size: int = config.IMAGE_SIZE,
    strict_single: bool = False,
    run_skeleton: bool = config.RUN_SKELETON_DEFAULT,
    skeleton_pad: int = config.SKELETON_PAD,
) -> List[Image.Image]:
    """Load NST đơn images for synthetic generation.

    Skeleton validation is skipped by default for speed. It is automatically enabled
    when strict_single=True because that rule needs endpoints/junctions.
    """
    if strict_single:
        run_skeleton = True

    images = []
    stats_rows = []
    for path in tqdm(list_images(single_dir), desc="load single chromosomes"):
        # Resize here too, so generation is consistent even if preprocessing was skipped.
        gray = read_grayscale(path, image_size=image_size)

        if run_skeleton:
            binary = binarize_chromosome(gray)
            skel = skeletonize_with_padding(binary, pad=skeleton_pad)
            stats = analyze_skeleton(binary, skel)
            stats["file"] = str(path)
            stats_rows.append(stats)
            if strict_single and not stats.get("valid_single_line", False):
                continue

        images.append(crop_foreground(gray))

    out_csv = config.GENERATED_DIR / "single_chromosome_validation.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if stats_rows:
        pd.DataFrame(stats_rows).to_csv(out_csv, index=False)
    else:
        pd.DataFrame([{"skeleton_mode": "skipped", "source": str(single_dir)}]).to_csv(out_csv, index=False)
    return images


def paste_chromosome(canvas: np.ndarray, obj: Image.Image, center: Tuple[int, int], angle: float, scale: float) -> np.ndarray:
    """Paste one grayscale chromosome onto white canvas by taking darker pixels."""
    obj = obj.convert("L")
    w, h = obj.size
    new_w = max(8, int(w * scale))
    new_h = max(8, int(h * scale))
    obj = obj.resize((new_w, new_h), Image.Resampling.BICUBIC)
    obj = obj.rotate(angle, expand=True, fillcolor=255, resample=Image.Resampling.BICUBIC)
    arr = np.array(obj, dtype=np.uint8)
    mask = arr < 245
    x0 = int(center[0] - arr.shape[1] // 2)
    y0 = int(center[1] - arr.shape[0] // 2)

    x1 = max(0, x0)
    y1 = max(0, y0)
    x2 = min(canvas.shape[1], x0 + arr.shape[1])
    y2 = min(canvas.shape[0], y0 + arr.shape[0])
    if x2 <= x1 or y2 <= y1:
        return canvas

    sx1 = x1 - x0
    sy1 = y1 - y0
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)
    region = canvas[y1:y2, x1:x2]
    arr_crop = arr[sy1:sy2, sx1:sx2]
    mask_crop = mask[sy1:sy2, sx1:sx2]
    region[mask_crop] = np.minimum(region[mask_crop], arr_crop[mask_crop])
    canvas[y1:y2, x1:x2] = region
    return canvas


def generate_pair(single_images: List[Image.Image], cls: str, image_size: int = config.IMAGE_SIZE) -> Image.Image:
    if len(single_images) < 2:
        raise ValueError("Need at least 2 single chromosome images to generate synthetic pairs.")
    obj1, obj2 = random.sample(single_images, 2)
    canvas = np.full((image_size, image_size), 255, dtype=np.uint8)
    cx = image_size // 2 + random.randint(-12, 12)
    cy = image_size // 2 + random.randint(-12, 12)

    if cls == "overlapping":
        center1 = (cx + random.randint(-5, 5), cy + random.randint(-5, 5))
        center2 = (cx + random.randint(-5, 5), cy + random.randint(-5, 5))
        angle1 = random.uniform(-70, 70)
        angle2 = angle1 + random.choice([-1, 1]) * random.uniform(45, 100)
    elif cls == "touching":
        offset = random.randint(35, 55)
        theta = random.uniform(0, 2 * np.pi)
        dx, dy = int(np.cos(theta) * offset), int(np.sin(theta) * offset)
        center1 = (cx - dx, cy - dy)
        center2 = (cx + dx, cy + dy)
        angle1 = random.uniform(-80, 80)
        angle2 = angle1 + random.uniform(-25, 25)
    else:  # touching_overlapping
        offset = random.randint(12, 28)
        theta = random.uniform(0, 2 * np.pi)
        dx, dy = int(np.cos(theta) * offset), int(np.sin(theta) * offset)
        center1 = (cx - dx, cy - dy)
        center2 = (cx + dx, cy + dy)
        angle1 = random.uniform(-80, 80)
        angle2 = angle1 + random.choice([-1, 1]) * random.uniform(25, 70)

    scale1 = random.uniform(0.75, 1.15)
    scale2 = random.uniform(0.75, 1.15)
    canvas = paste_chromosome(canvas, obj1, center1, angle1, scale1)
    canvas = paste_chromosome(canvas, obj2, center2, angle2, scale2)

    img = Image.fromarray(canvas, mode="L")
    if random.random() < 0.4:
        arr = np.array(img).astype(np.int16)
        noise = np.random.normal(0, random.uniform(1.5, 4.0), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")
    return img


def split_name(index: int, per_class: int, train_ratio: float = 0.8, val_ratio: float = 0.1) -> str:
    train_end = int(per_class * train_ratio)
    val_end = train_end + int(per_class * val_ratio)
    if index < train_end:
        return "train"
    if index < val_end:
        return "val"
    return "test"


def build_synthetic_dataset(
    single_dir: Path,
    dataset_dir: Path,
    per_class: int = config.DEFAULT_SYNTHETIC_PER_CLASS,
    image_size: int = config.IMAGE_SIZE,
    seed: int = config.RANDOM_SEED,
    strict_single: bool = False,
    run_skeleton: bool = config.RUN_SKELETON_DEFAULT,
    skeleton_pad: int = config.SKELETON_PAD,
) -> pd.DataFrame:
    set_seed(seed)
    single_images = load_single_chromosomes(
        single_dir,
        image_size=image_size,
        strict_single=strict_single,
        run_skeleton=run_skeleton,
        skeleton_pad=skeleton_pad,
    )
    if len(single_images) < 2:
        raise RuntimeError(
            f"Need at least 2 usable single chromosome images in {single_dir}. "
            "Put more images into source_data/single_chromosomes."
        )

    for split in ["train", "val", "test"]:
        clean_dir(Path(dataset_dir) / split)
        for cls in config.CLASSES:
            (Path(dataset_dir) / split / cls).mkdir(parents=True, exist_ok=True)

    rows = []
    for cls in config.CLASSES:
        for i in tqdm(range(per_class), desc=f"Generate {cls}"):
            split = split_name(i, per_class)
            img = generate_pair(single_images, cls, image_size=image_size)
            filename = f"{cls}_{i:06d}.png"
            out_path = Path(dataset_dir) / split / cls / filename
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(out_path)
            rows.append({"split": split, "label": cls, "file": str(out_path)})
    meta = pd.DataFrame(rows)
    meta.to_csv(Path(dataset_dir) / "synthetic_metadata.csv", index=False)
    return meta
