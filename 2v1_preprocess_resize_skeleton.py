from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.morphology import skeletonize
from tqdm import tqdm

from config import CFG, resolve_single_chromosome_dir
from utils_image import ensure_dir, list_images, otsu_foreground_mask, read_gray, resize_square


def zhang_suen_skeleton(binary_mask: np.ndarray) -> np.ndarray:
    padded = np.pad(binary_mask.astype(bool), pad_width=4, mode="constant", constant_values=False)
    try:
        skel = skeletonize(padded, method="zhang")
    except TypeError:
        skel = skeletonize(padded)
    skel = skel[4:-4, 4:-4]
    return skel.astype(np.uint8)


def skeleton_stats(skel: np.ndarray) -> Dict[str, int | str]:
    sk = skel.astype(np.uint8)
    h, w = sk.shape
    endpoints = 0
    branches = 0
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if sk[y, x] == 0:
                continue
            n = int(sk[y-1:y+2, x-1:x+2].sum()) - 1
            if n == 1:
                endpoints += 1
            elif n >= 3:
                branches += 1
    status = "single_path_no_branch" if endpoints == 2 and branches == 0 else "complex_or_overlap"
    return {"endpoints": endpoints, "branch_points": branches, "status": status}


def preprocess_folder(src_dir: Path, dst_dir: Path, image_size: int, use_skeleton: bool, save_skeleton_debug: bool, report_name: str) -> pd.DataFrame:
    ensure_dir(dst_dir)
    if save_skeleton_debug:
        ensure_dir(dst_dir.parent / f"{dst_dir.name}_skeleton_debug")

    rows = []
    paths = list_images(src_dir)
    if not paths:
        print(f"Warning: no images found in {src_dir}")
        return pd.DataFrame()

    for p in tqdm(paths, desc=f"preprocess {src_dir.name}"):
        gray = read_gray(p)
        resized = resize_square(gray, image_size)
        out_path = dst_dir / f"{p.stem}.png"
        Image.fromarray(resized).save(out_path)

        row = {
            "filename": out_path.name,
            "source_path": str(p),
            "resized_path": str(out_path),
            "image_size": image_size,
            "skeleton_used": bool(use_skeleton),
            "skeleton_status": "skipped",
            "endpoints": -1,
            "branch_points": -1,
        }

        if use_skeleton:
            mask = otsu_foreground_mask(resized)
            skel = zhang_suen_skeleton(mask)
            stats = skeleton_stats(skel)
            row.update(stats)
            if save_skeleton_debug:
                skel_path = dst_dir.parent / f"{dst_dir.name}_skeleton_debug" / out_path.name
                Image.fromarray((skel * 255).astype(np.uint8)).save(skel_path)
                row["skeleton_path"] = str(skel_path)
        rows.append(row)

    df = pd.DataFrame(rows)
    ensure_dir(CFG.RESULT_DIR)
    report_path = CFG.RESULT_DIR / report_name
    df.to_csv(report_path, index=False)
    print("Saved report:", report_path)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Grayscale + resize all images. Optional Zhang-Suen skeleton debug.")
    parser.add_argument("--image-size", type=int, default=CFG.IMAGE_SIZE)
    parser.add_argument("--use-skeleton", action="store_true")
    parser.add_argument("--save-skeleton-debug", action="store_true")
    args = parser.parse_args()

    cfg = CFG()
    single_dir = resolve_single_chromosome_dir(cfg.SOURCE_DIR)

    preprocess_folder(
        cfg.OVERLAP_RAW_DIR,
        cfg.RESIZED_DIR / "overlap_raw",
        args.image_size,
        args.use_skeleton,
        args.save_skeleton_debug,
        "preprocess_overlap_raw.csv",
    )
    preprocess_folder(
        single_dir,
        cfg.RESIZED_DIR / "single_chromosomes",
        args.image_size,
        args.use_skeleton,
        args.save_skeleton_debug,
        "preprocess_single_chromosomes.csv",
    )


if __name__ == "__main__":
    main()
