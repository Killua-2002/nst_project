from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

import config
from src.image_processing import (
    classify_by_skeleton,
    list_images,
    read_grayscale,
    resize_and_pad_gray,
    save_debug_images,
    to_binary,
    zhang_suen_skeleton,
)


def process_group(group_name: str, folder: Path):
    rows = []
    images = list_images(folder)
    print(f"[{group_name}] found {len(images)} images")
    for path in tqdm(images, desc=f"preprocess {group_name}"):
        try:
            gray = read_grayscale(path)
            gray = resize_and_pad_gray(gray, config.IMG_SIZE)
            binary = to_binary(gray, min_object_area=config.MIN_OBJECT_AREA)
            skeleton = zhang_suen_skeleton(binary, pad=config.SKELETON_PAD)
            stats = classify_by_skeleton(binary, skeleton, min_skeleton_pixels=config.MIN_SKELETON_PIXELS)

            rel_name = f"{group_name}__{path.stem}.png"
            Image.fromarray(gray).save(config.GENERATED_DIR / "grayscale" / rel_name)
            Image.fromarray((binary * 255).astype("uint8")).save(config.GENERATED_DIR / "binary" / rel_name)
            Image.fromarray((skeleton * 255).astype("uint8")).save(config.GENERATED_DIR / "skeleton" / rel_name)
            save_debug_images(gray, binary, skeleton, config.GENERATED_DIR / "overlay" / f"{group_name}__{path.stem}")

            rows.append({
                "original_path": str(path),
                "generated_file": rel_name,
                "source_group": group_name,
                "width": int(gray.shape[1]),
                "height": int(gray.shape[0]),
                **stats,
            })
        except Exception as exc:
            rows.append({
                "original_path": str(path),
                "generated_file": "",
                "source_group": group_name,
                "error": repr(exc),
            })
    return rows


def main() -> None:
    config.GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    rows.extend(process_group("overlap_raw", config.OVERLAP_RAW_DIR))
    rows.extend(process_group("single_chromosomes", config.SINGLE_CHR_DIR))

    stats_path = config.GENERATED_DIR / "skeleton_stats.csv"
    df = pd.DataFrame(rows)
    df.to_csv(stats_path, index=False)
    print(f"Saved skeleton stats: {stats_path}")

    if not df.empty and "predicted_rule_label" in df.columns:
        print("Rule-label summary:")
        print(df["predicted_rule_label"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
