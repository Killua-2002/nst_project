from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Dict, List

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import config
from src.image_processing import list_images, read_grayscale, resize_and_pad_gray


def clear_split_dirs() -> None:
    for split in ["train", "val", "test"]:
        for cls in config.CLASS_NAMES:
            folder = config.DATASET_DIR / split / cls
            if folder.exists():
                shutil.rmtree(folder)
            folder.mkdir(parents=True, exist_ok=True)
    unlabeled = config.DATASET_DIR / "unlabeled"
    if unlabeled.exists():
        shutil.rmtree(unlabeled)
    unlabeled.mkdir(parents=True, exist_ok=True)


def load_manual_labeled_samples() -> List[Dict]:
    """Optional layout: source_data/labeled/<class_name>/*.png"""
    samples = []
    for cls in config.CLASS_NAMES:
        folder = config.LABELED_DIR / cls
        for path in list_images(folder):
            samples.append({"path": path, "label": cls, "label_source": "manual_folder"})
    return samples


def load_rule_labeled_samples() -> List[Dict]:
    stats_path = config.GENERATED_DIR / "skeleton_stats.csv"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Missing {stats_path}. Run: python 2v1_preprocess_zhang_suen.py first"
        )
    df = pd.read_csv(stats_path)
    samples = []
    for _, row in df.iterrows():
        gen = str(row.get("generated_file", ""))
        label = str(row.get("predicted_rule_label", ""))
        if not gen or label not in config.CLASS_NAMES:
            continue
        path = config.GENERATED_DIR / "grayscale" / gen
        if path.exists():
            samples.append({"path": path, "label": label, "label_source": "skeleton_rule"})
    return samples


def copy_to_dataset(samples: List[Dict]) -> pd.DataFrame:
    labels = [s["label"] for s in samples]
    # Stratified split only if each class has enough samples.
    value_counts = pd.Series(labels).value_counts()
    stratify = labels if len(value_counts) > 1 and value_counts.min() >= 2 else None

    train_samples, tmp_samples = train_test_split(
        samples,
        test_size=(1.0 - config.TRAIN_RATIO),
        random_state=config.RANDOM_SEED,
        stratify=stratify,
    )
    tmp_labels = [s["label"] for s in tmp_samples]
    tmp_counts = pd.Series(tmp_labels).value_counts()
    tmp_stratify = tmp_labels if len(tmp_counts) > 1 and tmp_counts.min() >= 2 else None
    val_fraction_of_tmp = config.VAL_RATIO / max(config.VAL_RATIO + config.TEST_RATIO, 1e-8)
    val_samples, test_samples = train_test_split(
        tmp_samples,
        test_size=(1.0 - val_fraction_of_tmp),
        random_state=config.RANDOM_SEED,
        stratify=tmp_stratify,
    )

    all_rows = []
    for split, split_samples in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        for i, sample in enumerate(split_samples):
            src = Path(sample["path"])
            label = sample["label"]
            dst = config.DATASET_DIR / split / label / f"{src.stem}_{i:05d}.png"
            dst.parent.mkdir(parents=True, exist_ok=True)
            # Ensure all dataset images are saved as grayscale.
            img = read_grayscale(src)
            img = resize_and_pad_gray(img, config.IMG_SIZE)
            Image.fromarray(img).save(dst)
            all_rows.append({
                "split": split,
                "label": label,
                "path": str(dst),
                "source_path": str(src),
                "label_source": sample.get("label_source", "unknown"),
            })
    return pd.DataFrame(all_rows)


def copy_unlabeled_overlap_raw() -> int:
    count = 0
    for i, path in enumerate(list_images(config.OVERLAP_RAW_DIR)):
        try:
            img = read_grayscale(path)
            img = resize_and_pad_gray(img, config.IMG_SIZE)
            dst = config.DATASET_DIR / "unlabeled" / f"{path.stem}_{i:05d}.png"
            Image.fromarray(img).save(dst)
            count += 1
        except Exception:
            continue
    return count


def main() -> None:
    random.seed(config.RANDOM_SEED)
    clear_split_dirs()

    manual = load_manual_labeled_samples()
    if manual:
        print(f"Using {len(manual)} manually labeled samples from source_data/labeled/<class_name>.")
        samples = manual
    else:
        print("No manual labels found. Using skeleton-rule pseudo labels for supervised bootstrap.")
        samples = load_rule_labeled_samples()

    if len(samples) < 3:
        raise RuntimeError(
            "Not enough labeled / pseudo-labeled samples to build dataset. "
            "Add images to source_data/single_chromosomes and source_data/overlap_raw, "
            "or add manual labels under source_data/labeled/<class_name>."
        )

    split_df = copy_to_dataset(samples)
    split_csv = config.DATASET_DIR / "splits.csv"
    split_df.to_csv(split_csv, index=False)
    unlabeled_count = copy_unlabeled_overlap_raw()

    print(f"Saved dataset split: {split_csv}")
    print(split_df.groupby(["split", "label"]).size())
    print(f"Copied {unlabeled_count} unlabeled overlap_raw images to dataset/unlabeled")


if __name__ == "__main__":
    main()
