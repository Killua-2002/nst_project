import argparse
import shutil
from pathlib import Path

from config import DATASET_DIR, DEFAULT_IMAGE_SIZE, RESIZED_DIR, SINGLE_CHROMOSOMES_DIR
from utils_dataset import generate_synthetic_split
from utils_image import ensure_dir, list_images


def parse_args():
    p = argparse.ArgumentParser(description="Generate synthetic A/B/C segmentation masks from single chromosome images.")
    p.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    p.add_argument("--train-count", type=int, default=600)
    p.add_argument("--val-count", type=int, default=120)
    p.add_argument("--test-count", type=int, default=120)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--strict-skeleton", action="store_true", help="Only use single chromosomes with clean one-line skeleton.")
    p.add_argument("--clear-dataset", action="store_true", help="Remove previous synthetic dataset before generating.")
    return p.parse_args()


def _clear_split(split: str):
    for sub in ["images", "labels"]:
        folder = DATASET_DIR / split / sub
        if folder.exists():
            shutil.rmtree(folder)
        ensure_dir(folder)


def main():
    args = parse_args()
    if args.clear_dataset:
        for split in ["train", "val", "test"]:
            _clear_split(split)

    # Prefer resized single data after preprocessing; fallback to source.
    single_dir = RESIZED_DIR / "single_chromosomes"
    if len(list_images(single_dir)) < 2:
        single_dir = SINGLE_CHROMOSOMES_DIR

    print(f"[SYNTH] single_dir={single_dir}")
    print(f"[SYNTH] strict_skeleton={args.strict_skeleton}")

    counts = {
        "train": args.train_count,
        "val": args.val_count,
        "test": args.test_count,
    }
    total = 0
    for i, (split, count) in enumerate(counts.items()):
        made = generate_synthetic_split(
            single_dir=single_dir,
            out_images=DATASET_DIR / split / "images",
            out_labels=DATASET_DIR / split / "labels",
            count=count,
            image_size=args.image_size,
            seed=args.seed + i * 100,
            strict_skeleton=args.strict_skeleton,
        )
        print(f"[OK] {split}: {made}/{count}")
        total += made

    print(f"[OK] Synthetic A/B/C segmentation dataset generated: {total} images")


if __name__ == "__main__":
    main()
