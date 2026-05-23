import argparse

from config import (
    DEFAULT_IMAGE_SIZE,
    GENERATED_DIR,
    OVERLAP_RAW_DIR,
    RESIZED_DIR,
    SINGLE_CHROMOSOMES_DIR,
    SKELETON_DIR,
)
from utils_image import preprocess_folder


def parse_args():
    parser = argparse.ArgumentParser(description="Resize + grayscale all source images and optionally run Zhang-Suen skeleton QC.")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--use-skeleton", action="store_true", help="Enable Zhang-Suen skeleton QC.")
    parser.add_argument("--save-skeleton-debug", action="store_true", help="Save skeleton PNG files.")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"[PREPROCESS] image_size={args.image_size}, use_skeleton={args.use_skeleton}")

    rows_single = preprocess_folder(
        src_dir=SINGLE_CHROMOSOMES_DIR,
        dst_dir=RESIZED_DIR / "single_chromosomes",
        skeleton_dir=SKELETON_DIR / "single_chromosomes",
        image_size=args.image_size,
        use_skeleton=args.use_skeleton,
        save_skeleton_debug=args.save_skeleton_debug,
        csv_path=GENERATED_DIR / "single_chromosomes_skeleton_qc.csv",
    )

    rows_overlap = preprocess_folder(
        src_dir=OVERLAP_RAW_DIR,
        dst_dir=RESIZED_DIR / "overlap_raw",
        skeleton_dir=SKELETON_DIR / "overlap_raw",
        image_size=args.image_size,
        use_skeleton=args.use_skeleton,
        save_skeleton_debug=args.save_skeleton_debug,
        csv_path=GENERATED_DIR / "overlap_raw_skeleton_qc.csv",
    )

    print(f"[OK] resized single_chromosomes: {len(rows_single)}")
    print(f"[OK] resized overlap_raw: {len(rows_overlap)}")
    print("[OK] Preprocess done.")


if __name__ == "__main__":
    main()
