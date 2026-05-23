import argparse

import pandas as pd

import config
from src.skeleton_utils import process_folder
from src.utils import list_images


def main():
    parser = argparse.ArgumentParser(description="Preprocess: grayscale + resize all images, optional Zhang-Suen skeleton.")
    parser.add_argument("--image-size", type=int, default=config.IMAGE_SIZE)
    parser.add_argument("--pad", type=int, default=config.SKELETON_PAD, help="Padding border before Zhang-Suen thinning.")
    parser.add_argument("--use-skeleton", action="store_true", help="Enable binary + Zhang-Suen skeleton analysis.")
    parser.add_argument("--save-skeleton-debug", action="store_true", help="Save gray/binary/skeleton/overlay debug images.")
    args = parser.parse_args()

    config.ensure_project_dirs()
    single_dir = config.get_single_dir()

    print("Processing overlap_raw:", config.OVERLAP_RAW_DIR)
    overlap_df = process_folder(
        config.OVERLAP_RAW_DIR,
        config.GENERATED_DIR / "overlap_raw",
        image_size=args.image_size,
        pad=args.pad,
        run_skeleton=args.use_skeleton,
        save_skeleton_debug=args.save_skeleton_debug,
        resized_dir=config.RESIZED_OVERLAP_RAW_DIR,
    )
    print("Processing single chromosomes:", single_dir)
    single_df = process_folder(
        single_dir,
        config.GENERATED_DIR / "single_chromosomes",
        image_size=args.image_size,
        pad=args.pad,
        run_skeleton=args.use_skeleton,
        save_skeleton_debug=args.save_skeleton_debug,
        resized_dir=config.RESIZED_SINGLE_CHROMOSOMES_DIR,
    )

    overlap_df["source"] = "overlap_raw"
    single_df["source"] = "single_chromosomes"
    all_df = pd.concat([overlap_df, single_df], ignore_index=True)
    all_df.to_csv(config.GENERATED_DIR / "all_preprocess_stats.csv", index=False)

    print("Done.")
    print("original overlap_raw images:", len(list_images(config.OVERLAP_RAW_DIR)))
    print("original single_chromosome images:", len(list_images(single_dir)))
    print("resized overlap_raw images:", len(list_images(config.RESIZED_OVERLAP_RAW_DIR)))
    print("resized single_chromosome images:", len(list_images(config.RESIZED_SINGLE_CHROMOSOMES_DIR)))
    print("Stats saved to generated_data/all_preprocess_stats.csv")


if __name__ == "__main__":
    main()
