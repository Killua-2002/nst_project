import argparse
from pathlib import Path

import pandas as pd

import config
from src.skeleton_utils import process_folder
from src.utils import list_images


def main():
    parser = argparse.ArgumentParser(description="Grayscale + binary + Zhang-Suen skeleton preprocessing.")
    parser.add_argument("--image-size", type=int, default=config.IMAGE_SIZE)
    parser.add_argument("--pad", type=int, default=8, help="Padding border before Zhang-Suen thinning.")
    args = parser.parse_args()

    config.ensure_project_dirs()
    single_dir = config.get_single_dir()
    print("Processing overlap_raw:", config.OVERLAP_RAW_DIR)
    overlap_df = process_folder(
        config.OVERLAP_RAW_DIR,
        config.GENERATED_DIR / "overlap_raw",
        image_size=args.image_size,
        pad=args.pad,
    )
    print("Processing single chromosomes:", single_dir)
    single_df = process_folder(
        single_dir,
        config.GENERATED_DIR / "single_chromosomes",
        image_size=args.image_size,
        pad=args.pad,
    )

    overlap_df["source"] = "overlap_raw"
    single_df["source"] = "single_chromosomes"
    all_df = pd.concat([overlap_df, single_df], ignore_index=True)
    all_df.to_csv(config.GENERATED_DIR / "all_skeleton_stats.csv", index=False)

    print("Done.")
    print("overlap_raw images:", len(list_images(config.OVERLAP_RAW_DIR)))
    print("single_chromosome images:", len(list_images(single_dir)))
    print("Stats saved to generated_data/all_skeleton_stats.csv")


if __name__ == "__main__":
    main()
