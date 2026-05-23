import argparse

import config
from src.synthetic_data import build_synthetic_dataset
from src.utils import print_data_status


def main():
    parser = argparse.ArgumentParser(description="Build synthetic labeled dataset from single chromosome images only.")
    parser.add_argument("--synthetic-per-class", type=int, default=config.DEFAULT_SYNTHETIC_PER_CLASS)
    parser.add_argument("--image-size", type=int, default=config.IMAGE_SIZE)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--strict-single", action="store_true", help="Use only images whose skeleton is exactly one unbranched line.")
    args = parser.parse_args()

    config.ensure_project_dirs()
    single_dir = config.get_single_dir()
    print("Building synthetic dataset from:", single_dir)
    meta = build_synthetic_dataset(
        single_dir=single_dir,
        dataset_dir=config.DATASET_DIR,
        per_class=args.synthetic_per_class,
        image_size=args.image_size,
        seed=args.seed,
        strict_single=args.strict_single,
    )
    print(meta.groupby(["split", "label"]).size())
    print_data_status()


if __name__ == "__main__":
    main()
