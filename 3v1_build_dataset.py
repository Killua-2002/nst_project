import argparse

import config
from src.synthetic_data import build_synthetic_dataset
from src.utils import list_images, print_data_status


def choose_single_dir(skip_resized: bool = False):
    if not skip_resized and len(list_images(config.RESIZED_SINGLE_CHROMOSOMES_DIR)) > 0:
        return config.RESIZED_SINGLE_CHROMOSOMES_DIR
    return config.get_single_dir()


def main():
    parser = argparse.ArgumentParser(description="Build synthetic labeled dataset from single chromosome images only.")
    parser.add_argument("--synthetic-per-class", type=int, default=config.DEFAULT_SYNTHETIC_PER_CLASS)
    parser.add_argument("--image-size", type=int, default=config.IMAGE_SIZE)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--strict-single", action="store_true", help="Use only images whose skeleton is exactly one unbranched line.")
    parser.add_argument("--use-skeleton", action="store_true", help="Run skeleton validation while loading single chromosomes.")
    parser.add_argument("--skeleton-pad", type=int, default=config.SKELETON_PAD)
    parser.add_argument("--skip-resized", action="store_true", help="Use source_data/single_chromosomes instead of generated_data/resized.")
    args = parser.parse_args()

    config.ensure_project_dirs()
    single_dir = choose_single_dir(args.skip_resized)
    print("Building synthetic dataset from:", single_dir)
    meta = build_synthetic_dataset(
        single_dir=single_dir,
        dataset_dir=config.DATASET_DIR,
        per_class=args.synthetic_per_class,
        image_size=args.image_size,
        seed=args.seed,
        strict_single=args.strict_single,
        run_skeleton=args.use_skeleton,
        skeleton_pad=args.skeleton_pad,
    )
    print(meta.groupby(["split", "label"]).size())
    print_data_status()


if __name__ == "__main__":
    main()
