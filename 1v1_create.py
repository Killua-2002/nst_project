from pathlib import Path
import argparse
from config import CFG, resolve_single_chromosome_dir
from utils_image import ensure_dir


def create_folders() -> None:
    cfg = CFG()
    cfg.SINGLE_CHR_DIR = resolve_single_chromosome_dir(cfg.SOURCE_DIR)

    folders = [
        cfg.SOURCE_DIR,
        cfg.OVERLAP_RAW_DIR,
        cfg.SINGLE_CHR_DIR,
        cfg.GENERATED_DIR,
        cfg.RESIZED_DIR / "overlap_raw",
        cfg.RESIZED_DIR / "single_chromosomes",
        cfg.DATASET_DIR,
        cfg.DATASET_DIR / "train" / "images",
        cfg.DATASET_DIR / "train" / "labels",
        cfg.DATASET_DIR / "train" / "masks_A",
        cfg.DATASET_DIR / "train" / "masks_B",
        cfg.DATASET_DIR / "train" / "masks_C",
        cfg.DATASET_DIR / "val" / "images",
        cfg.DATASET_DIR / "val" / "labels",
        cfg.DATASET_DIR / "val" / "masks_A",
        cfg.DATASET_DIR / "val" / "masks_B",
        cfg.DATASET_DIR / "val" / "masks_C",
        cfg.DATASET_DIR / "test" / "images",
        cfg.DATASET_DIR / "test" / "labels",
        cfg.DATASET_DIR / "test" / "masks_A",
        cfg.DATASET_DIR / "test" / "masks_B",
        cfg.DATASET_DIR / "test" / "masks_C",
        cfg.DATASET_DIR / "pseudo" / "images",
        cfg.DATASET_DIR / "pseudo" / "labels",
        cfg.DATASET_DIR / "pseudo" / "masks_A",
        cfg.DATASET_DIR / "pseudo" / "masks_B",
        cfg.DATASET_DIR / "pseudo" / "masks_C",
        cfg.RESULT_DIR,
        cfg.RESULT_DIR / "models",
        cfg.RESULT_DIR / "plots",
        cfg.RESULT_DIR / "overlap_raw" / "labels",
        cfg.RESULT_DIR / "overlap_raw" / "masks_A",
        cfg.RESULT_DIR / "overlap_raw" / "masks_B",
        cfg.RESULT_DIR / "overlap_raw" / "masks_C",
        cfg.RESULT_DIR / "overlap_raw" / "overlays",
    ]
    for f in folders:
        ensure_dir(Path(f))
    print("Created/checked project folders.")
    print("Single chromosome folder:", cfg.SINGLE_CHR_DIR)
    print("Overlap raw folder:", cfg.OVERLAP_RAW_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _ = parser.parse_args()
    create_folders()
