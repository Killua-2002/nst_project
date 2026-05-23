from pathlib import Path

from config import PROJECT_ROOT
from utils_image import ensure_dir


FOLDERS = [
    "source_data/overlap_raw",
    "source_data/single_chromosomes",
    "generated_data/resized/overlap_raw",
    "generated_data/resized/single_chromosomes",
    "generated_data/skeletons/overlap_raw",
    "generated_data/skeletons/single_chromosomes",
    "dataset/train/images",
    "dataset/train/labels",
    "dataset/val/images",
    "dataset/val/labels",
    "dataset/test/images",
    "dataset/test/labels",
    "dataset/pseudo/images",
    "dataset/pseudo/labels",
    "result/models",
    "result/figures",
    "result/overlap_raw/labels",
    "result/overlap_raw/masks_A",
    "result/overlap_raw/masks_B",
    "result/overlap_raw/masks_C",
    "result/overlap_raw/skeleton_qc",
    "result/overlap_raw/overlays",
]


def main():
    for folder in FOLDERS:
        path = PROJECT_ROOT / folder
        ensure_dir(path)
        (path / ".gitkeep").touch(exist_ok=True)
    print("[OK] Project folders are ready.")
    print(PROJECT_ROOT)


if __name__ == "__main__":
    main()
