"""Global configuration for the chromosome classification project.

Data assumption:
- source_data/overlap_raw: real overlapped/touching chromosome images to solve/classify.
- source_data/single_chromosomes or source_data/single_chromosome: only single chromosome images.
- dataset/ is generated automatically from source_data/single_chromosomes.

Fast default pipeline:
source_data -> generated_data/resized grayscale 224x224 -> synthetic dataset -> Teacher -> pseudo labels -> Student.
Zhang-Suen skeleton is optional because it is CPU-heavy.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

SOURCE_DIR = PROJECT_ROOT / "source_data"
OVERLAP_RAW_DIR = SOURCE_DIR / "overlap_raw"
SINGLE_CHROMOSOMES_DIR = SOURCE_DIR / "single_chromosomes"
SINGLE_CHROMOSOME_ALT_DIR = SOURCE_DIR / "single_chromosome"

GENERATED_DIR = PROJECT_ROOT / "generated_data"
RESIZED_DIR = GENERATED_DIR / "resized"
RESIZED_OVERLAP_RAW_DIR = RESIZED_DIR / "overlap_raw"
RESIZED_SINGLE_CHROMOSOMES_DIR = RESIZED_DIR / "single_chromosomes"

DATASET_DIR = PROJECT_ROOT / "dataset"
RESULT_DIR = PROJECT_ROOT / "result"
CHECKPOINT_DIR = RESULT_DIR / "checkpoints"

CLASSES = ["touching", "overlapping", "touching_overlapping"]
NUM_CLASSES = len(CLASSES)

IMAGE_SIZE = 224
DEFAULT_EPOCHS = 200
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-4
DEFAULT_DROPOUT = 0.5
DEFAULT_PATIENCE = 10
DEFAULT_SYNTHETIC_PER_CLASS = 600
DEFAULT_PSEUDO_THRESHOLD = 0.75
RANDOM_SEED = 42

# Skeleton is OFF by default to keep Colab preprocessing fast.
# Turn it on with: python main.py --use-skeleton
RUN_SKELETON_DEFAULT = False
SAVE_SKELETON_DEBUG_DEFAULT = False
SKELETON_PAD = 8

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def get_single_dir() -> Path:
    """Support both folder names: single_chromosomes and single_chromosome."""
    if SINGLE_CHROMOSOMES_DIR.exists():
        return SINGLE_CHROMOSOMES_DIR
    if SINGLE_CHROMOSOME_ALT_DIR.exists():
        return SINGLE_CHROMOSOME_ALT_DIR
    return SINGLE_CHROMOSOMES_DIR


def ensure_project_dirs() -> None:
    dirs = [
        OVERLAP_RAW_DIR,
        SINGLE_CHROMOSOMES_DIR,
        GENERATED_DIR,
        RESIZED_OVERLAP_RAW_DIR,
        RESIZED_SINGLE_CHROMOSOMES_DIR,
        DATASET_DIR / "train",
        DATASET_DIR / "val",
        DATASET_DIR / "test",
        DATASET_DIR / "unlabeled",
        DATASET_DIR / "pseudo_labeled",
        DATASET_DIR / "student_train",
        RESULT_DIR,
        CHECKPOINT_DIR,
    ]
    for split in ["train", "val", "test", "pseudo_labeled", "student_train"]:
        for cls in CLASSES:
            dirs.append(DATASET_DIR / split / cls)
    for folder in dirs:
        folder.mkdir(parents=True, exist_ok=True)
