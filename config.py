from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

SOURCE_DIR = PROJECT_ROOT / "source_data"
OVERLAP_RAW_DIR = SOURCE_DIR / "overlap_raw"

# Support both names in case the repo already has one of them.
SINGLE_CHROMOSOMES_DIR = SOURCE_DIR / "single_chromosomes"
if not SINGLE_CHROMOSOMES_DIR.exists() and (SOURCE_DIR / "single_chromosome").exists():
    SINGLE_CHROMOSOMES_DIR = SOURCE_DIR / "single_chromosome"

GENERATED_DIR = PROJECT_ROOT / "generated_data"
RESIZED_DIR = GENERATED_DIR / "resized"
SKELETON_DIR = GENERATED_DIR / "skeletons"

DATASET_DIR = PROJECT_ROOT / "dataset"
RESULT_DIR = PROJECT_ROOT / "result"
MODEL_DIR = RESULT_DIR / "models"
FIGURE_DIR = RESULT_DIR / "figures"
OVERLAP_RESULT_DIR = RESULT_DIR / "overlap_raw"

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

# Segmentation classes:
# 0 = background
# 1 = A-only
# 2 = B-only
# 3 = C-overlap
NUM_CLASSES = 4
CLASS_NAMES = ["background", "A_only", "B_only", "C_overlap"]

DEFAULT_IMAGE_SIZE = 224
DEFAULT_EPOCHS = 200
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-4
DEFAULT_DROPOUT = 0.5
DEFAULT_PATIENCE = 10
DEFAULT_PSEUDO_THRESHOLD = 0.55
