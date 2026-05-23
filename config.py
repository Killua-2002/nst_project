from pathlib import Path

# ===============================
# Global path config
# ===============================
PROJECT_ROOT = Path(__file__).resolve().parent

SOURCE_DIR = PROJECT_ROOT / "source_data"
OVERLAP_RAW_DIR = SOURCE_DIR / "overlap_raw"            # flat unlabeled/raw images to classify
SINGLE_CHR_DIR = SOURCE_DIR / "single_chromosomes"      # single chromosome images
LABELED_DIR = SOURCE_DIR / "labeled"                    # optional: labeled/<class_name>/*.png

GENERATED_DIR = PROJECT_ROOT / "generated_data"
DATASET_DIR = PROJECT_ROOT / "dataset"
RESULT_DIR = PROJECT_ROOT / "result"
CHECKPOINT_DIR = RESULT_DIR / "checkpoints"

# ===============================
# Classification classes
# ===============================
# You may change these classes if your report / lecturer requires another label set.
# Current default follows the skeleton-based classification pipeline:
# - single_path: one chromosome, one unbranched skeleton path
# - two_single_paths: exactly two unbranched skeleton paths in one image
# - complex_overlap: branch / junction / complex skeleton, likely touching or overlapping cluster
# - invalid_or_noise: unusable image, bad binary, tiny object, noisy shape
CLASS_NAMES = [
    "single_path",
    "two_single_paths",
    "complex_overlap",
    "invalid_or_noise",
]
NUM_CLASSES = len(CLASS_NAMES)

# ===============================
# Image / skeleton preprocessing
# ===============================
IMG_SIZE = 224
SKELETON_PAD = 12       # skimage padding border before Zhang-Suen, then crop back
MIN_OBJECT_AREA = 30    # remove tiny noise components
MIN_SKELETON_PIXELS = 10

# ===============================
# Training hyperparameters required by report
# ===============================
EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
LOSS_NAME = "CrossEntropyLoss"
DROPOUT = 0.5
EARLY_STOPPING_PATIENCE = 10

# Semi-supervised teacher-student config
PSEUDO_LABEL_THRESHOLD = 0.70
UNLABELED_LOSS_WEIGHT = 0.50

# Split ratio: train / val / test
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# Model names
TEACHER_MODEL = "cci_net"
STUDENT_MODEL = "swin_resnet50_fpn_v2"
