from pathlib import Path
from dataclasses import dataclass

@dataclass
class CFG:
    PROJECT_ROOT: Path = Path(__file__).resolve().parent

    # Raw data folders. Keep these exactly as the GitHub data structure.
    SOURCE_DIR: Path = PROJECT_ROOT / "source_data"
    OVERLAP_RAW_DIR: Path = SOURCE_DIR / "overlap_raw"
    SINGLE_CHR_DIR: Path = SOURCE_DIR / "single_chromosomes"

    # Auto-created working folders.
    GENERATED_DIR: Path = PROJECT_ROOT / "generated_data"
    RESIZED_DIR: Path = GENERATED_DIR / "resized"
    DATASET_DIR: Path = PROJECT_ROOT / "dataset"
    RESULT_DIR: Path = PROJECT_ROOT / "result"

    IMAGE_SIZE: int = 224
    NUM_CLASSES: int = 4  # 0=background, 1=A-only, 2=B-only, 3=C-overlap
    CLASS_NAMES = ["background", "A_only", "B_only", "C_overlap"]

    EPOCHS: int = 200
    BATCH_SIZE: int = 64
    LR: float = 1e-4
    DROPOUT: float = 0.5
    EARLY_STOPPING: int = 10

    # Synthetic data size per split.
    SYNTHETIC_TRAIN: int = 600
    SYNTHETIC_VAL: int = 120
    SYNTHETIC_TEST: int = 120

    # Skeleton is OFF by default because Zhang-Suen is CPU-heavy.
    USE_SKELETON: bool = False
    SAVE_SKELETON_DEBUG: bool = False

    # Semi-supervised pseudo label settings.
    PSEUDO_THRESHOLD: float = 0.65
    PSEUDO_LOSS_WEIGHT: float = 0.35

    SEED: int = 42


def resolve_single_chromosome_dir(source_dir: Path) -> Path:
    """Accept both single_chromosomes and single_chromosome for safety."""
    plural = source_dir / "single_chromosomes"
    singular = source_dir / "single_chromosome"
    if plural.exists():
        return plural
    if singular.exists():
        return singular
    return plural
