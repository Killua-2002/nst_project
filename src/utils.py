import json
import os
import random
import shutil
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch

import config


def set_seed(seed: int = config.RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def list_images(folder: Path) -> List[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS
    )


def clean_dir(folder: Path) -> None:
    folder = Path(folder)
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)


def ensure_empty_class_dirs(root: Path, classes: Iterable[str] = config.CLASSES) -> None:
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    for cls in classes:
        (root / cls).mkdir(parents=True, exist_ok=True)


def save_json(data, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def copy_file(src: Path, dst: Path) -> None:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def get_device(device_arg: str = "auto") -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def count_images_by_class(root: Path) -> dict:
    root = Path(root)
    result = {}
    for cls in config.CLASSES:
        result[cls] = len(list_images(root / cls))
    return result


def print_data_status() -> None:
    single_dir = config.get_single_dir()
    print("Project root:", config.PROJECT_ROOT)
    print("overlap_raw:", config.OVERLAP_RAW_DIR, "images=", len(list_images(config.OVERLAP_RAW_DIR)))
    print("single_chromosomes:", single_dir, "images=", len(list_images(single_dir)))
    for split in ["train", "val", "test", "pseudo_labeled", "student_train"]:
        root = config.DATASET_DIR / split
        if root.exists():
            print(split, count_images_by_class(root))


def safe_name(path: Path) -> str:
    return Path(path).stem.replace(" ", "_").replace("/", "_").replace("\\", "_")
