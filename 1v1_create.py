from pathlib import Path

import config


def main() -> None:
    folders = [
        config.SOURCE_DIR,
        config.OVERLAP_RAW_DIR,
        config.SINGLE_CHR_DIR,
        config.LABELED_DIR,
        config.GENERATED_DIR / "grayscale",
        config.GENERATED_DIR / "binary",
        config.GENERATED_DIR / "skeleton",
        config.GENERATED_DIR / "overlay",
        config.DATASET_DIR,
        config.DATASET_DIR / "unlabeled",
        config.RESULT_DIR,
        config.CHECKPOINT_DIR,
    ]
    for split in ["train", "val", "test"]:
        for cls in config.CLASS_NAMES:
            folders.append(config.DATASET_DIR / split / cls)

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)

    print("Created project folders:")
    for folder in folders:
        print("-", folder.relative_to(config.PROJECT_ROOT))


if __name__ == "__main__":
    main()
