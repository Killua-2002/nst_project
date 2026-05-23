import argparse

import config
from src.datasets import make_dataloaders, make_imagefolder_loader
from src.models import build_model
from src.train_utils import (
    build_student_train_from_synthetic_and_pseudo,
    load_checkpoint,
    pseudo_label_overlap_raw,
    train_model,
)
from src.utils import get_device, list_images, set_seed


def choose_overlap_dir(skip_resized: bool = False):
    if not skip_resized and len(list_images(config.RESIZED_OVERLAP_RAW_DIR)) > 0:
        return config.RESIZED_OVERLAP_RAW_DIR
    return config.OVERLAP_RAW_DIR


def main():
    parser = argparse.ArgumentParser(description="Teacher-student semi-supervised training.")
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.DEFAULT_LR)
    parser.add_argument("--patience", type=int, default=config.DEFAULT_PATIENCE)
    parser.add_argument("--dropout", type=float, default=config.DEFAULT_DROPOUT)
    parser.add_argument("--threshold", type=float, default=config.DEFAULT_PSEUDO_THRESHOLD)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=config.IMAGE_SIZE)
    parser.add_argument("--skip-resized", action="store_true", help="Use source_data/overlap_raw instead of generated_data/resized.")
    args = parser.parse_args()

    set_seed(config.RANDOM_SEED)
    config.ensure_project_dirs()
    device = get_device(args.device)
    overlap_dir = choose_overlap_dir(args.skip_resized)

    base_loaders = make_dataloaders(config.DATASET_DIR, batch_size=args.batch_size, num_workers=args.num_workers, image_size=args.image_size)
    class_names = base_loaders["class_names"]
    teacher = build_model("teacher", num_classes=len(class_names), dropout=args.dropout, pretrained=False)
    load_checkpoint(teacher, config.CHECKPOINT_DIR / "teacher_ccinet_best.pt", device)
    pseudo_label_overlap_raw(
        teacher=teacher,
        overlap_dir=overlap_dir,
        output_root=config.DATASET_DIR / "pseudo_labeled",
        device=device,
        class_names=class_names,
        threshold=args.threshold,
        batch_size=args.batch_size,
        image_size=args.image_size,
    )

    build_student_train_from_synthetic_and_pseudo(config.DATASET_DIR)

    train_ds, train_loader = make_imagefolder_loader(config.DATASET_DIR / "student_train", args.batch_size, train=True, num_workers=args.num_workers, image_size=args.image_size)
    val_ds, val_loader = make_imagefolder_loader(config.DATASET_DIR / "val", args.batch_size, train=False, num_workers=args.num_workers, image_size=args.image_size)
    test_ds, test_loader = make_imagefolder_loader(config.DATASET_DIR / "test", args.batch_size, train=False, num_workers=args.num_workers, image_size=args.image_size)
    loaders = {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "class_names": train_ds.classes,
    }

    student = build_model("student", num_classes=len(class_names), dropout=args.dropout, pretrained=args.pretrained)
    ckpt = train_model(student, loaders, device, "student_swin_resnet50fpnv2", epochs=args.epochs, lr=args.lr, patience=args.patience)
    print("Student checkpoint saved:", ckpt)


if __name__ == "__main__":
    main()
