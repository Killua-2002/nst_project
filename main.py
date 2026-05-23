import argparse
import sys

import config
from src.skeleton_utils import process_folder
from src.synthetic_data import build_synthetic_dataset
from src.utils import get_device, list_images, print_data_status, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="NST chromosome Teacher-Student classification pipeline.")
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.DEFAULT_LR)
    parser.add_argument("--patience", type=int, default=config.DEFAULT_PATIENCE)
    parser.add_argument("--dropout", type=float, default=config.DEFAULT_DROPOUT)
    parser.add_argument("--threshold", type=float, default=config.DEFAULT_PSEUDO_THRESHOLD)
    parser.add_argument("--synthetic-per-class", type=int, default=config.DEFAULT_SYNTHETIC_PER_CLASS)
    parser.add_argument("--image-size", type=int, default=config.IMAGE_SIZE)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained Swin/ResNet weights if internet is available.")
    parser.add_argument("--strict-single", action="store_true", help="Only use single images that pass skeleton single-line check.")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-build-dataset", action="store_true")
    parser.add_argument("--skip-train", action="store_true", help="Only create folders/preprocess/build dataset/classify if checkpoints exist.")
    parser.add_argument("--teacher-only", action="store_true")
    parser.add_argument("--student-only", action="store_true")
    return parser.parse_args()


def validate_source_data():
    single_dir = config.get_single_dir()
    overlap_imgs = list_images(config.OVERLAP_RAW_DIR)
    single_imgs = list_images(single_dir)
    if len(single_imgs) < 2:
        raise RuntimeError(
            f"Not enough single chromosome images in {single_dir}. "
            "This fixed project expects source_data/single_chromosomes to contain at least 2 images."
        )
    if len(overlap_imgs) == 0:
        print("Warning: source_data/overlap_raw is empty. Training can run, but real classification will be skipped.")
    return single_dir, overlap_imgs, single_imgs


def main():
    args = parse_args()
    set_seed(config.RANDOM_SEED)
    config.ensure_project_dirs()
    single_dir, overlap_imgs, single_imgs = validate_source_data()
    device = get_device(args.device)
    print_data_status()
    print("Device:", device)

    if not args.skip_preprocess:
        print("\n[1/6] Preprocess grayscale + Zhang-Suen skeleton")
        process_folder(config.OVERLAP_RAW_DIR, config.GENERATED_DIR / "overlap_raw", image_size=args.image_size, pad=8)
        process_folder(single_dir, config.GENERATED_DIR / "single_chromosomes", image_size=args.image_size, pad=8)

    if not args.skip_build_dataset:
        print("\n[2/6] Build synthetic train/val/test from single_chromosomes")
        build_synthetic_dataset(
            single_dir=single_dir,
            dataset_dir=config.DATASET_DIR,
            per_class=args.synthetic_per_class,
            image_size=args.image_size,
            seed=config.RANDOM_SEED,
            strict_single=args.strict_single,
        )

    if args.skip_train:
        print("\n--skip-train enabled. Skipping teacher/student training.")
    else:
        from src.datasets import make_dataloaders, make_imagefolder_loader
        from src.models import build_model
        from src.train_utils import (
            build_student_train_from_synthetic_and_pseudo,
            classify_folder,
            evaluate_model,
            load_checkpoint,
            pseudo_label_overlap_raw,
            train_model,
        )

        loaders = make_dataloaders(config.DATASET_DIR, batch_size=args.batch_size, num_workers=args.num_workers)
        class_names = loaders["class_names"]

        if not args.student_only:
            print("\n[3/6] Train Teacher = CCI-Net style")
            teacher = build_model("teacher", num_classes=len(class_names), dropout=args.dropout, pretrained=False)
            train_model(teacher, loaders, device, "teacher_ccinet", epochs=args.epochs, lr=args.lr, patience=args.patience)

        if not args.teacher_only:
            print("\n[4/6] Pseudo-label overlap_raw using Teacher")
            teacher = build_model("teacher", num_classes=len(class_names), dropout=args.dropout, pretrained=False)
            load_checkpoint(teacher, config.CHECKPOINT_DIR / "teacher_ccinet_best.pt", device)
            if overlap_imgs:
                pseudo_label_overlap_raw(
                    teacher=teacher,
                    overlap_dir=config.OVERLAP_RAW_DIR,
                    output_root=config.DATASET_DIR / "pseudo_labeled",
                    device=device,
                    class_names=class_names,
                    threshold=args.threshold,
                    batch_size=args.batch_size,
                )
            else:
                print("No overlap_raw images to pseudo-label.")

            print("\n[5/6] Train Student = Swin Transformer + ResNet50 FPN v2")
            build_student_train_from_synthetic_and_pseudo(config.DATASET_DIR)
            train_ds, train_loader = make_imagefolder_loader(config.DATASET_DIR / "student_train", args.batch_size, train=True, num_workers=args.num_workers)
            val_ds, val_loader = make_imagefolder_loader(config.DATASET_DIR / "val", args.batch_size, train=False, num_workers=args.num_workers)
            test_ds, test_loader = make_imagefolder_loader(config.DATASET_DIR / "test", args.batch_size, train=False, num_workers=args.num_workers)
            student_loaders = {
                "train_ds": train_ds,
                "val_ds": val_ds,
                "test_ds": test_ds,
                "train": train_loader,
                "val": val_loader,
                "test": test_loader,
                "class_names": train_ds.classes,
            }
            student = build_model("student", num_classes=len(class_names), dropout=args.dropout, pretrained=args.pretrained)
            train_model(student, student_loaders, device, "student_swin_resnet50fpnv2", epochs=args.epochs, lr=args.lr, patience=args.patience)

        print("\n[6/6] Evaluate and classify overlap_raw")
        try:
            teacher = build_model("teacher", num_classes=config.NUM_CLASSES, dropout=args.dropout, pretrained=False)
            load_checkpoint(teacher, config.CHECKPOINT_DIR / "teacher_ccinet_best.pt", device)
            evaluate_model(teacher, config.DATASET_DIR, device, "teacher_ccinet", args.batch_size)
        except Exception as exc:
            print("Teacher evaluation skipped:", exc)

        try:
            student = build_model("student", num_classes=config.NUM_CLASSES, dropout=args.dropout, pretrained=args.pretrained)
            load_checkpoint(student, config.CHECKPOINT_DIR / "student_swin_resnet50fpnv2_best.pt", device)
            evaluate_model(student, config.DATASET_DIR, device, "student_swin_resnet50fpnv2", args.batch_size)
            if overlap_imgs:
                classify_folder(
                    model=student,
                    folder=config.OVERLAP_RAW_DIR,
                    output_csv=config.RESULT_DIR / "overlap_raw_predictions_student.csv",
                    device=device,
                    class_names=config.CLASSES,
                    batch_size=args.batch_size,
                    output_folder=config.RESULT_DIR / "classified_overlap_raw_student",
                )
        except Exception as exc:
            print("Student evaluation/classification skipped:", exc)

    print("\nDone. Check result/ for confusion matrix, reports, pseudo labels and predictions.")


if __name__ == "__main__":
    main()
