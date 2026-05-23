from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

from config import CFG


def run(cmd: list[str], skip: bool = False) -> None:
    if skip:
        print("SKIP:", " ".join(cmd))
        return
    print("\n>>> RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Full pipeline: synthetic A/B/C segmentation + Teacher-Student SSL + overlap_raw output.")
    parser.add_argument("--image-size", type=int, default=CFG.IMAGE_SIZE)
    parser.add_argument("--epochs", type=int, default=CFG.EPOCHS)
    parser.add_argument("--teacher-epochs", type=int, default=None)
    parser.add_argument("--student-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=CFG.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=CFG.LR)
    parser.add_argument("--dropout", type=float, default=CFG.DROPOUT)
    parser.add_argument("--patience", type=int, default=CFG.EARLY_STOPPING)
    parser.add_argument("--train-count", type=int, default=CFG.SYNTHETIC_TRAIN)
    parser.add_argument("--val-count", type=int, default=CFG.SYNTHETIC_VAL)
    parser.add_argument("--test-count", type=int, default=CFG.SYNTHETIC_TEST)
    parser.add_argument("--pseudo-threshold", type=float, default=CFG.PSEUDO_THRESHOLD)
    parser.add_argument("--pseudo-loss-weight", type=float, default=CFG.PSEUDO_LOSS_WEIGHT)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-skeleton", action="store_true", help="Enable Zhang-Suen skeleton stats. OFF by default because it is slow on CPU.")
    parser.add_argument("--save-skeleton-debug", action="store_true")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-teacher", action="store_true")
    parser.add_argument("--skip-student", action="store_true")
    parser.add_argument("--skip-infer", action="store_true")
    parser.add_argument("--only-preprocess", action="store_true")
    parser.add_argument("--only-generate", action="store_true")
    args = parser.parse_args()

    py = sys.executable
    teacher_epochs = args.teacher_epochs if args.teacher_epochs is not None else args.epochs
    student_epochs = args.student_epochs if args.student_epochs is not None else args.epochs

    run([py, "1v1_create.py"])

    preprocess_cmd = [py, "2v1_preprocess_resize_skeleton.py", "--image-size", str(args.image_size)]
    if args.use_skeleton:
        preprocess_cmd.append("--use-skeleton")
    if args.save_skeleton_debug:
        preprocess_cmd.append("--save-skeleton-debug")
    run(preprocess_cmd, skip=args.skip_preprocess)
    if args.only_preprocess:
        return

    generate_cmd = [
        py,
        "3v1_generate_synthetic_masks.py",
        "--image-size", str(args.image_size),
        "--train", str(args.train_count),
        "--val", str(args.val_count),
        "--test", str(args.test_count),
    ]
    run(generate_cmd, skip=args.skip_generate)
    if args.only_generate:
        return

    if not args.skip_train:
        if not args.skip_teacher:
            run([
                py, "5v1_train_teacher.py",
                "--epochs", str(teacher_epochs),
                "--batch-size", str(args.batch_size),
                "--lr", str(args.lr),
                "--dropout", str(args.dropout),
                "--patience", str(args.patience),
                "--device", args.device,
            ])
        if not args.skip_student:
            run([
                py, "6v1_train_student_ssl.py",
                "--epochs", str(student_epochs),
                "--batch-size", str(args.batch_size),
                "--lr", str(args.lr),
                "--dropout", str(args.dropout),
                "--patience", str(args.patience),
                "--pseudo-threshold", str(args.pseudo_threshold),
                "--pseudo-loss-weight", str(args.pseudo_loss_weight),
                "--device", args.device,
            ])
            run([py, "7v1_evaluate_compare.py", "--batch-size", str(args.batch_size), "--device", args.device])

    if not args.skip_infer:
        # Prefer student if trained, otherwise teacher.
        student_ckpt = CFG.RESULT_DIR / "models" / "student_swin_resnet_fpn_best.pth"
        teacher_ckpt = CFG.RESULT_DIR / "models" / "teacher_cci_best.pth"
        if student_ckpt.exists():
            run([py, "8v1_segment_overlap_raw.py", "--model", "student", "--device", args.device])
        elif teacher_ckpt.exists():
            run([py, "8v1_segment_overlap_raw.py", "--model", "teacher", "--device", args.device])
        else:
            print("No trained checkpoint found, skip overlap_raw inference.")


if __name__ == "__main__":
    main()
