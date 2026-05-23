import argparse
import subprocess
import sys
from pathlib import Path

from config import DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_IMAGE_SIZE, DEFAULT_LR, DEFAULT_PATIENCE, DEFAULT_PSEUDO_THRESHOLD, PROJECT_ROOT


def run(cmd):
    print("\n" + "=" * 90)
    print("[RUN]", " ".join(cmd))
    print("=" * 90)
    subprocess.check_call(cmd, cwd=PROJECT_ROOT)


def parse_args():
    p = argparse.ArgumentParser(description="Full semi-supervised A/B/C chromosome segmentation pipeline.")
    p.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    p.add_argument("--train-count", type=int, default=600)
    p.add_argument("--val-count", type=int, default=120)
    p.add_argument("--test-count", type=int, default=120)
    p.add_argument("--pseudo-threshold", type=float, default=DEFAULT_PSEUDO_THRESHOLD)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use-skeleton", action="store_true", help="Run Zhang-Suen skeleton QC during preprocessing.")
    p.add_argument("--strict-skeleton", action="store_true", help="Use only single chromosomes with valid one-path skeleton when generating synthetic data.")
    p.add_argument("--save-skeleton-debug", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--only-preprocess", action="store_true")
    p.add_argument("--keep-largest", action="store_true", help="Postprocess final A/B masks by keeping largest component.")
    p.add_argument("--no-shape-postprocess", action="store_true", help="Disable fill holes / smooth A/B shape after segmentation.")
    p.add_argument("--no-skeleton-repair", action="store_true", help="Disable Zhang-Suen-guided candidate selection for A/B masks.")
    p.add_argument("--close-radius", type=int, default=2, help="Morphological closing radius for A/B shape repair.")
    p.add_argument("--hole-area", type=int, default=768, help="Fill holes up to this area in A/B masks.")
    return p.parse_args()


def main():
    args = parse_args()
    py = sys.executable

    run([py, "1v1_create.py"])

    preprocess_cmd = [
        py, "2v1_preprocess_zhang_suen.py",
        "--image-size", str(args.image_size),
    ]
    if args.use_skeleton:
        preprocess_cmd.append("--use-skeleton")
    if args.save_skeleton_debug:
        preprocess_cmd.append("--save-skeleton-debug")
    run(preprocess_cmd)

    if args.only_preprocess:
        print("[OK] only-preprocess finished.")
        return

    synth_cmd = [
        py, "3v1_generate_synthetic_masks.py",
        "--image-size", str(args.image_size),
        "--train-count", str(args.train_count),
        "--val-count", str(args.val_count),
        "--test-count", str(args.test_count),
        "--clear-dataset",
    ]
    if args.strict_skeleton:
        synth_cmd.append("--strict-skeleton")
    run(synth_cmd)

    if not args.skip_train:
        common_train_args = [
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--patience", str(args.patience),
            "--image-size", str(args.image_size),
            "--device", args.device,
        ]

        run([py, "5v1_train_teacher.py", *common_train_args])

        run([
            py, "6v1_train_student_ssl.py",
            *common_train_args,
            "--pseudo-threshold", str(args.pseudo_threshold),
        ])

        run([
            py, "7v1_evaluate_compare.py",
            "--batch-size", str(args.batch_size),
            "--image-size", str(args.image_size),
            "--device", args.device,
        ])

    seg_cmd = [
        py, "8v1_segment_overlap_raw.py",
        "--model", "student",
        "--image-size", str(args.image_size),
        "--device", args.device,
    ]
    if args.keep_largest:
        seg_cmd.append("--keep-largest")
    if args.no_shape_postprocess:
        seg_cmd.append("--no-shape-postprocess")
    if args.no_skeleton_repair:
        seg_cmd.append("--no-skeleton-repair")
    seg_cmd.extend(["--close-radius", str(args.close_radius), "--hole-area", str(args.hole_area)])
    run(seg_cmd)

    print("\n[OK] FULL PIPELINE DONE")
    print("Check result/overlap_raw for masks_A, masks_B, masks_C, overlays, skeleton_qc and CSV.")


if __name__ == "__main__":
    main()
