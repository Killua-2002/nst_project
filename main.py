from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(script: str, extra_args=None) -> None:
    cmd = [sys.executable, str(ROOT / script)]
    if extra_args:
        cmd.extend(extra_args)
    print("\n" + "=" * 90)
    print("RUN:", " ".join(cmd))
    print("=" * 90)
    subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run full chromosome semi-supervised classification pipeline")
    parser.add_argument("--skip-train", action="store_true", help="Only create folders, preprocess, build dataset")
    parser.add_argument("--skip-teacher", action="store_true", help="Skip teacher training")
    parser.add_argument("--skip-student", action="store_true", help="Skip student training")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for quick testing")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", type=str, default=None, help="cuda, cpu, cuda:0, ...")
    return parser.parse_args()


def common_args(args):
    out = []
    if args.epochs is not None:
        out += ["--epochs", str(args.epochs)]
    if args.batch_size is not None:
        out += ["--batch-size", str(args.batch_size)]
    if args.device is not None:
        out += ["--device", args.device]
    return out


def main() -> None:
    args = parse_args()
    run("1v1_create.py")
    run("2v1_preprocess_zhang_suen.py")
    run("3v1_build_dataset.py")

    if args.skip_train:
        print("Done preprocessing and dataset building. Training skipped.")
        return

    if not args.skip_teacher:
        run("5v1_train_teacher.py", common_args(args))
    if not args.skip_student:
        run("6v1_train_student_ssl.py", common_args(args))
    run("7v1_evaluate_compare.py", [a for a in common_args(args) if a not in {"--epochs"}])
    run("8v1_classify_overlap_raw.py", ["--device", args.device] if args.device else [])


if __name__ == "__main__":
    main()
