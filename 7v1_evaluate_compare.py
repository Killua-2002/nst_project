from __future__ import annotations

import argparse
import json

import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import CFG
from models import build_model
from train_utils import SegmentationDataset, evaluate, load_checkpoint, plot_confusion_matrix


def evaluate_one(model_name: str, ckpt_path, loader, cfg: CFG, device: str, dropout: float):
    model = build_model(model_name, num_classes=cfg.NUM_CLASSES, dropout=dropout)
    load_checkpoint(model, ckpt_path, device)
    model.to(device)
    metrics = evaluate(model, loader, device, cfg.CLASS_NAMES)
    cm = metrics.pop("cm")
    return metrics, cm


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Teacher and Student on synthetic test set using pixel-level confusion matrix.")
    parser.add_argument("--batch-size", type=int, default=CFG.BATCH_SIZE)
    parser.add_argument("--dropout", type=float, default=CFG.DROPOUT)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = CFG()
    test_ds = SegmentationDataset(cfg.DATASET_DIR, "test", augment=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    teacher_ckpt = cfg.RESULT_DIR / "models" / "teacher_cci_best.pth"
    student_ckpt = cfg.RESULT_DIR / "models" / "student_swin_resnet_fpn_best.pth"
    rows = []

    if teacher_ckpt.exists():
        m, cm = evaluate_one("teacher", teacher_ckpt, test_loader, cfg, args.device, args.dropout)
        rows.append({"model": "Teacher CCI-Net", **m})
        plot_confusion_matrix(cm, cfg.CLASS_NAMES, cfg.RESULT_DIR / "plots" / "compare_teacher_confusion_matrix.png", "Teacher test confusion matrix")
    else:
        print("Teacher checkpoint not found:", teacher_ckpt)

    if student_ckpt.exists():
        m, cm = evaluate_one("student", student_ckpt, test_loader, cfg, args.device, args.dropout)
        rows.append({"model": "Student Swin+ResNet-FPN", **m})
        plot_confusion_matrix(cm, cfg.CLASS_NAMES, cfg.RESULT_DIR / "plots" / "compare_student_confusion_matrix.png", "Student test confusion matrix")
    else:
        print("Student checkpoint not found:", student_ckpt)

    if rows:
        df = pd.DataFrame(rows)
        out = cfg.RESULT_DIR / "teacher_student_comparison.csv"
        df.to_csv(out, index=False)
        print(df)
        print("Saved:", out)


if __name__ == "__main__":
    main()
