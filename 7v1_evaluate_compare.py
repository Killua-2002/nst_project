from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

import config
from src.datasets import ChromosomeFolderDataset
from src.train_utils import (
    classification_report_dict,
    ensure_device,
    evaluate,
    save_confusion_matrix,
    save_metrics_json,
)

models_mod = importlib.import_module("4v1_models")
get_model = models_mod.get_model
load_checkpoint = models_mod.load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate teacher and student, export confusion matrices")
    parser.add_argument("--teacher-ckpt", type=str, default=str(config.CHECKPOINT_DIR / "teacher_cci_net_best.pt"))
    parser.add_argument("--student-ckpt", type=str, default=str(config.CHECKPOINT_DIR / "student_swin_resnet50_fpn_v2_best.pt"))
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def eval_one(name: str, model_name: str, ckpt_path: str, loader, criterion, device):
    model = get_model(model_name, num_classes=config.NUM_CLASSES, dropout=config.DROPOUT).to(device)
    load_checkpoint(model, ckpt_path, map_location=device)
    loss, acc, f1, y_true, y_pred = evaluate(model, loader, criterion, device)
    save_confusion_matrix(
        y_true,
        y_pred,
        config.RESULT_DIR / f"{name}_confusion_matrix.png",
        config.RESULT_DIR / f"{name}_confusion_matrix.csv",
    )
    metrics = {
        "model": name,
        "checkpoint": ckpt_path,
        "loss": loss,
        "accuracy": acc,
        "macro_f1": f1,
        "classification_report": classification_report_dict(y_true, y_pred),
    }
    save_metrics_json(metrics, config.RESULT_DIR / f"{name}_eval_metrics.json")
    return {"model": name, "loss": loss, "accuracy": acc, "macro_f1": f1}


def main() -> None:
    args = parse_args()
    device = ensure_device(args.device)
    test_ds = ChromosomeFolderDataset(config.DATASET_DIR / "test", train=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    criterion = nn.CrossEntropyLoss()

    rows = []
    if Path(args.teacher_ckpt).exists():
        rows.append(eval_one("teacher", config.TEACHER_MODEL, args.teacher_ckpt, test_loader, criterion, device))
    else:
        print("Teacher checkpoint not found:", args.teacher_ckpt)
    if Path(args.student_ckpt).exists():
        rows.append(eval_one("student", config.STUDENT_MODEL, args.student_ckpt, test_loader, criterion, device))
    else:
        print("Student checkpoint not found:", args.student_ckpt)

    if rows:
        df = pd.DataFrame(rows)
        out_csv = config.RESULT_DIR / "teacher_student_comparison.csv"
        df.to_csv(out_csv, index=False)
        print(df)
        print("Saved:", out_csv)


if __name__ == "__main__":
    main()
