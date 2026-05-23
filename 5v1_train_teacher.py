from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import CFG
from models import build_model
from train_utils import SegmentationDataset, evaluate, fit_model, load_checkpoint, plot_confusion_matrix
from utils_image import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Teacher CCI-Net for A/B/C segmentation on synthetic masks.")
    parser.add_argument("--epochs", type=int, default=CFG.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=CFG.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=CFG.LR)
    parser.add_argument("--dropout", type=float, default=CFG.DROPOUT)
    parser.add_argument("--patience", type=int, default=CFG.EARLY_STOPPING)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=CFG.SEED)
    args = parser.parse_args()

    seed_everything(args.seed)
    cfg = CFG()
    train_ds = SegmentationDataset(cfg.DATASET_DIR, "train", augment=True)
    val_ds = SegmentationDataset(cfg.DATASET_DIR, "val", augment=False)
    test_ds = SegmentationDataset(cfg.DATASET_DIR, "test", augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = build_model("teacher", num_classes=cfg.NUM_CLASSES, dropout=args.dropout)
    ckpt = cfg.RESULT_DIR / "models" / "teacher_cci_best.pth"
    log_csv = cfg.RESULT_DIR / "teacher_training_log.csv"
    fit_model(
        model,
        train_loader,
        val_loader,
        args.device,
        args.epochs,
        args.lr,
        args.patience,
        ckpt,
        log_csv,
        cfg.CLASS_NAMES,
    )

    model = build_model("teacher", num_classes=cfg.NUM_CLASSES, dropout=args.dropout)
    load_checkpoint(model, ckpt, args.device)
    model.to(args.device)
    test_metrics = evaluate(model, test_loader, args.device, cfg.CLASS_NAMES)
    print("Teacher test metrics:", {k: v for k, v in test_metrics.items() if k != "cm"})
    plot_confusion_matrix(
        test_metrics["cm"],
        cfg.CLASS_NAMES,
        cfg.RESULT_DIR / "plots" / "teacher_pixel_confusion_matrix.png",
        "Teacher CCI-Net pixel confusion matrix",
    )


if __name__ == "__main__":
    main()
