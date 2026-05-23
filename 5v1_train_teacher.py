from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

import config
from src.datasets import ChromosomeFolderDataset
from src.train_utils import (
    EarlyStopping,
    classification_report_dict,
    ensure_device,
    evaluate,
    save_confusion_matrix,
    save_history_plot,
    save_metrics_json,
    train_one_epoch,
)
import importlib
get_model = importlib.import_module("4v1_models").get_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train CCI-Net teacher classifier")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model", type=str, default=config.TEACHER_MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = ensure_device(args.device)
    print("Device:", device)

    train_ds = ChromosomeFolderDataset(config.DATASET_DIR / "train", train=True)
    val_ds = ChromosomeFolderDataset(config.DATASET_DIR / "val", train=False)
    test_ds = ChromosomeFolderDataset(config.DATASET_DIR / "test", train=False)
    if len(train_ds) == 0:
        raise RuntimeError("Empty training dataset. Run 3v1_build_dataset.py first.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = get_model(args.model, num_classes=config.NUM_CLASSES, dropout=config.DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    stopper = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, mode="min")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
    best_path = config.CHECKPOINT_DIR / "teacher_cci_net_best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1, y_true, y_pred = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        print(f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

        if stopper.step(val_loss):
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_name": args.model,
                "class_names": config.CLASS_NAMES,
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }, best_path)
            print("Saved best teacher:", best_path)
        if stopper.should_stop:
            print(f"Early stopping triggered after {config.EARLY_STOPPING_PATIENCE} epochs without improvement.")
            break

    model.load_state_dict(torch.load(best_path, map_location=device)["model_state"])
    test_loss, test_acc, test_f1, y_true, y_pred = evaluate(model, test_loader, criterion, device)

    save_history_plot(history, config.RESULT_DIR / "teacher_loss.png")
    save_confusion_matrix(y_true, y_pred, config.RESULT_DIR / "teacher_confusion_matrix.png", config.RESULT_DIR / "teacher_confusion_matrix.csv")
    save_metrics_json({
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_macro_f1": test_f1,
        "classification_report": classification_report_dict(y_true, y_pred),
        "history": history,
    }, config.RESULT_DIR / "teacher_metrics.json")
    print(f"Teacher test_acc={test_acc:.4f}, test_macro_f1={test_f1:.4f}")


if __name__ == "__main__":
    main()
