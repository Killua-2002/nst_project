import argparse
import importlib.util
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    CLASS_NAMES,
    DATASET_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_EPOCHS,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_LR,
    DEFAULT_PATIENCE,
    MODEL_DIR,
    RESULT_DIR,
)
from utils_train import (
    SegmentationDataset,
    load_checkpoint,
    metrics_from_cm,
    pixel_confusion_matrix,
    plot_confusion_matrix,
    save_checkpoint,
    write_history_csv,
    save_json,
)


def _load_models_module():
    spec = importlib.util.spec_from_file_location("models_v1", Path(__file__).resolve().parent / "4v1_models.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args():
    p = argparse.ArgumentParser(description="Train Teacher CCI-Net for A/B/C segmentation.")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    p.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    p.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def evaluate(model, loader, device):
    model.eval()
    cm = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.long, device="cpu")
    loss_sum = 0.0
    n = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            pred = logits.argmax(dim=1)
            cm += pixel_confusion_matrix(pred.cpu(), y.cpu())
            loss_sum += float(loss.item()) * x.size(0)
            n += x.size(0)
    metrics = metrics_from_cm(cm)
    metrics["loss"] = loss_sum / max(1, n)
    return metrics, cm


def main():
    args = parse_args()
    device = torch.device(args.device)
    models_mod = _load_models_module()
    model = models_mod.build_model("teacher", dropout=args.dropout).to(device)

    train_ds = SegmentationDataset(DATASET_DIR / "train/images", DATASET_DIR / "train/labels", augment=True, image_size=args.image_size)
    val_ds = SegmentationDataset(DATASET_DIR / "val/images", DATASET_DIR / "val/labels", augment=False, image_size=args.image_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    best_val = 10**9
    wait = 0
    history = []
    best_path = MODEL_DIR / "teacher_cci_net_best.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        pbar = tqdm(train_loader, desc=f"Teacher epoch {epoch}/{args.epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * x.size(0)
            total_n += x.size(0)
            pbar.set_postfix(loss=total_loss / max(1, total_n))

        train_loss = total_loss / max(1, total_n)
        val_metrics, val_cm = evaluate(model, val_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_pixel_acc": val_metrics["pixel_acc"],
            "val_mean_iou": val_metrics["mean_iou"],
            "val_mean_f1": val_metrics["mean_f1"],
        }
        history.append(row)
        print(row)

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            wait = 0
            save_checkpoint(best_path, model, {"epoch": epoch, "image_size": args.image_size, "model": "CCINetTeacher"})
        else:
            wait += 1
            if wait >= args.patience:
                print(f"[EARLY STOP] Teacher stopped at epoch {epoch}")
                break

    write_history_csv(RESULT_DIR / "teacher_training_log.csv", history)

    load_checkpoint(best_path, model, map_location=device)
    val_metrics, val_cm = evaluate(model, val_loader, device)
    plot_confusion_matrix(RESULT_DIR / "figures/teacher_val_confusion_matrix.png", val_cm, CLASS_NAMES)
    save_json(RESULT_DIR / "teacher_val_metrics.json", val_metrics)
    print("[OK] Teacher training done:", best_path)


if __name__ == "__main__":
    main()
