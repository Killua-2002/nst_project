from __future__ import annotations

import argparse
import importlib
from itertools import cycle

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

import config
from src.datasets import ChromosomeFolderDataset, UnlabeledChromosomeDataset
from src.train_utils import (
    EarlyStopping,
    classification_report_dict,
    ensure_device,
    evaluate,
    save_confusion_matrix,
    save_history_plot,
    save_metrics_json,
)

models_mod = importlib.import_module("4v1_models")
get_model = models_mod.get_model
load_checkpoint = models_mod.load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train Swin Transformer + ResNet50 FPN v2 student with teacher pseudo labels")
    parser.add_argument("--teacher-ckpt", type=str, default=str(config.CHECKPOINT_DIR / "teacher_cci_net_best.pt"))
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--threshold", type=float, default=config.PSEUDO_LABEL_THRESHOLD)
    parser.add_argument("--unlabeled-weight", type=float, default=config.UNLABELED_LOSS_WEIGHT)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--student-model", type=str, default=config.STUDENT_MODEL)
    parser.add_argument("--pretrained", action="store_true", help="Use torchvision pretrained weights if internet/cache is available")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = ensure_device(args.device)
    print("Device:", device)

    train_ds = ChromosomeFolderDataset(config.DATASET_DIR / "train", train=True)
    val_ds = ChromosomeFolderDataset(config.DATASET_DIR / "val", train=False)
    test_ds = ChromosomeFolderDataset(config.DATASET_DIR / "test", train=False)
    unlabeled_ds = UnlabeledChromosomeDataset(config.DATASET_DIR / "unlabeled", strong=True)

    if len(train_ds) == 0:
        raise RuntimeError("Empty training dataset. Run 3v1_build_dataset.py first.")
    if len(unlabeled_ds) == 0:
        print("[WARNING] dataset/unlabeled is empty. Student will train supervised only.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    unlabeled_loader = DataLoader(unlabeled_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True) if len(unlabeled_ds) else None

    teacher = get_model(config.TEACHER_MODEL, num_classes=config.NUM_CLASSES, dropout=config.DROPOUT).to(device)
    load_checkpoint(teacher, args.teacher_ckpt, map_location=device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = get_model(args.student_model, num_classes=config.NUM_CLASSES, dropout=config.DROPOUT, pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    stopper = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, mode="min")
    best_path = config.CHECKPOINT_DIR / "student_swin_resnet50_fpn_v2_best.pt"

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": [], "pseudo_used": []}

    for epoch in range(1, args.epochs + 1):
        student.train()
        total_loss, correct, total, pseudo_used = 0.0, 0, 0, 0
        unlabeled_iter = cycle(unlabeled_loader) if unlabeled_loader is not None else None

        for images, labels in tqdm(train_loader, desc=f"student train {epoch}", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = student(images)
            supervised_loss = criterion(logits, labels)
            loss = supervised_loss

            if unlabeled_iter is not None:
                u_images, _ = next(unlabeled_iter)
                u_images = u_images.to(device)
                with torch.no_grad():
                    t_logits = teacher(u_images)
                    probs = F.softmax(t_logits, dim=1)
                    conf, pseudo = probs.max(dim=1)
                    mask = conf >= args.threshold
                if mask.any():
                    s_logits = student(u_images[mask])
                    unsup_loss = criterion(s_logits, pseudo[mask])
                    loss = supervised_loss + args.unlabeled_weight * unsup_loss
                    pseudo_used += int(mask.sum().item())

            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * images.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == labels).sum().item())
            total += int(labels.numel())

        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc, val_f1, y_true, y_pred = evaluate(student, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["pseudo_used"].append(pseudo_used)
        print(f"Epoch {epoch:03d}/{args.epochs} | loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} pseudo_used={pseudo_used}")

        if stopper.step(val_loss):
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_name": args.student_model,
                "class_names": config.CLASS_NAMES,
                "model_state": student.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }, best_path)
            print("Saved best student:", best_path)
        if stopper.should_stop:
            print(f"Early stopping triggered after {config.EARLY_STOPPING_PATIENCE} epochs without improvement.")
            break

    student.load_state_dict(torch.load(best_path, map_location=device)["model_state"], strict=False)
    test_loss, test_acc, test_f1, y_true, y_pred = evaluate(student, test_loader, criterion, device)

    save_history_plot(history, config.RESULT_DIR / "student_loss.png")
    save_confusion_matrix(y_true, y_pred, config.RESULT_DIR / "student_confusion_matrix.png", config.RESULT_DIR / "student_confusion_matrix.csv")
    save_metrics_json({
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_macro_f1": test_f1,
        "classification_report": classification_report_dict(y_true, y_pred),
        "history": history,
        "pseudo_label_threshold": args.threshold,
    }, config.RESULT_DIR / "student_metrics.json")
    print(f"Student test_acc={test_acc:.4f}, test_macro_f1={test_f1:.4f}")


if __name__ == "__main__":
    main()
