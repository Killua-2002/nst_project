from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from src.datasets import UnlabeledImageDataset, get_transforms, make_dataloaders, make_imagefolder_loader
from src.utils import clean_dir, copy_file, list_images, safe_name


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    y_true, y_pred = [], []
    for images, labels, _paths in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * images.size(0)
        preds = logits.argmax(dim=1).detach().cpu().numpy().tolist()
        y_pred.extend(preds)
        y_true.extend(labels.detach().cpu().numpy().tolist())
    avg_loss = total_loss / max(1, len(loader.dataset))
    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) if y_true else 0.0
    return avg_loss, acc, f1


@torch.no_grad()
def evaluate_loader(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    y_true, y_pred, probs_all, paths_all = [], [], [], []
    for images, labels, paths in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        total_loss += float(loss.item()) * images.size(0)
        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())
        probs_all.extend(probs.cpu().numpy().tolist())
        paths_all.extend(list(paths))
    avg_loss = total_loss / max(1, len(loader.dataset))
    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) if y_true else 0.0
    return avg_loss, acc, f1, y_true, y_pred, probs_all, paths_all


def save_history_plot(history: List[dict], out_path: Path) -> None:
    if not history:
        return
    df = pd.DataFrame(history)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path.with_suffix(".csv"), index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    plt.plot(df["epoch"], df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def train_model(
    model: nn.Module,
    dataloaders: Dict,
    device: torch.device,
    model_name: str,
    epochs: int = config.DEFAULT_EPOCHS,
    lr: float = config.DEFAULT_LR,
    patience: int = config.DEFAULT_PATIENCE,
    weight_decay: float = 1e-4,
) -> Path:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = float("inf")
    best_epoch = -1
    wait = 0
    history = []
    ckpt_path = config.CHECKPOINT_DIR / f"{model_name}_best.pt"
    model.to(device)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(model, dataloaders["train"], criterion, optimizer, device)
        val_loss, val_acc, val_f1, *_ = evaluate_loader(model, dataloaders["val"], criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
        }
        history.append(row)
        print(
            f"[{model_name}] epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            wait = 0
            torch.save({
                "model_state": model.state_dict(),
                "class_names": dataloaders["class_names"],
                "epoch": epoch,
                "val_loss": val_loss,
            }, ckpt_path)
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}. Best epoch = {best_epoch}.")
                break

    save_history_plot(history, config.RESULT_DIR / f"{model_name}_loss.png")
    return ckpt_path


def load_checkpoint(model: nn.Module, ckpt_path: Path, device: torch.device) -> None:
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])


def save_confusion_and_report(y_true, y_pred, class_names: List[str], prefix: Path) -> Dict:
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(prefix.with_name(prefix.name + "_classification_report.csv"))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(prefix.with_name(prefix.name + "_confusion_matrix.csv"))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(7, 7))
    disp.plot(ax=ax, xticks_rotation=35, colorbar=False)
    ax.set_title(prefix.name.replace("_", " ").title())
    plt.tight_layout()
    fig.savefig(prefix.with_name(prefix.name + "_confusion_matrix.png"), dpi=160)
    plt.close(fig)
    return {"accuracy": accuracy_score(y_true, y_pred), "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0)}


def evaluate_model(model: nn.Module, dataset_root: Path, device: torch.device, model_name: str, batch_size: int):
    _ds, loader = make_imagefolder_loader(Path(dataset_root) / "test", batch_size=batch_size, train=False)
    criterion = nn.CrossEntropyLoss()
    _loss, acc, f1, y_true, y_pred, probs, paths = evaluate_loader(model, loader, criterion, device)
    metrics = save_confusion_and_report(y_true, y_pred, _ds.classes, config.RESULT_DIR / model_name)
    rows = []
    for true, pred, prob, path in zip(y_true, y_pred, probs, paths):
        rows.append({
            "file": path,
            "true_label": _ds.classes[true],
            "pred_label": _ds.classes[pred],
            "confidence": float(max(prob)),
        })
    pd.DataFrame(rows).to_csv(config.RESULT_DIR / f"{model_name}_test_predictions.csv", index=False)
    return metrics


@torch.no_grad()
def pseudo_label_overlap_raw(
    teacher: nn.Module,
    overlap_dir: Path,
    output_root: Path,
    device: torch.device,
    class_names: List[str],
    threshold: float = config.DEFAULT_PSEUDO_THRESHOLD,
    batch_size: int = config.DEFAULT_BATCH_SIZE,
) -> pd.DataFrame:
    clean_dir(output_root)
    for cls in class_names:
        (Path(output_root) / cls).mkdir(parents=True, exist_ok=True)

    image_paths = list_images(overlap_dir)
    transform = get_transforms(train=False)
    ds = UnlabeledImageDataset(image_paths, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    rows = []
    teacher.eval().to(device)
    for images, paths in tqdm(loader, desc="pseudo-label overlap_raw"):
        images = images.to(device, non_blocking=True)
        logits = teacher(images)
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        for path, c, p, prob_vec in zip(paths, conf.cpu().numpy(), pred.cpu().numpy(), probs.cpu().numpy()):
            label = class_names[int(p)]
            accepted = float(c) >= threshold
            rows.append({
                "file": path,
                "pseudo_label": label,
                "confidence": float(c),
                "accepted": bool(accepted),
                **{f"prob_{class_names[i]}": float(prob_vec[i]) for i in range(len(class_names))},
            })
            if accepted:
                dst = Path(output_root) / label / Path(path).name
                copy_file(Path(path), dst)
    df = pd.DataFrame(rows)
    df.to_csv(config.RESULT_DIR / "pseudo_labels_overlap_raw.csv", index=False)
    return df


def build_student_train_from_synthetic_and_pseudo(dataset_dir: Path) -> Path:
    dataset_dir = Path(dataset_dir)
    student_root = dataset_dir / "student_train"
    clean_dir(student_root)
    for cls in config.CLASSES:
        (student_root / cls).mkdir(parents=True, exist_ok=True)

    for cls in config.CLASSES:
        for src in list_images(dataset_dir / "train" / cls):
            copy_file(src, student_root / cls / f"synthetic_{src.name}")
        for src in list_images(dataset_dir / "pseudo_labeled" / cls):
            copy_file(src, student_root / cls / f"pseudo_{src.name}")
    return student_root


@torch.no_grad()
def classify_folder(
    model: nn.Module,
    folder: Path,
    output_csv: Path,
    device: torch.device,
    class_names: List[str],
    batch_size: int = config.DEFAULT_BATCH_SIZE,
    copy_to_class_folders: bool = True,
    output_folder: Path | None = None,
) -> pd.DataFrame:
    image_paths = list_images(folder)
    ds = UnlabeledImageDataset(image_paths, transform=get_transforms(train=False))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    model.eval().to(device)
    rows = []
    if output_folder and copy_to_class_folders:
        clean_dir(output_folder)
        for cls in class_names:
            (Path(output_folder) / cls).mkdir(parents=True, exist_ok=True)

    for images, paths in tqdm(loader, desc=f"classify {Path(folder).name}"):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        for path, c, p, prob_vec in zip(paths, conf.cpu().numpy(), pred.cpu().numpy(), probs.cpu().numpy()):
            label = class_names[int(p)]
            row = {
                "file": path,
                "pred_label": label,
                "confidence": float(c),
                **{f"prob_{class_names[i]}": float(prob_vec[i]) for i in range(len(class_names))},
            }
            rows.append(row)
            if output_folder and copy_to_class_folders:
                copy_file(Path(path), Path(output_folder) / label / Path(path).name)
    df = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df
