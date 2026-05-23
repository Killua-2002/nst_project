from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from tqdm import tqdm

import config


def ensure_device(device_arg: str | None = None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EarlyStopping:
    def __init__(self, patience: int = 10, mode: str = "min"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            return True
        improved = value < self.best if self.mode == "min" else value > self.best
        if improved:
            self.best = value
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


def train_one_epoch(model, loader, optimizer, criterion, device) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * images.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> Tuple[float, float, float, List[int], List[int]]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    y_true, y_pred = [], []
    for images, labels in tqdm(loader, desc="eval", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += float(loss.item()) * images.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(pred.detach().cpu().numpy().tolist())
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) if y_true else 0.0
    return total_loss / max(total, 1), correct / max(total, 1), macro_f1, y_true, y_pred


def save_history_plot(history: Dict[str, List[float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for key, values in history.items():
        if "loss" in key:
            plt.plot(values, label=key)
    plt.title("Training / validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_confusion_matrix(y_true: List[int], y_pred: List[int], out_png: Path, out_csv: Path | None = None) -> np.ndarray:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    labels = list(range(len(config.CLASS_NAMES)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xticklabels(config.CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(config.CLASS_NAMES)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

    if out_csv is not None:
        import pandas as pd
        pd.DataFrame(cm, index=config.CLASS_NAMES, columns=config.CLASS_NAMES).to_csv(out_csv)
    return cm


def save_metrics_json(metrics: Dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def classification_report_dict(y_true, y_pred) -> Dict:
    return classification_report(
        y_true,
        y_pred,
        labels=list(range(len(config.CLASS_NAMES))),
        target_names=config.CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )
