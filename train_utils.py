from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from config import CFG
from utils_image import ensure_dir, list_images


class SegmentationDataset(Dataset):
    def __init__(self, root: Path, split: str, augment: bool = False):
        self.root = Path(root)
        self.split = split
        self.augment = augment
        self.image_dir = self.root / split / "images"
        self.label_dir = self.root / split / "labels"
        self.images = list_images(self.image_dir)
        if not self.images:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def _load(self, path: Path) -> Tuple[Image.Image, Image.Image]:
        img = Image.open(path).convert("L")
        lab_path = self.label_dir / path.name
        label = Image.open(lab_path).convert("L")
        return img, label

    def _augment(self, img: Image.Image, label: Image.Image) -> Tuple[Image.Image, Image.Image]:
        # RandomResizedCrop, same params for image/label. Implemented with PIL only
        # to avoid torchvision binary issues in some Colab/local environments.
        if random.random() < 0.8:
            width, height = img.size
            scale = random.uniform(0.80, 1.00)
            ratio = random.uniform(0.90, 1.10)
            crop_h = int(height * scale)
            crop_w = int(crop_h * ratio)
            crop_w = min(crop_w, width)
            crop_h = min(crop_h, height)
            top = random.randint(0, max(0, height - crop_h))
            left = random.randint(0, max(0, width - crop_w))
            img = img.crop((left, top, left + crop_w, top + crop_h)).resize((width, height), resample=Image.BILINEAR)
            label = label.crop((left, top, left + crop_w, top + crop_h)).resize((width, height), resample=Image.NEAREST)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            label = label.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < 0.8:
            angle = random.uniform(-15, 15)
            img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=255)
            # 255 label is ignore_index for pseudo data; for synthetic labels fill background.
            fill = 255 if self.split == "pseudo" else 0
            label = label.rotate(angle, resample=Image.NEAREST, fillcolor=fill)
        return img, label

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        path = self.images[idx]
        img, label = self._load(path)
        if self.augment:
            img, label = self._augment(img, label)
        img_arr = np.array(img, dtype=np.float32) / 255.0
        # Invert intensity so chromosome foreground tends to be high response.
        img_arr = 1.0 - img_arr
        label_arr = np.array(label, dtype=np.int64)
        img_tensor = torch.from_numpy(img_arr).unsqueeze(0)
        label_tensor = torch.from_numpy(label_arr)
        return {"image": img_tensor, "label": label_tensor, "filename": path.name}


class CombinedSyntheticPseudoDataset(Dataset):
    def __init__(self, synthetic: SegmentationDataset, pseudo: Optional[SegmentationDataset] = None):
        self.synthetic = synthetic
        self.pseudo = pseudo
        self.n_syn = len(synthetic)
        self.n_pse = len(pseudo) if pseudo is not None else 0

    def __len__(self) -> int:
        return self.n_syn + self.n_pse

    def __getitem__(self, idx: int):
        if idx < self.n_syn:
            item = self.synthetic[idx]
            item["is_pseudo"] = torch.tensor(0, dtype=torch.long)
            return item
        item = self.pseudo[idx - self.n_syn]
        item["is_pseudo"] = torch.tensor(1, dtype=torch.long)
        return item


def save_checkpoint(model: nn.Module, path: Path) -> None:
    ensure_dir(path.parent)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: nn.Module, path: Path, device: str) -> nn.Module:
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model


def compute_pixel_confusion(pred: np.ndarray, target: np.ndarray, num_classes: int = 4, ignore_index: int = 255) -> np.ndarray:
    valid = target != ignore_index
    pred = pred[valid].reshape(-1)
    target = target[valid].reshape(-1)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(target, pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[int(t), int(p)] += 1
    return cm


def metrics_from_cm(cm: np.ndarray, class_names: List[str]) -> Dict[str, float]:
    rows = {}
    ious = []
    dices = []
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        denom_iou = tp + fp + fn
        denom_dice = 2 * tp + fp + fn
        iou = float(tp / denom_iou) if denom_iou > 0 else 0.0
        dice = float((2 * tp) / denom_dice) if denom_dice > 0 else 0.0
        rows[f"iou_{name}"] = iou
        rows[f"dice_{name}"] = dice
        ious.append(iou)
        dices.append(dice)
    rows["mean_iou"] = float(np.mean(ious))
    rows["mean_dice"] = float(np.mean(dices))
    rows["pixel_accuracy"] = float(np.trace(cm) / max(cm.sum(), 1))
    return rows


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], out_path: Path, title: str) -> None:
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            ax.text(j, i, str(val), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def append_csv(path: Path, row: Dict[str, object]) -> None:
    ensure_dir(path.parent)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train_one_epoch(model: nn.Module, loader, optimizer, criterion, device: str, scaler=None, pseudo_weight: float = 1.0) -> float:
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        is_pseudo = batch.get("is_pseudo", torch.zeros(x.size(0), dtype=torch.long)).to(device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss_map = F.cross_entropy(logits, y, ignore_index=255, reduction="none")
                weights = torch.ones_like(loss_map)
                if pseudo_weight != 1.0:
                    w = torch.where(is_pseudo.view(-1, 1, 1) == 1, pseudo_weight, 1.0).to(loss_map.dtype)
                    weights = weights * w
                valid = (y != 255).float()
                loss = (loss_map * weights * valid).sum() / valid.sum().clamp_min(1.0)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss_map = F.cross_entropy(logits, y, ignore_index=255, reduction="none")
            weights = torch.ones_like(loss_map)
            if pseudo_weight != 1.0:
                w = torch.where(is_pseudo.view(-1, 1, 1) == 1, pseudo_weight, 1.0).to(loss_map.dtype)
                weights = weights * w
            valid = (y != 255).float()
            loss = (loss_map * weights * valid).sum() / valid.sum().clamp_min(1.0)
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        total += float(loss.detach().cpu()) * bs
        n += bs
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str, class_names: List[str]) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    n = 0
    cm = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y, ignore_index=255)
        pred = logits.argmax(dim=1).detach().cpu().numpy().astype(np.uint8)
        target = y.detach().cpu().numpy().astype(np.uint8)
        for p, t in zip(pred, target):
            cm += compute_pixel_confusion(p, t, num_classes=len(class_names))
        bs = x.size(0)
        total_loss += float(loss.detach().cpu()) * bs
        n += bs
    metrics = metrics_from_cm(cm, class_names)
    metrics["loss"] = total_loss / max(n, 1)
    metrics["cm"] = cm
    return metrics


def fit_model(model: nn.Module, train_loader, val_loader, device: str, epochs: int, lr: float, patience: int, out_ckpt: Path, log_csv: Path, class_names: List[str], pseudo_weight: float = 1.0) -> Dict[str, object]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler() if device.startswith("cuda") else None
    model.to(device)

    best_val = math.inf
    best_metrics = None
    bad_epochs = 0
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, None, device, scaler=scaler, pseudo_weight=pseudo_weight)
        val_metrics = evaluate(model, val_loader, device, class_names)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_mean_iou": val_metrics["mean_iou"],
            "val_mean_dice": val_metrics["mean_dice"],
            "val_pixel_accuracy": val_metrics["pixel_accuracy"],
        }
        append_csv(log_csv, row)
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | mIoU={val_metrics['mean_iou']:.4f} | Dice={val_metrics['mean_dice']:.4f}")

        if val_metrics["loss"] < best_val:
            best_val = float(val_metrics["loss"])
            best_metrics = val_metrics
            save_checkpoint(model, out_ckpt)
            bad_epochs = 0
            print("  saved best:", out_ckpt)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping after {patience} epochs without improvement.")
                break
    return best_metrics or {}
