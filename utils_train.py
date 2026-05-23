from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from config import NUM_CLASSES
from utils_image import list_images, ensure_dir


class SegmentationDataset(Dataset):
    def __init__(self, image_dir: Path, label_dir: Path, augment: bool = False, image_size: int = 224):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.augment = augment
        self.image_size = image_size

        images = list_images(self.image_dir)
        self.items = []
        for img_path in images:
            label_path = self.label_dir / f"{img_path.stem}.png"
            if label_path.exists():
                self.items.append((img_path, label_path))
        if not self.items:
            raise RuntimeError(f"No image/label pairs found in {self.image_dir} and {self.label_dir}")

    def __len__(self):
        return len(self.items)

    def _load_pair(self, img_path: Path, label_path: Path):
        img = Image.open(img_path).convert("L").resize((self.image_size, self.image_size), Image.BILINEAR)
        lab = Image.open(label_path).convert("L").resize((self.image_size, self.image_size), Image.NEAREST)
        return img, lab

    def _augment(self, img, lab):
        import random
        # RandomResizedCrop equivalent for paired image/label.
        if random.random() < 0.80:
            scale = random.uniform(0.75, 1.0)
            crop = int(self.image_size * scale)
            if crop < self.image_size:
                top = random.randint(0, self.image_size - crop)
                left = random.randint(0, self.image_size - crop)
                img = TF.resized_crop(img, top, left, crop, crop, (self.image_size, self.image_size), interpolation=TF.InterpolationMode.BILINEAR)
                lab = TF.resized_crop(lab, top, left, crop, crop, (self.image_size, self.image_size), interpolation=TF.InterpolationMode.NEAREST)

        if random.random() < 0.5:
            img = TF.hflip(img)
            lab = TF.hflip(lab)
        if random.random() < 0.5:
            img = TF.vflip(img)
            lab = TF.vflip(lab)

        angle = random.uniform(-15, 15)
        img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=255)
        lab = TF.rotate(lab, angle, interpolation=TF.InterpolationMode.NEAREST, fill=0)
        return img, lab

    def __getitem__(self, idx):
        img_path, label_path = self.items[idx]
        img, lab = self._load_pair(img_path, label_path)
        if self.augment:
            img, lab = self._augment(img, lab)

        x = TF.to_tensor(img)  # [1,H,W], 0..1
        y = torch.from_numpy(np.asarray(lab, dtype=np.int64))
        y = torch.clamp(y, min=0, max=NUM_CLASSES - 1)
        return x, y


class CombinedSegmentationDataset(Dataset):
    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets
        self.cumulative = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cumulative.append(total)

    def __len__(self):
        return self.cumulative[-1] if self.cumulative else 0

    def __getitem__(self, idx):
        for ds_idx, end in enumerate(self.cumulative):
            start = 0 if ds_idx == 0 else self.cumulative[ds_idx - 1]
            if idx < end:
                return self.datasets[ds_idx][idx - start]
        raise IndexError(idx)


def save_checkpoint(path: Path, model, extra: Optional[dict] = None):
    ensure_dir(path.parent)
    payload = {"model_state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: Path, model, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    return ckpt


def pixel_confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    pred = pred.view(-1).long()
    target = target.view(-1).long()
    mask = (target >= 0) & (target < num_classes)
    inds = num_classes * target[mask] + pred[mask]
    cm = torch.bincount(inds, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return cm


def metrics_from_cm(cm: torch.Tensor) -> Dict[str, float]:
    cm = cm.float()
    diag = torch.diag(cm)
    total = cm.sum().clamp(min=1)
    pixel_acc = diag.sum() / total
    precision = diag / cm.sum(dim=0).clamp(min=1)
    recall = diag / cm.sum(dim=1).clamp(min=1)
    iou = diag / (cm.sum(dim=1) + cm.sum(dim=0) - diag).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
    return {
        "pixel_acc": float(pixel_acc.item()),
        "mean_iou": float(iou.mean().item()),
        "mean_f1": float(f1.mean().item()),
        "mean_precision": float(precision.mean().item()),
        "mean_recall": float(recall.mean().item()),
    }


def write_history_csv(path: Path, rows: List[Dict]):
    ensure_dir(path.parent)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_confusion_matrix(path: Path, cm: torch.Tensor, class_names: List[str]):
    import matplotlib.pyplot as plt
    ensure_dir(path.parent)
    arr = cm.cpu().numpy()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(arr)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, str(int(arr[i, j])), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_json(path: Path, data: Dict):
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
