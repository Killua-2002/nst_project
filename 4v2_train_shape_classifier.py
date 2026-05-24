
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from skimage.morphology import binary_dilation, binary_erosion, disk
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_LR,
    MODEL_DIR,
    RESIZED_DIR,
    SINGLE_CHROMOSOMES_DIR,
)
from shape_classifier_model import ShapeClassifierCCINet
from utils_image import clean_mask, ensure_dir, foreground_mask, list_images, read_gray
from utils_train import save_checkpoint


def _random_disc(mask: np.ndarray, value: bool, r_min: int, r_max: int) -> np.ndarray:
    h, w = mask.shape
    rr = random.randint(r_min, r_max)
    cy = random.randint(rr, max(rr, h - rr - 1))
    cx = random.randint(rr, max(rr, w - rr - 1))
    y, x = np.ogrid[:h, :w]
    disc = (x - cx) ** 2 + (y - cy) ** 2 <= rr ** 2
    out = mask.copy()
    out[disc] = value
    return out


def corrupt_mask(mask: np.ndarray) -> np.ndarray:
    """Create invalid/broken masks used as negative samples for shape classification."""
    out = mask.astype(bool).copy()
    h, w = out.shape

    mode = random.choice(["holes", "cut", "islands", "erode", "dilate", "fragment", "noise"])

    if mode == "holes":
        for _ in range(random.randint(2, 7)):
            out = _random_disc(out, False, max(2, h // 45), max(4, h // 15))

    elif mode == "cut":
        # remove a random rectangle/stripe through the chromosome body
        ys, xs = np.where(out)
        if len(xs) > 0:
            cx = int(np.median(xs))
            cy = int(np.median(ys))
        else:
            cx, cy = w // 2, h // 2
        if random.random() < 0.5:
            x0 = max(0, cx - random.randint(4, 18))
            x1 = min(w, cx + random.randint(4, 18))
            out[:, x0:x1] = False
        else:
            y0 = max(0, cy - random.randint(4, 18))
            y1 = min(h, cy + random.randint(4, 18))
            out[y0:y1, :] = False

    elif mode == "islands":
        for _ in range(random.randint(2, 6)):
            out = _random_disc(out, True, max(2, h // 55), max(4, h // 22))

    elif mode == "erode":
        out = binary_erosion(out, disk(random.randint(2, 5)))

    elif mode == "dilate":
        out = binary_dilation(out, disk(random.randint(4, 9)))

    elif mode == "fragment":
        if random.random() < 0.5:
            out[:, : random.randint(w // 3, w // 2)] = False
        else:
            out[: random.randint(h // 3, h // 2), :] = False

    else:  # noise
        noise = np.random.random((h, w)) < random.uniform(0.01, 0.05)
        out = np.logical_xor(out, noise)

    out = clean_mask(out, min_size=12)
    return out.astype(np.float32)


class ShapeDataset(Dataset):
    def __init__(self, image_paths: List[Path], image_size: int = 224, max_count: int = 4000):
        self.paths = list(image_paths)
        random.shuffle(self.paths)
        if max_count > 0:
            self.paths = self.paths[:max_count]
        self.image_size = image_size
        if len(self.paths) < 2:
            raise RuntimeError("Need at least 2 single chromosome images for shape-classifier training.")

    def __len__(self):
        # Each source can create one positive and one negative.
        return len(self.paths) * 2

    def _positive_mask(self, path: Path) -> np.ndarray:
        gray = read_gray(path, image_size=self.image_size)
        mask = clean_mask(foreground_mask(gray), min_size=max(12, self.image_size // 12))
        # Avoid empty foreground by falling back to threshold-like dark pixels.
        if mask.sum() < 8:
            mask = gray < np.percentile(gray, 45)
            mask = clean_mask(mask, min_size=8)
        return mask.astype(np.float32)

    def __getitem__(self, idx):
        path = self.paths[idx // 2]
        pos = (idx % 2 == 0)
        mask = self._positive_mask(path)
        if pos:
            y = 1
        else:
            mask = corrupt_mask(mask)
            y = 0
        x = torch.from_numpy(mask[None, :, :].astype(np.float32))
        return x, torch.tensor(y, dtype=torch.long)


def parse_args():
    p = argparse.ArgumentParser(description="Train classification model to validate single chromosome shape.")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    p.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    p.add_argument("--max-count", type=int, default=4000, help="Max single images used for this small classifier.")
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            pred = logits.argmax(1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
            loss_sum += float(loss.item()) * y.numel()
    return {"loss": loss_sum / max(1, total), "acc": correct / max(1, total)}


def main():
    args = parse_args()
    random.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)

    single_dir = RESIZED_DIR / "single_chromosomes"
    if len(list_images(single_dir)) < 2:
        single_dir = SINGLE_CHROMOSOMES_DIR
    paths = list_images(single_dir)
    print(f"[SHAPE-CLASSIFIER] using {len(paths)} single images from {single_dir}")

    ds = ShapeDataset(paths, image_size=args.image_size, max_count=args.max_count)
    val_len = max(20, int(len(ds) * 0.15))
    train_len = len(ds) - val_len
    train_ds, val_ds = random_split(ds, [train_len, val_len], generator=torch.Generator().manual_seed(1337))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device(args.device)
    model = ShapeClassifierCCINet(dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    ensure_dir(MODEL_DIR)
    best_path = MODEL_DIR / "shape_classifier_valid_nst_best.pth"
    best_val = 10**9
    wait = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        pbar = tqdm(train_loader, desc=f"Shape classifier epoch {epoch}/{args.epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * y.numel()
            total_n += int(y.numel())
            pbar.set_postfix(loss=total_loss / max(1, total_n))

        val = evaluate(model, val_loader, device)
        print({"epoch": epoch, "train_loss": total_loss / max(1, total_n), **{f"val_{k}": v for k, v in val.items()}})

        if val["loss"] < best_val:
            best_val = val["loss"]
            wait = 0
            save_checkpoint(best_path, model, {"epoch": epoch, "image_size": args.image_size, "model": "ShapeClassifierCCINet"})
        else:
            wait += 1
            if wait >= args.patience:
                print(f"[EARLY STOP] shape classifier stopped at epoch {epoch}")
                break

    print("[OK] Shape classifier saved:", best_path)


if __name__ == "__main__":
    main()
