from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def list_images(folder: Path) -> List[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS])


def ensure_dir(path: Path) -> Path:
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def read_gray(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        pil = Image.open(path).convert("L")
        arr = np.array(pil)
    return arr


def resize_square(gray: np.ndarray, size: int = 224) -> np.ndarray:
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)


def otsu_foreground_mask(gray: np.ndarray) -> np.ndarray:
    """Return foreground mask for chromosome, robust to dark-on-light or light-on-dark."""
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m1 = th > 0
    m2 = ~m1

    def score(mask: np.ndarray) -> float:
        # Prefer smaller foreground not touching full background too much.
        area = mask.mean()
        if area <= 0.001 or area >= 0.95:
            return -1e9
        # Chromosomes usually occupy less than half image.
        return -abs(area - 0.12)

    mask = m1 if score(m1) > score(m2) else m2

    # Keep largest connected component to remove dust.
    mask_uint = mask.astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint, 8)
    if num > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = 1 + int(np.argmax(areas))
        mask = labels == keep

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1) > 0
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1) > 0
    return mask.astype(np.uint8)


def crop_object(gray: np.ndarray, mask: np.ndarray, pad: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return gray.copy(), mask.copy()
    y1, y2 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, gray.shape[0])
    x1, x2 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, gray.shape[1])
    return gray[y1:y2, x1:x2].copy(), mask[y1:y2, x1:x2].copy()


def rotate_image_and_mask(img: np.ndarray, mask: np.ndarray, angle: float, bg_value: int = 255) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    nw = int((h * sin) + (w * cos))
    nh = int((h * cos) + (w * sin))
    M[0, 2] += (nw / 2) - center[0]
    M[1, 2] += (nh / 2) - center[1]
    rot_img = cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=bg_value)
    rot_mask = cv2.warpAffine(mask.astype(np.uint8) * 255, M, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=0) > 0
    return rot_img, rot_mask.astype(np.uint8)


def resize_object(img: np.ndarray, mask: np.ndarray, target_long_side: int) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    if max(h, w) == 0:
        return img, mask
    scale = target_long_side / max(h, w)
    nw, nh = max(2, int(w * scale)), max(2, int(h * scale))
    img2 = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    mask2 = cv2.resize(mask.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST) > 0
    return img2, mask2.astype(np.uint8)


def paste_object(canvas: np.ndarray, obj: np.ndarray, mask: np.ndarray, center_xy: Tuple[int, int]) -> np.ndarray:
    out = canvas.copy()
    H, W = canvas.shape[:2]
    h, w = obj.shape[:2]
    cx, cy = center_xy
    x1 = int(cx - w // 2); y1 = int(cy - h // 2)
    x2 = x1 + w; y2 = y1 + h

    ox1 = max(0, -x1); oy1 = max(0, -y1)
    ox2 = w - max(0, x2 - W); oy2 = h - max(0, y2 - H)
    tx1 = max(0, x1); ty1 = max(0, y1)
    tx2 = min(W, x2); ty2 = min(H, y2)

    if tx1 >= tx2 or ty1 >= ty2:
        return out
    roi = out[ty1:ty2, tx1:tx2]
    obj_roi = obj[oy1:oy2, ox1:ox2]
    mask_roi = mask[oy1:oy2, ox1:ox2] > 0
    roi[mask_roi] = np.minimum(roi[mask_roi], obj_roi[mask_roi])
    out[ty1:ty2, tx1:tx2] = roi
    return out


def paste_mask(mask_canvas: np.ndarray, mask: np.ndarray, center_xy: Tuple[int, int]) -> np.ndarray:
    out = mask_canvas.copy()
    H, W = mask_canvas.shape[:2]
    h, w = mask.shape[:2]
    cx, cy = center_xy
    x1 = int(cx - w // 2); y1 = int(cy - h // 2)
    x2 = x1 + w; y2 = y1 + h

    ox1 = max(0, -x1); oy1 = max(0, -y1)
    ox2 = w - max(0, x2 - W); oy2 = h - max(0, y2 - H)
    tx1 = max(0, x1); ty1 = max(0, y1)
    tx2 = min(W, x2); ty2 = min(H, y2)
    if tx1 >= tx2 or ty1 >= ty2:
        return out
    roi = out[ty1:ty2, tx1:tx2]
    roi[mask[oy1:oy2, ox1:ox2] > 0] = 1
    out[ty1:ty2, tx1:tx2] = roi
    return out


def label_to_masks(label: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # label: 0 background, 1 A-only, 2 B-only, 3 overlap C.
    mask_a = ((label == 1) | (label == 3)).astype(np.uint8) * 255
    mask_b = ((label == 2) | (label == 3)).astype(np.uint8) * 255
    mask_c = (label == 3).astype(np.uint8) * 255
    return mask_a, mask_b, mask_c


def save_label_and_masks(label: np.ndarray, out_label: Path, out_a: Path, out_b: Path, out_c: Path) -> None:
    ensure_dir(out_label.parent); ensure_dir(out_a.parent); ensure_dir(out_b.parent); ensure_dir(out_c.parent)
    Image.fromarray(label.astype(np.uint8)).save(out_label)
    a, b, c = label_to_masks(label)
    Image.fromarray(a).save(out_a)
    Image.fromarray(b).save(out_b)
    Image.fromarray(c).save(out_c)


def overlay_abc(gray: np.ndarray, label: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Create RGB overlay: A=red, B=green, C=blue. No dependency on matplotlib."""
    if gray.ndim == 2:
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    else:
        rgb = gray.copy()
    overlay = rgb.copy().astype(np.float32)
    colors = {
        1: np.array([255, 30, 30], dtype=np.float32),
        2: np.array([30, 220, 30], dtype=np.float32),
        3: np.array([30, 80, 255], dtype=np.float32),
    }
    for cls, color in colors.items():
        m = label == cls
        overlay[m] = (1 - alpha) * overlay[m] + alpha * color
    return np.clip(overlay, 0, 255).astype(np.uint8)
