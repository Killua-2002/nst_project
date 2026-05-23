from __future__ import annotations

import csv
import inspect
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    from skimage.filters import threshold_otsu
    from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes, binary_closing, binary_opening, binary_dilation, binary_erosion, disk
    try:
        from skimage.morphology import closing as morph_closing, opening as morph_opening
    except Exception:
        morph_closing, morph_opening = binary_closing, binary_opening
    from skimage.measure import label as cc_label
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "Missing scikit-image. Install with: pip install scikit-image"
    ) from exc

from config import IMAGE_EXTS


def list_images(folder: Path) -> List[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS])


def ensure_dir(path: Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)




def _remove_small_objects_compat(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Version-safe wrapper for skimage.remove_small_objects.

    The broken version used max_size on Colab builds that only accept min_size,
    then recursively called itself in the except block.  This wrapper checks the
    installed scikit-image signature and never recurses.
    """
    mask = mask.astype(bool)
    params = inspect.signature(remove_small_objects).parameters
    # New scikit-image builds renamed the size threshold to max_size;
    # old Colab builds still use min_size.
    if "max_size" in params:
        try:
            return remove_small_objects(mask, max_size=min_size)
        except TypeError:
            pass
    if "min_size" in params:
        try:
            return remove_small_objects(mask, min_size=min_size)
        except TypeError:
            pass
    return remove_small_objects(mask, min_size)


def _remove_small_holes_compat(mask: np.ndarray, hole_area: int) -> np.ndarray:
    """Version-safe wrapper for skimage.remove_small_holes."""
    mask = mask.astype(bool)
    params = inspect.signature(remove_small_holes).parameters
    if "max_size" in params:
        try:
            return remove_small_holes(mask, max_size=hole_area)
        except TypeError:
            pass
    if "area_threshold" in params:
        try:
            return remove_small_holes(mask, area_threshold=hole_area)
        except TypeError:
            pass
    return remove_small_holes(mask, hole_area)


def read_gray(path: Path, image_size: int | None = None) -> np.ndarray:
    img = Image.open(path).convert("L")
    if image_size is not None:
        img = img.resize((image_size, image_size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def save_gray(path: Path, arr: np.ndarray) -> None:
    ensure_dir(path.parent)
    Image.fromarray(arr.astype(np.uint8), mode="L").save(path)


def save_label(path: Path, label: np.ndarray) -> None:
    ensure_dir(path.parent)
    Image.fromarray(label.astype(np.uint8), mode="L").save(path)


def foreground_mask(gray: np.ndarray) -> np.ndarray:
    """Return chromosome foreground as boolean mask.

    Most G-band chromosome crops are dark chromosome on bright background.
    This function also handles inverted images by choosing the more plausible
    small connected foreground.
    """
    if gray.ndim != 2:
        raise ValueError("foreground_mask expects grayscale image")
    # Avoid otsu crash on constant images.
    if gray.max() == gray.min():
        return np.zeros_like(gray, dtype=bool)

    t = threshold_otsu(gray)
    dark = gray < t
    light = gray > t

    def score(mask: np.ndarray) -> float:
        ratio = float(mask.mean())
        # chromosome usually occupies around 1% to 60%.
        if ratio < 0.002 or ratio > 0.75:
            return 999.0
        return abs(ratio - 0.18)

    return dark if score(dark) <= score(light) else light


def clean_mask(mask: np.ndarray, min_size: int = 30) -> np.ndarray:
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return mask
    mask = morph_opening(mask, disk(1))
    mask = morph_closing(mask, disk(1))
    mask = _remove_small_objects_compat(mask, min_size=min_size)
    return mask.astype(bool)


def largest_component(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return mask
    lab = cc_label(mask, connectivity=2)
    if lab.max() == 0:
        return mask
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    return lab == counts.argmax()





def skeleton_quality_score(info: Dict[str, int | str | float]) -> float:
    """Lower is better. Used to pick the smoothest chromosome-like mask.

    A good A/B chromosome mask should be one connected object whose skeleton is
    one clean path: 2 endpoints, 0 branch points. This does not replace model
    learning; it is a post-processing shape prior to stop holey / fragmented
    segmentation output.
    """
    status = str(info.get("status", ""))
    if status == "valid_single_path":
        return 0.0
    pixels = int(info.get("skeleton_pixels", 0))
    if pixels <= 0:
        return 10_000.0
    components = int(info.get("components", 0))
    endpoints = int(info.get("endpoints", 0))
    branches = int(info.get("branch_points", 0))
    return (
        abs(components - 1) * 80.0
        + abs(endpoints - 2) * 35.0
        + branches * 20.0
        + (0 if components >= 1 else 200.0)
    )


def refine_binary_mask(
    mask: np.ndarray,
    foreground: np.ndarray | None = None,
    close_radius: int = 2,
    hole_area: int = 512,
    min_size: int = 30,
    keep_largest: bool = True,
    foreground_margin: int = 2,
) -> np.ndarray:
    """Clean a predicted chromosome mask.

    Main fixes:
    - fill internal holes so A/B are not spotty;
    - close tiny gaps in the chromosome body;
    - remove small islands;
    - optionally keep one main chromosome component;
    - optionally constrain to a dilated foreground extracted from the original image.
    """
    original = mask.astype(bool)
    mask = original.copy()

    if foreground is not None and foreground.any():
        fg = foreground.astype(bool)
        if foreground_margin > 0:
            fg = binary_dilation(fg, disk(foreground_margin))
        restricted = mask & fg
        # Do not destroy the prediction if Otsu foreground is imperfect.
        if restricted.sum() >= max(8, int(mask.sum() * 0.35)):
            mask = restricted

    if close_radius > 0:
        mask = morph_closing(mask, disk(close_radius))
    if hole_area > 0:
        mask = _remove_small_holes_compat(mask, hole_area=hole_area)
    if min_size > 0:
        mask = _remove_small_objects_compat(mask, min_size=min_size)
    if keep_largest:
        mask = largest_component(mask)
    if close_radius > 0:
        mask = morph_closing(mask, disk(max(1, close_radius // 2)))
    if hole_area > 0:
        mask = _remove_small_holes_compat(mask, hole_area=max(16, hole_area // 2))
    return mask.astype(bool)


def refine_single_chromosome_shape(
    mask: np.ndarray,
    foreground: np.ndarray | None = None,
    min_size: int = 30,
    close_radius: int = 2,
    hole_area: int = 512,
    keep_largest: bool = True,
    skeleton_repair: bool = True,
) -> Tuple[np.ndarray, Dict[str, int | str | float]]:
    """Refine A or B so the output keeps chromosome-like shape.

    The model gives pixel probabilities. This function adds a morphology +
    Zhang-Suen quality prior after prediction. It tries a few close/fill settings
    and chooses the mask whose skeleton is closest to a single unbranched path.
    """
    base = refine_binary_mask(
        mask,
        foreground=foreground,
        close_radius=close_radius,
        hole_area=hole_area,
        min_size=min_size,
        keep_largest=keep_largest,
    )
    best = base
    best_info = analyze_skeleton(best, already_skeleton=False)
    best_score = skeleton_quality_score(best_info)

    if not skeleton_repair:
        return best.astype(bool), best_info

    # Try stronger smoothing/filling when the first mask is holey or fragmented.
    close_values = sorted(set([1, close_radius, close_radius + 1, close_radius + 2]))
    hole_values = sorted(set([64, hole_area, hole_area * 2, hole_area * 4]))
    for cr in close_values:
        for ha in hole_values:
            cand = refine_binary_mask(
                mask,
                foreground=foreground,
                close_radius=cr,
                hole_area=ha,
                min_size=min_size,
                keep_largest=keep_largest,
            )
            info = analyze_skeleton(cand, already_skeleton=False)
            score = skeleton_quality_score(info)
            # Prefer skeleton-valid candidates, but avoid a mask that shrinks too much.
            area_ratio = cand.sum() / max(1, base.sum())
            if area_ratio < 0.55 or area_ratio > 1.80:
                score += 50.0
            if score < best_score:
                best, best_info, best_score = cand, info, score

    return best.astype(bool), best_info


def zhang_suen_skeleton(mask: np.ndarray, pad: int = 4) -> np.ndarray:
    """Skeletonize binary mask with border padding.

    skimage.morphology.skeletonize uses a thinning algorithm suitable for
    Zhang-Suen style skeleton extraction on binary objects. Padding reduces
    fake skeleton legs created at image borders.
    """
    mask = mask.astype(bool)
    padded = np.pad(mask, pad_width=pad, mode="constant", constant_values=False)
    skel = skeletonize(padded)
    if pad > 0:
        skel = skel[pad:-pad, pad:-pad]
    return skel.astype(bool)


def _neighbor_count(binary: np.ndarray) -> np.ndarray:
    b = binary.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    count = cv2.filter2D(b, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    return count - b


def analyze_skeleton(mask_or_skeleton: np.ndarray, already_skeleton: bool = False) -> Dict[str, int | str | float]:
    if already_skeleton:
        skel = mask_or_skeleton.astype(bool)
    else:
        skel = zhang_suen_skeleton(mask_or_skeleton)

    n = _neighbor_count(skel)
    endpoints = int(np.logical_and(skel, n == 1).sum())
    branch_points = int(np.logical_and(skel, n >= 3).sum())
    components = int(cc_label(skel, connectivity=2).max())
    pixels = int(skel.sum())

    # A clean single chromosome skeleton is expected to be one unbranched path.
    if pixels == 0:
        status = "empty"
    elif components != 1:
        status = "invalid_components"
    elif endpoints != 2:
        status = "invalid_endpoints"
    elif branch_points != 0:
        status = "invalid_branch_points"
    else:
        status = "valid_single_path"

    return {
        "status": status,
        "components": components,
        "endpoints": endpoints,
        "branch_points": branch_points,
        "skeleton_pixels": pixels,
    }


def preprocess_folder(
    src_dir: Path,
    dst_dir: Path,
    skeleton_dir: Path | None,
    image_size: int,
    use_skeleton: bool,
    save_skeleton_debug: bool,
    csv_path: Path,
) -> List[Dict[str, str | int | float]]:
    ensure_dir(dst_dir)
    if skeleton_dir is not None:
        ensure_dir(skeleton_dir)
    rows = []
    for path in list_images(src_dir):
        gray = read_gray(path, image_size=image_size)
        out_path = dst_dir / f"{path.stem}.png"
        save_gray(out_path, gray)

        row: Dict[str, str | int | float] = {
            "filename": out_path.name,
            "source_path": str(path),
            "resized_path": str(out_path),
            "image_size": image_size,
        }

        if use_skeleton:
            mask = clean_mask(foreground_mask(gray), min_size=max(10, image_size // 10))
            skel = zhang_suen_skeleton(mask, pad=4)
            info = analyze_skeleton(skel, already_skeleton=True)
            row.update(info)
            if save_skeleton_debug and skeleton_dir is not None:
                skel_path = skeleton_dir / f"{path.stem}_skeleton.png"
                save_gray(skel_path, (skel.astype(np.uint8) * 255))
                row["skeleton_path"] = str(skel_path)
        else:
            row.update({
                "status": "skipped",
                "components": -1,
                "endpoints": -1,
                "branch_points": -1,
                "skeleton_pixels": -1,
            })

        rows.append(row)

    ensure_dir(csv_path.parent)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        else:
            f.write("filename,status\n")
    return rows


def make_overlay(gray: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray, mask_c: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    base = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    color = base.copy()
    # A = red, B = green, C = yellow for easy visual checking.
    color[mask_a.astype(bool)] = np.array([255, 60, 60], dtype=np.float32)
    color[mask_b.astype(bool)] = np.array([60, 255, 60], dtype=np.float32)
    color[mask_c.astype(bool)] = np.array([255, 230, 40], dtype=np.float32)
    out = (base * (1 - alpha) + color * alpha).clip(0, 255).astype(np.uint8)
    return out


def save_rgb(path: Path, arr: np.ndarray) -> None:
    ensure_dir(path.parent)
    Image.fromarray(arr.astype(np.uint8), mode="RGB").save(path)


def read_label(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def write_csv(path: Path, rows: List[Dict]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
