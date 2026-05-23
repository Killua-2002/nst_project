from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage import filters, morphology, measure, util

import config
from src.utils import list_images, safe_name


def read_grayscale(path: Path, image_size: int | None = None) -> np.ndarray:
    """Read image as uint8 grayscale, optionally resized to square."""
    img = Image.open(path).convert("L")
    if image_size is not None:
        img.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
        canvas = Image.new("L", (image_size, image_size), 255)
        x = (image_size - img.width) // 2
        y = (image_size - img.height) // 2
        canvas.paste(img, (x, y))
        img = canvas
    return np.array(img, dtype=np.uint8)


def binarize_chromosome(gray: np.ndarray) -> np.ndarray:
    """Return foreground=True binary mask.

    Most G-band chromosome crops are dark object on bright background. The function
    uses Otsu threshold and chooses the foreground candidate with a reasonable area.
    """
    if gray.ndim != 2:
        raise ValueError("binarize_chromosome expects grayscale image")
    # light denoise without killing band structures
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    try:
        threshold = filters.threshold_otsu(blurred)
    except ValueError:
        threshold = int(np.mean(blurred))

    dark_fg = blurred < threshold
    light_fg = blurred > threshold

    def score(mask: np.ndarray) -> float:
        frac = float(mask.mean())
        # prefer object area neither tiny nor whole image; chromosomes are usually minority pixels
        if frac < 0.003 or frac > 0.85:
            return 999.0
        return abs(frac - 0.18)

    binary = dark_fg if score(dark_fg) <= score(light_fg) else light_fg
    binary = morphology.remove_small_objects(binary.astype(bool), min_size=20)
    binary = morphology.remove_small_holes(binary, area_threshold=20)
    return binary.astype(bool)


def zhang_suen_thinning(binary: np.ndarray) -> np.ndarray:
    """Pure Zhang-Suen thinning implementation.

    Input and output are boolean arrays where True means foreground. This is kept in
    the project because some environments do not expose skimage's method='zhang'.
    """
    img = binary.astype(np.uint8).copy()
    changed = True

    def neighbours(x: int, y: int):
        # p2,p3,p4,p5,p6,p7,p8,p9 clockwise
        return [
            img[x - 1, y], img[x - 1, y + 1], img[x, y + 1], img[x + 1, y + 1],
            img[x + 1, y], img[x + 1, y - 1], img[x, y - 1], img[x - 1, y - 1]
        ]

    def transitions(ns) -> int:
        seq = ns + [ns[0]]
        return sum((seq[i] == 0 and seq[i + 1] == 1) for i in range(8))

    rows, cols = img.shape
    while changed:
        changed = False
        to_remove = []
        for x in range(1, rows - 1):
            for y in range(1, cols - 1):
                if img[x, y] != 1:
                    continue
                ns = neighbours(x, y)
                n_sum = sum(ns)
                if not (2 <= n_sum <= 6):
                    continue
                if transitions(ns) != 1:
                    continue
                p2, p3, p4, p5, p6, p7, p8, p9 = ns
                if p2 * p4 * p6 != 0:
                    continue
                if p4 * p6 * p8 != 0:
                    continue
                to_remove.append((x, y))
        if to_remove:
            changed = True
            for x, y in to_remove:
                img[x, y] = 0

        to_remove = []
        for x in range(1, rows - 1):
            for y in range(1, cols - 1):
                if img[x, y] != 1:
                    continue
                ns = neighbours(x, y)
                n_sum = sum(ns)
                if not (2 <= n_sum <= 6):
                    continue
                if transitions(ns) != 1:
                    continue
                p2, p3, p4, p5, p6, p7, p8, p9 = ns
                if p2 * p4 * p8 != 0:
                    continue
                if p2 * p6 * p8 != 0:
                    continue
                to_remove.append((x, y))
        if to_remove:
            changed = True
            for x, y in to_remove:
                img[x, y] = 0
    return img.astype(bool)


def skeletonize_with_padding(binary: np.ndarray, pad: int = 8) -> np.ndarray:
    """Pad border before thinning to reduce false legs at image boundary."""
    padded = np.pad(binary.astype(bool), pad_width=pad, mode="constant", constant_values=False)
    # Use our Zhang-Suen implementation to satisfy project requirement.
    skel = zhang_suen_thinning(padded)
    if pad > 0:
        skel = skel[pad:-pad, pad:-pad]
    return skel.astype(bool)


def count_neighbors(skel: np.ndarray) -> np.ndarray:
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    conv = cv2.filter2D(skel.astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT)
    # subtract center marker 10
    return np.where(skel, conv - 10, 0)


def connected_components_bool(mask: np.ndarray) -> Tuple[int, np.ndarray]:
    labels = measure.label(mask.astype(bool), connectivity=2)
    return int(labels.max()), labels


def _component_endpoint_junction_counts(skel: np.ndarray, labels: np.ndarray, comp_id: int):
    comp = labels == comp_id
    neigh = count_neighbors(comp)
    endpoints = int(np.logical_and(comp, neigh == 1).sum())
    junctions = int(np.logical_and(comp, neigh >= 3).sum())
    pixels = int(comp.sum())
    return endpoints, junctions, pixels


def analyze_skeleton(binary: np.ndarray, skel: np.ndarray) -> Dict:
    """Return skeleton statistics for validation and filtering.

    valid_single_line: one connected skeleton component, exactly 2 endpoints, no junctions.
    valid_two_line_candidate:
      - either two separate unbranched components; or
      - one connected component with around 4 endpoints and at least 1 junction.
    """
    n_bin, bin_labels = connected_components_bool(binary)
    n_skel, skel_labels = connected_components_bool(skel)
    neigh = count_neighbors(skel)
    endpoints = int(np.logical_and(skel, neigh == 1).sum())
    junctions = int(np.logical_and(skel, neigh >= 3).sum())
    skeleton_pixels = int(skel.sum())
    foreground_pixels = int(binary.sum())

    component_stats = []
    single_like_components = 0
    for comp_id in range(1, n_skel + 1):
        ep, jn, px = _component_endpoint_junction_counts(skel, skel_labels, comp_id)
        component_stats.append({"component": comp_id, "endpoints": ep, "junctions": jn, "pixels": px})
        if ep == 2 and jn == 0 and px >= 8:
            single_like_components += 1

    valid_single_line = (n_skel == 1 and endpoints == 2 and junctions == 0)
    valid_two_separate_lines = (n_skel == 2 and single_like_components == 2)
    valid_two_crossing_lines = (n_skel == 1 and endpoints >= 4 and endpoints <= 6 and junctions >= 1)
    valid_two_line_candidate = bool(valid_two_separate_lines or valid_two_crossing_lines)

    return {
        "binary_components": n_bin,
        "skeleton_components": n_skel,
        "endpoints": endpoints,
        "junctions": junctions,
        "skeleton_pixels": skeleton_pixels,
        "foreground_pixels": foreground_pixels,
        "valid_single_line": bool(valid_single_line),
        "valid_two_line_candidate": valid_two_line_candidate,
        "valid_two_separate_lines": bool(valid_two_separate_lines),
        "valid_two_crossing_lines": bool(valid_two_crossing_lines),
        "component_stats": component_stats,
    }


def save_debug_images(gray: np.ndarray, binary: np.ndarray, skel: np.ndarray, out_prefix: Path) -> None:
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(gray).save(str(out_prefix) + "_gray.png")
    Image.fromarray((binary.astype(np.uint8) * 255)).save(str(out_prefix) + "_binary.png")
    Image.fromarray((skel.astype(np.uint8) * 255)).save(str(out_prefix) + "_skeleton.png")

    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    overlay[binary] = np.clip(0.65 * overlay[binary] + np.array([0, 100, 0]), 0, 255)
    overlay[skel] = np.array([255, 0, 0])
    Image.fromarray(overlay.astype(np.uint8)).save(str(out_prefix) + "_overlay.png")


def process_one_image(path: Path, out_dir: Path, image_size: int = config.IMAGE_SIZE, pad: int = 8) -> Dict:
    gray = read_grayscale(path, image_size=image_size)
    binary = binarize_chromosome(gray)
    skel = skeletonize_with_padding(binary, pad=pad)
    stats = analyze_skeleton(binary, skel)
    prefix = Path(out_dir) / safe_name(path)
    save_debug_images(gray, binary, skel, prefix)
    stats.update({"file": str(path), "stem": path.stem})
    return stats


def process_folder(folder: Path, out_dir: Path, image_size: int = config.IMAGE_SIZE, pad: int = 8) -> pd.DataFrame:
    folder = Path(folder)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for img_path in list_images(folder):
        try:
            rows.append(process_one_image(img_path, out_dir, image_size=image_size, pad=pad))
        except Exception as exc:
            rows.append({"file": str(img_path), "error": repr(exc)})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "skeleton_stats.csv", index=False)
    return df
