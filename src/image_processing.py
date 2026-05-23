from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, skeletonize
from skimage.util import img_as_ubyte


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class SkeletonStats:
    filename: str
    source_group: str
    width: int
    height: int
    object_components: int
    skeleton_components: int
    total_endpoints: int
    total_branchpoints: int
    simple_path_components: int
    skeleton_pixels: int
    predicted_rule_label: str
    is_two_single_paths: bool
    is_valid_single_path: bool


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS])


def read_grayscale(path: Path) -> np.ndarray:
    """Read image as grayscale uint8."""
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.uint8)


def normalize_contrast(gray: np.ndarray) -> np.ndarray:
    """Light denoise + CLAHE contrast normalization for G-band chromosome images."""
    gray = cv2.medianBlur(gray, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def to_binary(gray: np.ndarray, min_object_area: int = 30) -> np.ndarray:
    """Convert grayscale chromosome image to binary foreground mask.

    The code tries both dark-object and bright-object assumptions, then chooses
    the mask with a more plausible foreground ratio.
    """
    if gray.ndim != 2:
        raise ValueError("to_binary expects a single-channel grayscale image")

    img = normalize_contrast(gray)
    if np.unique(img).size <= 2:
        # Already binary-like
        binary = img > 0
    else:
        try:
            t = threshold_otsu(img)
        except ValueError:
            t = int(np.mean(img))
        dark = img < t
        bright = img > t

        def score(mask: np.ndarray) -> float:
            ratio = mask.mean()
            # Most chromosome crops contain foreground but not the whole image.
            return abs(ratio - 0.18)

        binary = dark if score(dark) < score(bright) else bright

    binary = remove_small_objects(binary.astype(bool), min_size=min_object_area)
    binary = binary.astype(np.uint8)
    return binary


def zhang_suen_skeleton(binary: np.ndarray, pad: int = 12) -> np.ndarray:
    """Apply skimage padding + Zhang-Suen skeletonization.

    Padding reduces fake edge legs/feet when the chromosome touches image borders.
    The skeleton is cropped back to the original image size afterwards.
    """
    if binary.ndim != 2:
        raise ValueError("zhang_suen_skeleton expects a 2D binary image")
    padded = np.pad(binary.astype(bool), pad_width=pad, mode="constant", constant_values=False)
    try:
        skel = skeletonize(padded, method="zhang")
    except TypeError:
        # Older scikit-image versions do not expose the method argument.
        skel = skeletonize(padded)
    if pad > 0:
        skel = skel[pad:-pad, pad:-pad]
    return skel.astype(np.uint8)


def neighbor_count_map(skeleton: np.ndarray) -> np.ndarray:
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbors = cv2.filter2D(skeleton.astype(np.uint8), ddepth=-1, kernel=kernel)
    return neighbors


def skeleton_graph_stats(skeleton: np.ndarray) -> Tuple[int, int, int, int, int]:
    """Return skeleton components, endpoints, branchpoints, simple_path_components, pixels."""
    skel = skeleton.astype(bool)
    labeled = label(skel, connectivity=2)
    props = regionprops(labeled)
    total_endpoints = 0
    total_branchpoints = 0
    simple_paths = 0
    total_pixels = int(skel.sum())

    neighbors = neighbor_count_map(skel.astype(np.uint8))
    for prop in props:
        coords = prop.coords
        if len(coords) == 0:
            continue
        comp_mask = labeled == prop.label
        comp_neighbors = neighbors[comp_mask]
        endpoints = int(np.sum(comp_neighbors == 1))
        branchpoints = int(np.sum(comp_neighbors >= 3))
        total_endpoints += endpoints
        total_branchpoints += branchpoints
        if endpoints == 2 and branchpoints == 0:
            simple_paths += 1

    return len(props), total_endpoints, total_branchpoints, simple_paths, total_pixels


def classify_by_skeleton(binary: np.ndarray, skeleton: np.ndarray, min_skeleton_pixels: int = 10) -> Dict[str, object]:
    object_components = int(label(binary.astype(bool), connectivity=2).max())
    skel_components, endpoints, branchpoints, simple_paths, skel_pixels = skeleton_graph_stats(skeleton)

    is_valid_single_path = skel_components == 1 and simple_paths == 1 and endpoints == 2 and branchpoints == 0
    is_two_single_paths = skel_components == 2 and simple_paths == 2 and endpoints == 4 and branchpoints == 0

    if skel_pixels < min_skeleton_pixels or object_components == 0:
        label_name = "invalid_or_noise"
    elif is_valid_single_path:
        label_name = "single_path"
    elif is_two_single_paths:
        label_name = "two_single_paths"
    elif branchpoints > 0 or endpoints >= 3:
        label_name = "complex_overlap"
    else:
        label_name = "invalid_or_noise"

    return {
        "object_components": object_components,
        "skeleton_components": skel_components,
        "total_endpoints": endpoints,
        "total_branchpoints": branchpoints,
        "simple_path_components": simple_paths,
        "skeleton_pixels": skel_pixels,
        "predicted_rule_label": label_name,
        "is_two_single_paths": bool(is_two_single_paths),
        "is_valid_single_path": bool(is_valid_single_path),
    }


def save_debug_images(gray: np.ndarray, binary: np.ndarray, skeleton: np.ndarray, out_stem: Path) -> None:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(gray).save(out_stem.with_suffix(".gray.png"))
    Image.fromarray((binary.astype(np.uint8) * 255)).save(out_stem.with_suffix(".binary.png"))
    Image.fromarray((skeleton.astype(np.uint8) * 255)).save(out_stem.with_suffix(".skeleton.png"))

    # Overlay skeleton on grayscale for quick checking.
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay[skeleton.astype(bool)] = [0, 0, 255]
    cv2.imwrite(str(out_stem.with_suffix(".overlay.png")), overlay)


def resize_and_pad_gray(gray: np.ndarray, size: int = 224) -> np.ndarray:
    """Resize grayscale image while preserving aspect ratio, pad to square."""
    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("empty image")
    scale = min(size / h, size / w)
    new_h, new_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), dtype=np.uint8)
    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas
