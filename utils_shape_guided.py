
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label as cc_label
from skimage.morphology import binary_dilation, disk
from skimage.segmentation import watershed

from shape_classifier_model import shape_probability
from utils_image import (
    analyze_skeleton,
    clean_mask,
    foreground_mask,
    refine_binary_mask,
    refine_single_chromosome_shape,
    skeleton_quality_score,
)


@dataclass
class CandidatePair:
    name: str
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    score: float
    detail: Dict[str, float | int | str]


def _safe_mean(arr: np.ndarray, mask: np.ndarray) -> float:
    if mask is None or mask.sum() == 0:
        return 0.0
    return float(arr[mask].mean())


def _pca_split_seeds(fg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    coords = np.column_stack(np.where(fg))
    a = np.zeros_like(fg, dtype=bool)
    b = np.zeros_like(fg, dtype=bool)
    if len(coords) < 8:
        return a, b
    mean = coords.mean(axis=0)
    z = coords - mean
    try:
        _, _, vt = np.linalg.svd(z, full_matrices=False)
        axis = vt[0]
        proj = z @ axis
    except Exception:
        proj = coords[:, 0] - mean[0]

    q1, q2 = np.percentile(proj, [25, 75])
    a_coords = coords[proj <= q1]
    b_coords = coords[proj >= q2]
    a[a_coords[:, 0], a_coords[:, 1]] = True
    b[b_coords[:, 0], b_coords[:, 1]] = True
    return a, b


def _probability_seeds(prob: np.ndarray, label: np.ndarray, fg: np.ndarray, min_seed: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    p_a = prob[1] + 0.65 * prob[3]
    p_b = prob[2] + 0.65 * prob[3]
    p_c = prob[3]

    no_c = fg & (p_c < np.quantile(p_c[fg], 0.85) if fg.any() else True)
    diff = p_a - p_b

    a_seed = fg & no_c & ((label == 1) | (diff >= np.quantile(diff[fg], 0.80) if fg.any() else False))
    b_seed = fg & no_c & ((label == 2) | (diff <= np.quantile(diff[fg], 0.20) if fg.any() else False))

    if a_seed.sum() < min_seed or b_seed.sum() < min_seed:
        a_pca, b_pca = _pca_split_seeds(fg)
        if a_seed.sum() < min_seed:
            a_seed = a_pca
        if b_seed.sum() < min_seed:
            b_seed = b_pca

    a_seed = clean_mask(a_seed, min_size=5)
    b_seed = clean_mask(b_seed, min_size=5)

    # Ensure seeds do not overlap.
    both = a_seed & b_seed
    if both.any():
        a_seed[both] = diff[both] >= 0
        b_seed[both] = diff[both] < 0

    return a_seed.astype(bool), b_seed.astype(bool)


def watershed_ab_from_model(gray: np.ndarray, prob: np.ndarray, label: np.ndarray, min_object_size: int = 25):
    """Use model probabilities as seeds, but force final A/B to stay on real foreground.

    This is the key fix for holey segmentation:
    segmentation model says where A/B likely start;
    foreground from raw image gives the chromosome body;
    watershed assigns foreground pixels to A/B, so masks preserve NST shape.
    """
    fg = clean_mask(foreground_mask(gray), min_size=max(min_object_size, gray.shape[0] // 14))
    if fg.sum() < 10:
        return None

    a_seed, b_seed = _probability_seeds(prob, label, fg, min_seed=max(8, gray.shape[0] // 10))
    if a_seed.sum() == 0 or b_seed.sum() == 0:
        return None

    markers = np.zeros_like(gray, dtype=np.int32)
    markers[a_seed] = 1
    markers[b_seed] = 2

    # Use negative distance transform: inside body is basin-like and smoother than raw probability blobs.
    dist = ndi.distance_transform_edt(fg)
    ws = watershed(-dist, markers=markers, mask=fg)

    a = ws == 1
    b = ws == 2

    # C from model probability plus contact band between A and B.
    c_model = (prob[3] >= max(0.35, float(np.quantile(prob[3][fg], 0.90)))) & fg
    contact = binary_dilation(a, disk(2)) & binary_dilation(b, disk(2)) & fg
    c = clean_mask(c_model | contact, min_size=max(5, min_object_size // 3))

    # C belongs to both A and B in final masks.
    a = clean_mask(a | c, min_size=min_object_size)
    b = clean_mask(b | c, min_size=min_object_size)
    c = c & a & b
    return a.astype(bool), b.astype(bool), c.astype(bool)


def _candidate_score(
    name: str,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    prob: np.ndarray,
    fg: np.ndarray,
    shape_classifier=None,
    device=None,
) -> CandidatePair:
    a = a.astype(bool)
    b = b.astype(bool)
    c = c.astype(bool) & a & b

    if a.sum() < 8 or b.sum() < 8:
        return CandidatePair(name, a, b, c, -1e9, {"reason": "too_small"})

    # segmentation probability agreement
    p_a = prob[1] + prob[3]
    p_b = prob[2] + prob[3]
    p_c = prob[3]
    prob_score = _safe_mean(p_a, a) + _safe_mean(p_b, b) + 0.6 * _safe_mean(p_c, c)

    # foreground agreement: penalize masks outside real chromosome body and masks covering too little of body
    fg = fg.astype(bool)
    union = a | b
    outside = float((union & ~fg).sum()) / max(1, int(union.sum()))
    cover = float((union & fg).sum()) / max(1, int(fg.sum()))
    cover_score = 0.8 * min(cover, 1.0) - 1.6 * outside

    # skeleton shape prior: valid unbranched path is best
    info_a = analyze_skeleton(a, already_skeleton=False)
    info_b = analyze_skeleton(b, already_skeleton=False)
    skel_penalty = 0.010 * skeleton_quality_score(info_a) + 0.010 * skeleton_quality_score(info_b)

    # classification model score: does each mask look like a single NST?
    shape_a = shape_probability(shape_classifier, a, device=device)
    shape_b = shape_probability(shape_classifier, b, device=device)
    cls_score = 0.9 * (shape_a + shape_b)

    # discourage A/B being almost identical except overlap C
    inter = float((a & b).sum())
    small = max(1.0, float(min(a.sum(), b.sum())))
    identical_penalty = max(0.0, inter / small - 0.45) * 2.0

    score = prob_score + cover_score + cls_score - skel_penalty - identical_penalty

    detail = {
        "prob_score": float(prob_score),
        "cover_score": float(cover_score),
        "shape_A_prob": float(shape_a),
        "shape_B_prob": float(shape_b),
        "skel_penalty": float(skel_penalty),
        "identical_penalty": float(identical_penalty),
        "fg_cover": float(cover),
        "outside_ratio": float(outside),
        "A_status": info_a["status"],
        "B_status": info_b["status"],
        "A_endpoints": int(info_a["endpoints"]),
        "B_endpoints": int(info_b["endpoints"]),
        "A_branch_points": int(info_a["branch_points"]),
        "B_branch_points": int(info_b["branch_points"]),
    }
    return CandidatePair(name, a, b, c, float(score), detail)


def build_shape_guided_candidates(
    gray: np.ndarray,
    prob: np.ndarray,
    label: np.ndarray,
    raw_a: np.ndarray,
    raw_b: np.ndarray,
    raw_c: np.ndarray,
    base_a: np.ndarray,
    base_b: np.ndarray,
    base_c: np.ndarray,
    close_radius: int,
    hole_area: int,
    min_object_size: int,
    keep_largest: bool,
    shape_classifier=None,
    device=None,
) -> List[CandidatePair]:
    fg = clean_mask(foreground_mask(gray), min_size=max(min_object_size, gray.shape[0] // 14))
    candidates: List[CandidatePair] = []

    def add(name, a, b, c):
        candidates.append(_candidate_score(name, a, b, c, prob, fg, shape_classifier=shape_classifier, device=device))

    # Candidate 1: raw model output after normal morphology.
    add("segment_morphology", base_a, base_b, base_c)

    # Candidate 2: watershed assigns real foreground to A/B using model seeds.
    ws = watershed_ab_from_model(gray, prob, label, min_object_size=min_object_size)
    if ws is not None:
        add("classifier_shape_watershed", *ws)

    # Candidate 3/4: more aggressive hole filling / closing, selected by classifier+skeleton score.
    for r, h in [
        (max(1, close_radius + 1), max(hole_area, int(hole_area * 1.5))),
        (max(1, close_radius + 2), max(hole_area, int(hole_area * 2.0))),
    ]:
        a2, _ = refine_single_chromosome_shape(
            raw_a,
            foreground=fg,
            min_size=min_object_size,
            close_radius=r,
            hole_area=h,
            keep_largest=keep_largest,
            skeleton_repair=True,
        )
        b2, _ = refine_single_chromosome_shape(
            raw_b,
            foreground=fg,
            min_size=min_object_size,
            close_radius=r,
            hole_area=h,
            keep_largest=keep_largest,
            skeleton_repair=True,
        )
        c2 = refine_binary_mask(
            raw_c | (a2 & b2),
            foreground=fg,
            close_radius=max(1, r - 1),
            hole_area=max(32, h // 4),
            min_size=max(5, min_object_size // 3),
            keep_largest=False,
            foreground_margin=1,
        )
        c2 = c2 & a2 & b2
        add(f"classifier_shape_repair_r{r}_h{h}", a2 | c2, b2 | c2, c2)

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def choose_best_shape_guided_pair(*args, **kwargs) -> CandidatePair:
    candidates = build_shape_guided_candidates(*args, **kwargs)
    if not candidates:
        raise RuntimeError("No A/B candidate generated.")
    return candidates[0]
