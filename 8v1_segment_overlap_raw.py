import argparse
import csv
import importlib.util
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from config import DEFAULT_DROPOUT, DEFAULT_IMAGE_SIZE, MODEL_DIR, OVERLAP_RESULT_DIR, RESIZED_DIR
from utils_image import (
    analyze_skeleton,
    clean_mask,
    ensure_dir,
    foreground_mask,
    list_images,
    make_overlay,
    read_gray,
    refine_binary_mask,
    refine_single_chromosome_shape,
    save_gray,
    save_label,
    save_rgb,
    zhang_suen_skeleton,
)


def _load_models_module():
    spec = importlib.util.spec_from_file_location("models_v1", Path(__file__).resolve().parent / "4v1_models.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args():
    p = argparse.ArgumentParser(description="Segment overlap_raw into chromosome A, chromosome B, and overlap region C.")
    p.add_argument("--model", choices=["student", "teacher"], default="student")
    p.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    p.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    p.add_argument("--prob-threshold", type=float, default=0.45)
    p.add_argument("--keep-largest", action="store_true", help="Keep largest connected component of A and B masks.")
    p.add_argument("--no-shape-postprocess", action="store_true", help="Disable morphology hole filling / smoothing.")
    p.add_argument("--no-skeleton-repair", action="store_true", help="Disable skeleton-guided candidate selection.")
    p.add_argument("--close-radius", type=int, default=2, help="Morphological closing radius for A/B shape repair.")
    p.add_argument("--hole-area", type=int, default=768, help="Fill holes up to this area inside A/B masks.")
    p.add_argument("--min-object-size", type=int, default=35, help="Remove connected components smaller than this.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _load_checkpoint(path, model, device):
    ckpt = torch.load(path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    return model


def _predict(model, gray: np.ndarray, device):
    x = torch.from_numpy(gray.astype(np.float32) / 255.0)[None, None].to(device)
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    label = prob.argmax(axis=0).astype(np.uint8)
    confidence = float(prob.max(axis=0).mean())
    return prob, label, confidence


def _abc_from_prob(
    prob: np.ndarray,
    label: np.ndarray,
    gray: np.ndarray,
    threshold: float,
    keep_largest: bool,
    shape_postprocess: bool,
    skeleton_repair: bool,
    close_radius: int,
    hole_area: int,
    min_object_size: int,
):
    """Convert network output to A/B/C masks with shape-aware repair.

    The network predicts 4 exclusive classes: background, A-only, B-only, C-overlap.
    For final output, A and B are non-exclusive because both include the C region:
        A = A-only + C
        B = B-only + C
        C = overlap

    Segmentation often creates small holes inside A/B. This function fixes that by
    using the original image foreground + morphology + Zhang-Suen skeleton QC.
    """
    p_a = prob[1] + prob[3]
    p_b = prob[2] + prob[3]
    p_c = prob[3]

    raw_a = np.logical_or(p_a >= threshold, np.logical_or(label == 1, label == 3))
    raw_b = np.logical_or(p_b >= threshold, np.logical_or(label == 2, label == 3))
    raw_c = np.logical_or(p_c >= threshold, label == 3)

    # Foreground from original grayscale image keeps output on chromosome body,
    # so A/B shape is closer to the real object instead of holey probability blobs.
    fg = clean_mask(foreground_mask(gray), min_size=max(min_object_size, gray.shape[0] // 4))

    raw_a_info = analyze_skeleton(raw_a, already_skeleton=False)
    raw_b_info = analyze_skeleton(raw_b, already_skeleton=False)

    if shape_postprocess:
        mask_a, post_a_info = refine_single_chromosome_shape(
            raw_a,
            foreground=fg,
            min_size=min_object_size,
            close_radius=close_radius,
            hole_area=hole_area,
            keep_largest=keep_largest,
            skeleton_repair=skeleton_repair,
        )
        mask_b, post_b_info = refine_single_chromosome_shape(
            raw_b,
            foreground=fg,
            min_size=min_object_size,
            close_radius=close_radius,
            hole_area=hole_area,
            keep_largest=keep_largest,
            skeleton_repair=skeleton_repair,
        )
        mask_c = refine_binary_mask(
            raw_c,
            foreground=fg,
            close_radius=max(1, close_radius - 1),
            hole_area=max(32, hole_area // 4),
            min_size=max(8, min_object_size // 2),
            keep_largest=False,
            foreground_margin=1,
        )
    else:
        mask_a, mask_b, mask_c = raw_a, raw_b, raw_c
        post_a_info = analyze_skeleton(mask_a, already_skeleton=False)
        post_b_info = analyze_skeleton(mask_b, already_skeleton=False)

    # C must be the shared region between the two recognized chromosomes.
    shared = mask_a & mask_b
    if shared.sum() > 0:
        mask_c = (mask_c & shared) | shared
    else:
        mask_c = mask_c & (mask_a | mask_b)

    # Ensure final A/B include overlap C.
    mask_a = mask_a | mask_c
    mask_b = mask_b | mask_c

    label_out = np.zeros(label.shape, dtype=np.uint8)
    label_out[np.logical_and(mask_a, ~mask_b)] = 1
    label_out[np.logical_and(mask_b, ~mask_a)] = 2
    label_out[mask_c] = 3

    debug = {
        "raw_A_status": raw_a_info["status"],
        "raw_A_endpoints": raw_a_info["endpoints"],
        "raw_A_branch_points": raw_a_info["branch_points"],
        "raw_A_components": raw_a_info["components"],
        "raw_B_status": raw_b_info["status"],
        "raw_B_endpoints": raw_b_info["endpoints"],
        "raw_B_branch_points": raw_b_info["branch_points"],
        "raw_B_components": raw_b_info["components"],
        "post_A_status": post_a_info["status"],
        "post_A_endpoints": post_a_info["endpoints"],
        "post_A_branch_points": post_a_info["branch_points"],
        "post_A_components": post_a_info["components"],
        "post_B_status": post_b_info["status"],
        "post_B_endpoints": post_b_info["endpoints"],
        "post_B_branch_points": post_b_info["branch_points"],
        "post_B_components": post_b_info["components"],
    }
    return mask_a, mask_b, mask_c, label_out, raw_a, raw_b, raw_c, debug


def main():
    args = parse_args()
    device = torch.device(args.device)
    models_mod = _load_models_module()
    model_key = "student" if args.model == "student" else "teacher"
    model = models_mod.build_model(model_key, dropout=args.dropout).to(device)

    ckpt_name = "student_swin_resnet_fpn_v2_best.pth" if args.model == "student" else "teacher_cci_net_best.pth"
    ckpt_path = MODEL_DIR / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    _load_checkpoint(ckpt_path, model, device)
    model.eval()

    raw_dir = RESIZED_DIR / "overlap_raw"
    images = list_images(raw_dir)
    if not images:
        raise RuntimeError(f"No resized overlap_raw images found in {raw_dir}. Run 2v1_preprocess_zhang_suen.py first.")

    for sub in ["labels", "masks_A", "masks_B", "masks_C", "raw_masks", "skeleton_qc", "overlays"]:
        ensure_dir(OVERLAP_RESULT_DIR / sub)

    rows = []
    for path in tqdm(images, desc=f"Segment overlap_raw with {args.model}"):
        gray = read_gray(path, image_size=args.image_size)
        prob, raw_label, conf = _predict(model, gray, device)
        mask_a, mask_b, mask_c, label_out, raw_a, raw_b, raw_c, debug = _abc_from_prob(
            prob=prob,
            label=raw_label,
            gray=gray,
            threshold=args.prob_threshold,
            keep_largest=args.keep_largest,
            shape_postprocess=not args.no_shape_postprocess,
            skeleton_repair=not args.no_skeleton_repair,
            close_radius=args.close_radius,
            hole_area=args.hole_area,
            min_object_size=args.min_object_size,
        )

        stem = path.stem
        save_label(OVERLAP_RESULT_DIR / "labels" / f"{stem}_label_0_bg_1_A_2_B_3_C.png", label_out)
        save_gray(OVERLAP_RESULT_DIR / "masks_A" / f"{stem}_mask_A.png", mask_a.astype(np.uint8) * 255)
        save_gray(OVERLAP_RESULT_DIR / "masks_B" / f"{stem}_mask_B.png", mask_b.astype(np.uint8) * 255)
        save_gray(OVERLAP_RESULT_DIR / "masks_C" / f"{stem}_mask_C.png", mask_c.astype(np.uint8) * 255)
        save_gray(OVERLAP_RESULT_DIR / "raw_masks" / f"{stem}_raw_A_before_shape_repair.png", raw_a.astype(np.uint8) * 255)
        save_gray(OVERLAP_RESULT_DIR / "raw_masks" / f"{stem}_raw_B_before_shape_repair.png", raw_b.astype(np.uint8) * 255)
        save_gray(OVERLAP_RESULT_DIR / "raw_masks" / f"{stem}_raw_C_before_shape_repair.png", raw_c.astype(np.uint8) * 255)

        overlay = make_overlay(gray, mask_a, mask_b, mask_c)
        save_rgb(OVERLAP_RESULT_DIR / "overlays" / f"{stem}_overlay_A_B_C.png", overlay)

        skel_a = zhang_suen_skeleton(mask_a, pad=4)
        skel_b = zhang_suen_skeleton(mask_b, pad=4)
        skel_a_info = analyze_skeleton(skel_a, already_skeleton=True)
        skel_b_info = analyze_skeleton(skel_b, already_skeleton=True)
        save_gray(OVERLAP_RESULT_DIR / "skeleton_qc" / f"{stem}_A_skeleton.png", skel_a.astype(np.uint8) * 255)
        save_gray(OVERLAP_RESULT_DIR / "skeleton_qc" / f"{stem}_B_skeleton.png", skel_b.astype(np.uint8) * 255)

        rows.append({
            "filename": path.name,
            "model": args.model,
            "confidence_mean": conf,
            "shape_postprocess": not args.no_shape_postprocess,
            "skeleton_repair": not args.no_skeleton_repair,
            "A_pixels": int(mask_a.sum()),
            "B_pixels": int(mask_b.sum()),
            "C_pixels": int(mask_c.sum()),
            "A_skeleton_status": skel_a_info["status"],
            "A_endpoints": skel_a_info["endpoints"],
            "A_branch_points": skel_a_info["branch_points"],
            "A_components": skel_a_info["components"],
            "B_skeleton_status": skel_b_info["status"],
            "B_endpoints": skel_b_info["endpoints"],
            "B_branch_points": skel_b_info["branch_points"],
            "B_components": skel_b_info["components"],
            **debug,
        })

    csv_path = OVERLAP_RESULT_DIR / "overlap_raw_ABC_predictions_student.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("[OK] A/B/C segmentation outputs saved to:", OVERLAP_RESULT_DIR)
    print("[OK] Summary CSV:", csv_path)
    print("[OK] Shape repair is", "OFF" if args.no_shape_postprocess else "ON")


if __name__ == "__main__":
    main()
