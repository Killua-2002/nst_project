
import argparse
import csv
import importlib.util
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from config import DEFAULT_DROPOUT, DEFAULT_IMAGE_SIZE, MODEL_DIR, OVERLAP_RESULT_DIR, RESIZED_DIR
from shape_classifier_model import load_shape_classifier
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
from utils_shape_guided import choose_best_shape_guided_pair


def _load_models_module():
    spec = importlib.util.spec_from_file_location("models_v1", Path(__file__).resolve().parent / "4v1_models.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args():
    p = argparse.ArgumentParser(
        description="Hybrid Classif + Segmentation: segment overlap_raw into NST A, NST B and C-overlap."
    )
    p.add_argument("--model", choices=["student", "teacher"], default="student")
    p.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    p.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    p.add_argument("--prob-threshold", type=float, default=0.45)
    p.add_argument("--keep-largest", action="store_true", help="Keep largest connected component of A and B masks.")
    p.add_argument("--no-shape-postprocess", action="store_true", help="Disable morphology hole filling / smoothing.")
    p.add_argument("--no-skeleton-repair", action="store_true", help="Disable Zhang-Suen-guided candidate selection.")
    p.add_argument("--no-shape-classifier", action="store_true", help="Disable classifier-guided shape selection.")
    p.add_argument("--close-radius", type=int, default=2, help="Morphological closing radius for A/B shape repair.")
    p.add_argument("--hole-area", type=int, default=768, help="Fill holes up to this area inside A/B masks.")
    p.add_argument("--min-object-size", type=int, default=35, help="Remove connected components smaller than this.")
    p.add_argument("--visualize-limit", type=int, default=80, help="Save multi-panel raw visualizations for first N images. Use -1 for all.")
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


def _raw_masks_from_prob(prob: np.ndarray, label: np.ndarray, threshold: float):
    p_a = prob[1] + prob[3]
    p_b = prob[2] + prob[3]
    p_c = prob[3]
    raw_a = np.logical_or(p_a >= threshold, np.logical_or(label == 1, label == 3))
    raw_b = np.logical_or(p_b >= threshold, np.logical_or(label == 2, label == 3))
    raw_c = np.logical_or(p_c >= threshold, label == 3)
    return raw_a.astype(bool), raw_b.astype(bool), raw_c.astype(bool)


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
    shape_classifier=None,
    device=None,
):
    """Convert network output to final A/B/C with classifier + segment hybrid.

    Core idea:
    1. Student/Teacher segmentation gives pixel probabilities for A/B/C.
    2. Raw grayscale foreground gives actual NST body.
    3. Shape classifier checks whether A and B look like real single chromosomes.
    4. Zhang-Suen skeleton QC rejects holey/non-chromosome-like candidates.
    5. Best candidate is saved as final A/B/C.

    This avoids the old failure mode where mask_A/mask_B became spotty blobs that
    did not preserve NST morphology.
    """
    raw_a, raw_b, raw_c = _raw_masks_from_prob(prob, label, threshold)

    fg = clean_mask(foreground_mask(gray), min_size=max(min_object_size, gray.shape[0] // 14))

    raw_a_info = analyze_skeleton(raw_a, already_skeleton=False)
    raw_b_info = analyze_skeleton(raw_b, already_skeleton=False)

    if shape_postprocess:
        base_a, post_a_info = refine_single_chromosome_shape(
            raw_a,
            foreground=fg,
            min_size=min_object_size,
            close_radius=close_radius,
            hole_area=hole_area,
            keep_largest=keep_largest,
            skeleton_repair=skeleton_repair,
        )
        base_b, post_b_info = refine_single_chromosome_shape(
            raw_b,
            foreground=fg,
            min_size=min_object_size,
            close_radius=close_radius,
            hole_area=hole_area,
            keep_largest=keep_largest,
            skeleton_repair=skeleton_repair,
        )
        base_c = refine_binary_mask(
            raw_c,
            foreground=fg,
            close_radius=max(1, close_radius - 1),
            hole_area=max(32, hole_area // 4),
            min_size=max(8, min_object_size // 2),
            keep_largest=False,
            foreground_margin=1,
        )

        # Hybrid classification + segmentation candidate selection.
        if shape_classifier is not None or skeleton_repair:
            best = choose_best_shape_guided_pair(
                gray=gray,
                prob=prob,
                label=label,
                raw_a=raw_a,
                raw_b=raw_b,
                raw_c=raw_c,
                base_a=base_a,
                base_b=base_b,
                base_c=base_c,
                close_radius=close_radius,
                hole_area=hole_area,
                min_object_size=min_object_size,
                keep_largest=keep_largest,
                shape_classifier=shape_classifier,
                device=device,
            )
            mask_a, mask_b, mask_c = best.a, best.b, best.c
            chosen = best.name
            chosen_score = best.score
            chosen_detail = best.detail
        else:
            mask_a, mask_b, mask_c = base_a, base_b, base_c
            chosen = "segment_morphology_only"
            chosen_score = 0.0
            chosen_detail = {}
    else:
        mask_a, mask_b, mask_c = raw_a, raw_b, raw_c
        post_a_info = analyze_skeleton(mask_a, already_skeleton=False)
        post_b_info = analyze_skeleton(mask_b, already_skeleton=False)
        chosen = "raw_segment_only"
        chosen_score = 0.0
        chosen_detail = {}

    # C is the shared overlapping region between the two recognized chromosomes.
    mask_c = mask_c.astype(bool) & mask_a.astype(bool) & mask_b.astype(bool)
    if mask_c.sum() == 0:
        # Use a small contact band if classifier/watershed made A and B touch but C vanished.
        from skimage.morphology import binary_dilation, disk
        contact = binary_dilation(mask_a, disk(2)) & binary_dilation(mask_b, disk(2)) & fg
        mask_c = contact.astype(bool)

    # Ensure final A/B include overlap C.
    mask_a = mask_a.astype(bool) | mask_c
    mask_b = mask_b.astype(bool) | mask_c

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
        "chosen_refine_method": chosen,
        "chosen_refine_score": chosen_score,
        **{f"chosen_{k}": v for k, v in chosen_detail.items()},
    }
    return mask_a, mask_b, mask_c, label_out, raw_a, raw_b, raw_c, debug


def _make_visual_check(gray, raw_overlay, final_overlay, mask_a, mask_b, mask_c, skel_a, skel_b, out_path):
    """Save multi-panel visualization directly on the raw/resized NST image."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    axes = axes.ravel()

    panels = [
        ("Raw grayscale NST", gray, "gray"),
        ("Raw model overlay", raw_overlay, None),
        ("Final A/B/C overlay", final_overlay, None),
        ("Mask C overlap", mask_c.astype(np.uint8) * 255, "gray"),
        ("Final mask A", mask_a.astype(np.uint8) * 255, "gray"),
        ("Final mask B", mask_b.astype(np.uint8) * 255, "gray"),
        ("Skeleton A", skel_a.astype(np.uint8) * 255, "gray"),
        ("Skeleton B", skel_b.astype(np.uint8) * 255, "gray"),
    ]

    for ax, (title, img, cmap) in zip(axes, panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.tight_layout()
    ensure_dir(Path(out_path).parent)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


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

    shape_classifier = None
    shape_ckpt = MODEL_DIR / "shape_classifier_valid_nst_best.pth"
    if not args.no_shape_classifier:
        shape_classifier = load_shape_classifier(shape_ckpt, device=device, dropout=args.dropout)
        if shape_classifier is None:
            print(f"[WARN] Shape-classifier checkpoint not found: {shape_ckpt}")
            print("[WARN] Continuing with segmentation + skeleton scoring only.")
        else:
            print("[OK] Loaded shape classifier:", shape_ckpt)

    raw_dir = RESIZED_DIR / "overlap_raw"
    images = list_images(raw_dir)
    if not images:
        raise RuntimeError(f"No resized overlap_raw images found in {raw_dir}. Run 2v1_preprocess_zhang_suen.py first.")

    for sub in ["labels", "masks_A", "masks_B", "masks_C", "raw_masks", "skeleton_qc", "overlays", "visualizations"]:
        ensure_dir(OVERLAP_RESULT_DIR / sub)

    rows = []
    for idx, path in enumerate(tqdm(images, desc=f"Hybrid Classif+Seg overlap_raw with {args.model}")):
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
            shape_classifier=None if args.no_shape_classifier else shape_classifier,
            device=device,
        )

        stem = path.stem
        save_label(OVERLAP_RESULT_DIR / "labels" / f"{stem}_label_0_bg_1_A_2_B_3_C.png", label_out)
        save_gray(OVERLAP_RESULT_DIR / "masks_A" / f"{stem}_mask_A.png", mask_a.astype(np.uint8) * 255)
        save_gray(OVERLAP_RESULT_DIR / "masks_B" / f"{stem}_mask_B.png", mask_b.astype(np.uint8) * 255)
        save_gray(OVERLAP_RESULT_DIR / "masks_C" / f"{stem}_mask_C.png", mask_c.astype(np.uint8) * 255)
        save_gray(OVERLAP_RESULT_DIR / "raw_masks" / f"{stem}_raw_A_before_shape_repair.png", raw_a.astype(np.uint8) * 255)
        save_gray(OVERLAP_RESULT_DIR / "raw_masks" / f"{stem}_raw_B_before_shape_repair.png", raw_b.astype(np.uint8) * 255)
        save_gray(OVERLAP_RESULT_DIR / "raw_masks" / f"{stem}_raw_C_before_shape_repair.png", raw_c.astype(np.uint8) * 255)

        raw_overlay = make_overlay(gray, raw_a, raw_b, raw_c)
        final_overlay = make_overlay(gray, mask_a, mask_b, mask_c)
        save_rgb(OVERLAP_RESULT_DIR / "overlays" / f"{stem}_overlay_A_B_C.png", final_overlay)
        save_rgb(OVERLAP_RESULT_DIR / "overlays" / f"{stem}_raw_model_overlay_before_repair.png", raw_overlay)

        skel_a = zhang_suen_skeleton(mask_a, pad=4)
        skel_b = zhang_suen_skeleton(mask_b, pad=4)
        skel_a_info = analyze_skeleton(skel_a, already_skeleton=True)
        skel_b_info = analyze_skeleton(skel_b, already_skeleton=True)
        save_gray(OVERLAP_RESULT_DIR / "skeleton_qc" / f"{stem}_A_skeleton.png", skel_a.astype(np.uint8) * 255)
        save_gray(OVERLAP_RESULT_DIR / "skeleton_qc" / f"{stem}_B_skeleton.png", skel_b.astype(np.uint8) * 255)

        if args.visualize_limit < 0 or idx < args.visualize_limit:
            _make_visual_check(
                gray=gray,
                raw_overlay=raw_overlay,
                final_overlay=final_overlay,
                mask_a=mask_a,
                mask_b=mask_b,
                mask_c=mask_c,
                skel_a=skel_a,
                skel_b=skel_b,
                out_path=OVERLAP_RESULT_DIR / "visualizations" / f"{stem}_visual_check_raw_A_B_C.png",
            )

        rows.append({
            "filename": path.name,
            "model": args.model,
            "confidence_mean": conf,
            "hybrid_classif_seg": not args.no_shape_classifier,
            "shape_classifier_loaded": shape_classifier is not None,
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

    print("[OK] Hybrid A/B/C outputs saved to:", OVERLAP_RESULT_DIR)
    print("[OK] Summary CSV:", csv_path)
    print("[OK] Visual checks:", OVERLAP_RESULT_DIR / "visualizations")
    print("[OK] Shape classifier:", "OFF" if args.no_shape_classifier else ("LOADED" if shape_classifier is not None else "MISSING_FALLBACK_TO_SKELETON"))


if __name__ == "__main__":
    main()
