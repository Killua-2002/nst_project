from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from config import CFG
from models import build_model
from train_utils import load_checkpoint
from utils_image import ensure_dir, label_to_masks, list_images, overlay_abc, save_label_and_masks


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Segment overlap_raw into NST A, NST B, and overlap C masks.")
    parser.add_argument("--model", choices=["student", "teacher"], default="student")
    parser.add_argument("--dropout", type=float, default=CFG.DROPOUT)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = CFG()
    if args.model == "student":
        ckpt = cfg.RESULT_DIR / "models" / "student_swin_resnet_fpn_best.pth"
        out_model_name = "student"
    else:
        ckpt = cfg.RESULT_DIR / "models" / "teacher_cci_best.pth"
        out_model_name = "teacher"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

    model = build_model(args.model, num_classes=cfg.NUM_CLASSES, dropout=args.dropout)
    load_checkpoint(model, ckpt, args.device)
    model.to(args.device).eval()

    raw_dir = cfg.RESIZED_DIR / "overlap_raw"
    paths = list_images(raw_dir)
    if not paths:
        raise FileNotFoundError(f"No resized overlap_raw images found in {raw_dir}. Run preprocess first.")

    out_root = cfg.RESULT_DIR / "overlap_raw"
    dirs = {
        "labels": out_root / "labels",
        "masks_A": out_root / "masks_A",
        "masks_B": out_root / "masks_B",
        "masks_C": out_root / "masks_C",
        "overlays": out_root / "overlays",
    }
    for d in dirs.values():
        ensure_dir(d)

    rows = []
    for p in tqdm(paths, desc=f"segment overlap_raw using {out_model_name}"):
        gray = np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0
        x = torch.from_numpy(1.0 - gray).unsqueeze(0).unsqueeze(0).to(args.device)
        logits = model(x)
        prob = F.softmax(logits, dim=1)[0]
        conf, label = prob.max(dim=0)
        label_np = label.detach().cpu().numpy().astype(np.uint8)
        conf_np = conf.detach().cpu().numpy()

        name = p.name
        save_label_and_masks(
            label_np,
            dirs["labels"] / name,
            dirs["masks_A"] / name,
            dirs["masks_B"] / name,
            dirs["masks_C"] / name,
        )
        gray_uint8 = np.array(Image.open(p).convert("L"), dtype=np.uint8)
        overlay = overlay_abc(gray_uint8, label_np)
        Image.fromarray(overlay).save(dirs["overlays"] / name)
        a, b, c = label_to_masks(label_np)
        rows.append({
            "filename": name,
            "model": out_model_name,
            "mean_confidence": float(conf_np.mean()),
            "pixels_A": int((a > 0).sum()),
            "pixels_B": int((b > 0).sum()),
            "pixels_C_overlap": int((c > 0).sum()),
            "label_path": str(dirs["labels"] / name),
            "mask_A_path": str(dirs["masks_A"] / name),
            "mask_B_path": str(dirs["masks_B"] / name),
            "mask_C_path": str(dirs["masks_C"] / name),
            "overlay_path": str(dirs["overlays"] / name),
        })

    df = pd.DataFrame(rows)
    csv_path = out_root / f"overlap_raw_ABC_predictions_{out_model_name}.csv"
    df.to_csv(csv_path, index=False)
    print("Saved output CSV:", csv_path)
    print("Masks saved under:", out_root)
    print("Class meaning: A mask = label 1 or 3; B mask = label 2 or 3; C mask = label 3 overlap region.")


if __name__ == "__main__":
    main()
