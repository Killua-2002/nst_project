from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CFG
from models import build_model
from train_utils import (
    CombinedSyntheticPseudoDataset,
    SegmentationDataset,
    evaluate,
    fit_model,
    load_checkpoint,
    plot_confusion_matrix,
)
from utils_image import ensure_dir, label_to_masks, list_images, save_label_and_masks, seed_everything


def reset_pseudo_dirs(cfg: CFG) -> None:
    root = cfg.DATASET_DIR / "pseudo"
    for sub in ["images", "labels", "masks_A", "masks_B", "masks_C"]:
        d = root / sub
        if d.exists():
            shutil.rmtree(d)
        ensure_dir(d)


@torch.no_grad()
def create_pseudo_labels(cfg: CFG, teacher_ckpt: Path, device: str, threshold: float, dropout: float) -> pd.DataFrame:
    reset_pseudo_dirs(cfg)
    model = build_model("teacher", num_classes=cfg.NUM_CLASSES, dropout=dropout)
    load_checkpoint(model, teacher_ckpt, device)
    model.to(device).eval()

    raw_dir = cfg.RESIZED_DIR / "overlap_raw"
    paths = list_images(raw_dir)
    if not paths:
        print("Warning: no overlap_raw images for pseudo-labeling. Student will train only on synthetic data.")
        return pd.DataFrame()

    rows = []
    for p in tqdm(paths, desc="teacher pseudo-label overlap_raw"):
        gray = np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0
        x = torch.from_numpy(1.0 - gray).unsqueeze(0).unsqueeze(0).to(device)
        logits = model(x)
        prob = F.softmax(logits, dim=1)[0]
        conf, label = prob.max(dim=0)
        label_np = label.detach().cpu().numpy().astype(np.uint8)
        conf_np = conf.detach().cpu().numpy()

        # Ignore low confidence pixels during SSL student training.
        pseudo_train = label_np.copy()
        pseudo_train[conf_np < threshold] = 255

        out_name = p.name
        shutil.copy2(p, cfg.DATASET_DIR / "pseudo" / "images" / out_name)
        save_label_and_masks(
            pseudo_train,
            cfg.DATASET_DIR / "pseudo" / "labels" / out_name,
            cfg.DATASET_DIR / "pseudo" / "masks_A" / out_name,
            cfg.DATASET_DIR / "pseudo" / "masks_B" / out_name,
            cfg.DATASET_DIR / "pseudo" / "masks_C" / out_name,
        )
        rows.append({
            "filename": out_name,
            "mean_confidence": float(conf_np.mean()),
            "valid_pixel_ratio": float((conf_np >= threshold).mean()),
            "threshold": threshold,
            "pseudo_label": str(cfg.DATASET_DIR / "pseudo" / "labels" / out_name),
        })

    df = pd.DataFrame(rows)
    csv_path = cfg.RESULT_DIR / "pseudo_labels_manifest.csv"
    df.to_csv(csv_path, index=False)
    print("Saved pseudo manifest:", csv_path)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Student Swin+ResNet-FPN with synthetic labels + Teacher pseudo labels.")
    parser.add_argument("--epochs", type=int, default=CFG.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=CFG.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=CFG.LR)
    parser.add_argument("--dropout", type=float, default=CFG.DROPOUT)
    parser.add_argument("--patience", type=int, default=CFG.EARLY_STOPPING)
    parser.add_argument("--pseudo-threshold", type=float, default=CFG.PSEUDO_THRESHOLD)
    parser.add_argument("--pseudo-loss-weight", type=float, default=CFG.PSEUDO_LOSS_WEIGHT)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=CFG.SEED)
    args = parser.parse_args()

    seed_everything(args.seed)
    cfg = CFG()
    teacher_ckpt = cfg.RESULT_DIR / "models" / "teacher_cci_best.pth"
    if not teacher_ckpt.exists():
        raise FileNotFoundError(f"Missing teacher checkpoint: {teacher_ckpt}. Run 5v1_train_teacher.py first.")

    pseudo_df = create_pseudo_labels(cfg, teacher_ckpt, args.device, args.pseudo_threshold, args.dropout)

    syn_train = SegmentationDataset(cfg.DATASET_DIR, "train", augment=True)
    pseudo_ds = None
    try:
        if len(pseudo_df) > 0:
            pseudo_ds = SegmentationDataset(cfg.DATASET_DIR, "pseudo", augment=True)
    except FileNotFoundError:
        pseudo_ds = None

    train_ds = CombinedSyntheticPseudoDataset(syn_train, pseudo_ds)
    val_ds = SegmentationDataset(cfg.DATASET_DIR, "val", augment=False)
    test_ds = SegmentationDataset(cfg.DATASET_DIR, "test", augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = build_model("student", num_classes=cfg.NUM_CLASSES, dropout=args.dropout)
    ckpt = cfg.RESULT_DIR / "models" / "student_swin_resnet_fpn_best.pth"
    log_csv = cfg.RESULT_DIR / "student_ssl_training_log.csv"
    fit_model(
        model,
        train_loader,
        val_loader,
        args.device,
        args.epochs,
        args.lr,
        args.patience,
        ckpt,
        log_csv,
        cfg.CLASS_NAMES,
        pseudo_weight=args.pseudo_loss_weight,
    )

    model = build_model("student", num_classes=cfg.NUM_CLASSES, dropout=args.dropout)
    load_checkpoint(model, ckpt, args.device)
    model.to(args.device)
    test_metrics = evaluate(model, test_loader, args.device, cfg.CLASS_NAMES)
    print("Student test metrics:", {k: v for k, v in test_metrics.items() if k != "cm"})
    plot_confusion_matrix(
        test_metrics["cm"],
        cfg.CLASS_NAMES,
        cfg.RESULT_DIR / "plots" / "student_pixel_confusion_matrix.png",
        "Student Swin+ResNet-FPN pixel confusion matrix",
    )


if __name__ == "__main__":
    main()
