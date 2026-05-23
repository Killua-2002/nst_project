import argparse
import csv
import importlib.util
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    CLASS_NAMES,
    DATASET_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_EPOCHS,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_LR,
    DEFAULT_PATIENCE,
    DEFAULT_PSEUDO_THRESHOLD,
    MODEL_DIR,
    RESIZED_DIR,
    RESULT_DIR,
)
from utils_image import ensure_dir, list_images, save_label
from utils_train import (
    CombinedSegmentationDataset,
    SegmentationDataset,
    load_checkpoint,
    metrics_from_cm,
    pixel_confusion_matrix,
    plot_confusion_matrix,
    save_checkpoint,
    write_history_csv,
    save_json,
)


def _load_models_module():
    spec = importlib.util.spec_from_file_location("models_v1", Path(__file__).resolve().parent / "4v1_models.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args():
    p = argparse.ArgumentParser(description="Semi-supervised Teacher-Student training for A/B/C segmentation.")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    p.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    p.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    p.add_argument("--pseudo-threshold", type=float, default=DEFAULT_PSEUDO_THRESHOLD)
    p.add_argument("--min-pseudo", type=int, default=1, help="If no raw image passes threshold, still keep top-N pseudo labels.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _load_gray_tensor(path: Path, image_size: int, device):
    img = Image.open(path).convert("L").resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr)[None, None, :, :].to(device)
    return x


def create_pseudo_labels(args, teacher):
    raw_dir = RESIZED_DIR / "overlap_raw"
    pseudo_img_dir = DATASET_DIR / "pseudo/images"
    pseudo_lab_dir = DATASET_DIR / "pseudo/labels"
    if pseudo_img_dir.exists():
        shutil.rmtree(pseudo_img_dir)
    if pseudo_lab_dir.exists():
        shutil.rmtree(pseudo_lab_dir)
    ensure_dir(pseudo_img_dir)
    ensure_dir(pseudo_lab_dir)

    candidates = []
    teacher.eval()
    raw_images = list_images(raw_dir)
    if not raw_images:
        print(f"[WARN] No overlap_raw images found in {raw_dir}. Semi step will have no pseudo data.")
        return []

    with torch.no_grad():
        for path in tqdm(raw_images, desc="Teacher pseudo-label overlap_raw"):
            x = _load_gray_tensor(path, args.image_size, next(teacher.parameters()).device)
            logits = teacher(x)
            prob = torch.softmax(logits, dim=1)
            max_prob, pred = prob.max(dim=1)
            conf = float(max_prob.mean().item())
            label = pred.squeeze(0).cpu().numpy().astype(np.uint8)
            candidates.append((conf, path, label))

    accepted = [(c, p, lab) for c, p, lab in candidates if c >= args.pseudo_threshold]
    if len(accepted) < args.min_pseudo and candidates:
        # Keep top candidates so the pipeline is still semi-supervised.
        candidates_sorted = sorted(candidates, key=lambda x: x[0], reverse=True)
        accepted = candidates_sorted[: min(args.min_pseudo, len(candidates_sorted))]
        print(f"[WARN] Low teacher confidence. Keeping top {len(accepted)} pseudo labels for semi-supervised training.")

    rows = []
    for conf, path, label in accepted:
        out_img = pseudo_img_dir / path.name
        out_lab = pseudo_lab_dir / f"{path.stem}.png"
        shutil.copy2(path, out_img)
        save_label(out_lab, label)
        rows.append({
            "filename": path.name,
            "confidence": conf,
            "label_path": str(out_lab),
            "status": "accepted" if conf >= args.pseudo_threshold else "accepted_low_confidence",
        })

    with open(RESULT_DIR / "pseudo_labels.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "confidence", "label_path", "status"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Pseudo labels accepted: {len(rows)}/{len(raw_images)}")
    return rows


def evaluate(model, loader, device):
    model.eval()
    cm = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.long, device="cpu")
    loss_sum = 0.0
    n = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            pred = logits.argmax(dim=1)
            cm += pixel_confusion_matrix(pred.cpu(), y.cpu())
            loss_sum += float(loss.item()) * x.size(0)
            n += x.size(0)
    metrics = metrics_from_cm(cm)
    metrics["loss"] = loss_sum / max(1, n)
    return metrics, cm


def main():
    args = parse_args()
    device = torch.device(args.device)
    models_mod = _load_models_module()

    teacher = models_mod.build_model("teacher", dropout=args.dropout).to(device)
    teacher_path = MODEL_DIR / "teacher_cci_net_best.pth"
    if not teacher_path.exists():
        raise FileNotFoundError(f"Missing teacher checkpoint: {teacher_path}. Run 5v1_train_teacher.py first.")
    load_checkpoint(teacher_path, teacher, map_location=device)

    pseudo_rows = create_pseudo_labels(args, teacher)

    student = models_mod.build_model("student", dropout=args.dropout).to(device)

    train_ds = SegmentationDataset(DATASET_DIR / "train/images", DATASET_DIR / "train/labels", augment=True, image_size=args.image_size)
    datasets = [train_ds]
    if pseudo_rows:
        pseudo_ds = SegmentationDataset(DATASET_DIR / "pseudo/images", DATASET_DIR / "pseudo/labels", augment=True, image_size=args.image_size)
        datasets.append(pseudo_ds)
    combined_train = CombinedSegmentationDataset(datasets)

    val_ds = SegmentationDataset(DATASET_DIR / "val/images", DATASET_DIR / "val/labels", augment=False, image_size=args.image_size)

    train_loader = DataLoader(combined_train, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    best_val = 10**9
    wait = 0
    history = []
    best_path = MODEL_DIR / "student_swin_resnet_fpn_v2_best.pth"

    for epoch in range(1, args.epochs + 1):
        student.train()
        total_loss = 0.0
        total_n = 0
        pbar = tqdm(train_loader, desc=f"Student SSL epoch {epoch}/{args.epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = student(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * x.size(0)
            total_n += x.size(0)
            pbar.set_postfix(loss=total_loss / max(1, total_n))

        train_loss = total_loss / max(1, total_n)
        val_metrics, val_cm = evaluate(student, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_pixel_acc": val_metrics["pixel_acc"],
            "val_mean_iou": val_metrics["mean_iou"],
            "val_mean_f1": val_metrics["mean_f1"],
            "pseudo_count": len(pseudo_rows),
        }
        history.append(row)
        print(row)

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            wait = 0
            save_checkpoint(best_path, student, {"epoch": epoch, "image_size": args.image_size, "model": "StudentSwinResNetFPNV2"})
        else:
            wait += 1
            if wait >= args.patience:
                print(f"[EARLY STOP] Student stopped at epoch {epoch}")
                break

    write_history_csv(RESULT_DIR / "student_ssl_training_log.csv", history)

    load_checkpoint(best_path, student, map_location=device)
    val_metrics, val_cm = evaluate(student, val_loader, device)
    plot_confusion_matrix(RESULT_DIR / "figures/student_val_confusion_matrix.png", val_cm, CLASS_NAMES)
    save_json(RESULT_DIR / "student_val_metrics.json", val_metrics)
    print("[OK] Student semi-supervised training done:", best_path)


if __name__ == "__main__":
    main()
