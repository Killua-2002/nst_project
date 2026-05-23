import argparse
import importlib.util
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import CLASS_NAMES, DATASET_DIR, DEFAULT_BATCH_SIZE, DEFAULT_DROPOUT, DEFAULT_IMAGE_SIZE, MODEL_DIR, RESULT_DIR
from utils_train import SegmentationDataset, load_checkpoint, metrics_from_cm, pixel_confusion_matrix, plot_confusion_matrix, save_json


def _load_models_module():
    spec = importlib.util.spec_from_file_location("models_v1", Path(__file__).resolve().parent / "4v1_models.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Teacher and Student on synthetic test labels.")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    p.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def evaluate(model, loader, device):
    model.eval()
    cm = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.long)
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

    test_ds = SegmentationDataset(DATASET_DIR / "test/images", DATASET_DIR / "test/labels", augment=False, image_size=args.image_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    for name, model_key, ckpt_name in [
        ("teacher", "teacher", "teacher_cci_net_best.pth"),
        ("student", "student", "student_swin_resnet_fpn_v2_best.pth"),
    ]:
        path = MODEL_DIR / ckpt_name
        if not path.exists():
            print(f"[WARN] Missing checkpoint for {name}: {path}")
            continue
        model = models_mod.build_model(model_key, dropout=args.dropout).to(device)
        load_checkpoint(path, model, map_location=device)
        metrics, cm = evaluate(model, test_loader, device)
        plot_confusion_matrix(RESULT_DIR / f"figures/{name}_test_confusion_matrix.png", cm, CLASS_NAMES)
        save_json(RESULT_DIR / f"{name}_test_metrics.json", metrics)
        print(f"[{name.upper()}]", metrics)

    print("[OK] Evaluation done.")


if __name__ == "__main__":
    main()
