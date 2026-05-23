from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

import config
from src.image_processing import list_images, read_grayscale, resize_and_pad_gray, to_binary, zhang_suen_skeleton, classify_by_skeleton, save_debug_images
from src.train_utils import ensure_device

models_mod = importlib.import_module("4v1_models")
get_model = models_mod.get_model
load_checkpoint = models_mod.load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Classify raw overlapping chromosome images and export result CSV")
    parser.add_argument("--ckpt", type=str, default=str(config.CHECKPOINT_DIR / "student_swin_resnet50_fpn_v2_best.pt"))
    parser.add_argument("--model", type=str, default=config.STUDENT_MODEL)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--input-dir", type=str, default=str(config.OVERLAP_RAW_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = ensure_device(args.device)
    out_dir = config.RESULT_DIR / "classified_overlap_raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = get_model(args.model, num_classes=config.NUM_CLASSES, dropout=config.DROPOUT).to(device)
    load_checkpoint(model, args.ckpt, map_location=device)
    model.eval()

    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    rows = []
    for path in list_images(Path(args.input_dir)):
        try:
            gray = resize_and_pad_gray(read_grayscale(path), config.IMG_SIZE)
            binary = to_binary(gray, min_object_area=config.MIN_OBJECT_AREA)
            skeleton = zhang_suen_skeleton(binary, pad=config.SKELETON_PAD)
            rule = classify_by_skeleton(binary, skeleton, min_skeleton_pixels=config.MIN_SKELETON_PIXELS)
            save_debug_images(gray, binary, skeleton, out_dir / path.stem)

            img = Image.fromarray(gray).convert("L")
            x = tfm(img).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(x), dim=1).squeeze(0).cpu().numpy()
            pred_idx = int(probs.argmax())
            pred_label = config.CLASS_NAMES[pred_idx]
            rows.append({
                "filename": path.name,
                "path": str(path),
                "model_label": pred_label,
                "model_confidence": float(probs[pred_idx]),
                "skeleton_rule_label": rule["predicted_rule_label"],
                "is_two_single_paths_by_rule": rule["is_two_single_paths"],
                "endpoints": rule["total_endpoints"],
                "branchpoints": rule["total_branchpoints"],
                "skeleton_components": rule["skeleton_components"],
            })
        except Exception as exc:
            rows.append({"filename": path.name, "path": str(path), "error": repr(exc)})

    out_csv = config.RESULT_DIR / "classified_overlap_raw.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("Saved classification result:", out_csv)
    print("Debug grayscale/binary/skeleton/overlay images:", out_dir)


if __name__ == "__main__":
    main()
