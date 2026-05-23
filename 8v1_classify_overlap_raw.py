import argparse

import pandas as pd

import config
from src.models import build_model
from src.train_utils import classify_folder, load_checkpoint
from src.utils import get_device, list_images


def choose_overlap_dir(skip_resized: bool = False):
    if not skip_resized and len(list_images(config.RESIZED_OVERLAP_RAW_DIR)) > 0:
        return config.RESIZED_OVERLAP_RAW_DIR
    return config.OVERLAP_RAW_DIR


def main():
    parser = argparse.ArgumentParser(description="Classify real overlap_raw images.")
    parser.add_argument("--model", choices=["teacher", "student"], default="student")
    parser.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dropout", type=float, default=config.DEFAULT_DROPOUT)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--image-size", type=int, default=config.IMAGE_SIZE)
    parser.add_argument("--skip-resized", action="store_true", help="Use source_data/overlap_raw instead of generated_data/resized.")
    args = parser.parse_args()

    device = get_device(args.device)
    overlap_dir = choose_overlap_dir(args.skip_resized)
    model_key = "teacher" if args.model == "teacher" else "student"
    model = build_model(model_key, num_classes=config.NUM_CLASSES, dropout=args.dropout, pretrained=args.pretrained)
    ckpt_name = "teacher_ccinet_best.pt" if args.model == "teacher" else "student_swin_resnet50fpnv2_best.pt"
    load_checkpoint(model, config.CHECKPOINT_DIR / ckpt_name, device)

    out_csv = config.RESULT_DIR / f"overlap_raw_predictions_{args.model}.csv"
    out_dir = config.RESULT_DIR / f"classified_overlap_raw_{args.model}"
    df = classify_folder(
        model=model,
        folder=overlap_dir,
        output_csv=out_csv,
        device=device,
        class_names=config.CLASSES,
        batch_size=args.batch_size,
        output_folder=out_dir,
        image_size=args.image_size,
    )

    stats_csv = config.GENERATED_DIR / "overlap_raw" / "preprocess_stats.csv"
    if stats_csv.exists() and len(df) > 0:
        stats = pd.read_csv(stats_csv)
        stats["file_name"] = stats["resized_path"].fillna(stats["file"]).apply(lambda x: str(x).split("/")[-1].split("\\")[-1])
        df["file_name"] = df["file"].apply(lambda x: str(x).split("/")[-1].split("\\")[-1])
        merged = df.merge(stats.drop(columns=["file"], errors="ignore"), on="file_name", how="left")
        merged.to_csv(out_csv, index=False)
    print("Using overlap folder:", overlap_dir)
    print("Saved:", out_csv)
    print("Copied classified images to:", out_dir)


if __name__ == "__main__":
    main()
