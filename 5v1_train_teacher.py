import argparse

import config
from src.datasets import make_dataloaders
from src.models import build_model
from src.train_utils import train_model
from src.utils import get_device, set_seed


def main():
    parser = argparse.ArgumentParser(description="Train CCI-Net style Teacher on synthetic labeled dataset.")
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.DEFAULT_LR)
    parser.add_argument("--patience", type=int, default=config.DEFAULT_PATIENCE)
    parser.add_argument("--dropout", type=float, default=config.DEFAULT_DROPOUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=config.IMAGE_SIZE)
    args = parser.parse_args()

    set_seed(config.RANDOM_SEED)
    config.ensure_project_dirs()
    device = get_device(args.device)
    loaders = make_dataloaders(config.DATASET_DIR, batch_size=args.batch_size, num_workers=args.num_workers, image_size=args.image_size)
    model = build_model("teacher", num_classes=len(loaders["class_names"]), dropout=args.dropout, pretrained=False)
    ckpt = train_model(model, loaders, device, "teacher_ccinet", epochs=args.epochs, lr=args.lr, patience=args.patience)
    print("Teacher checkpoint saved:", ckpt)


if __name__ == "__main__":
    main()
