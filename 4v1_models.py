import argparse

import torch

import config
from src.models import build_model, count_parameters


def main():
    parser = argparse.ArgumentParser(description="Print model information.")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    for name in ["teacher", "student"]:
        model = build_model(name, num_classes=config.NUM_CLASSES, dropout=config.DEFAULT_DROPOUT, pretrained=args.pretrained)
        print("=" * 80)
        print(name)
        print(model.__class__.__name__)
        print("trainable parameters:", f"{count_parameters(model):,}")
        dummy = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
        with torch.no_grad():
            out = model(dummy)
        print("output shape:", tuple(out.shape))


if __name__ == "__main__":
    main()
