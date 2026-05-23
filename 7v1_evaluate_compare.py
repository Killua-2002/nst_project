import argparse

import config
from src.datasets import make_dataloaders
from src.models import build_model
from src.train_utils import evaluate_model, load_checkpoint
from src.utils import get_device


def main():
    parser = argparse.ArgumentParser(description="Evaluate Teacher and Student, save confusion matrices.")
    parser.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dropout", type=float, default=config.DEFAULT_DROPOUT)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--image-size", type=int, default=config.IMAGE_SIZE)
    args = parser.parse_args()

    device = get_device(args.device)
    loaders = make_dataloaders(config.DATASET_DIR, batch_size=args.batch_size, image_size=args.image_size)
    class_names = loaders["class_names"]

    teacher = build_model("teacher", num_classes=len(class_names), dropout=args.dropout, pretrained=False)
    load_checkpoint(teacher, config.CHECKPOINT_DIR / "teacher_ccinet_best.pt", device)
    teacher.to(device)
    teacher_metrics = evaluate_model(teacher, config.DATASET_DIR, device, "teacher_ccinet", args.batch_size, image_size=args.image_size)
    print("Teacher metrics:", teacher_metrics)

    student = build_model("student", num_classes=len(class_names), dropout=args.dropout, pretrained=args.pretrained)
    load_checkpoint(student, config.CHECKPOINT_DIR / "student_swin_resnet50fpnv2_best.pt", device)
    student.to(device)
    student_metrics = evaluate_model(student, config.DATASET_DIR, device, "student_swin_resnet50fpnv2", args.batch_size, image_size=args.image_size)
    print("Student metrics:", student_metrics)


if __name__ == "__main__":
    main()
