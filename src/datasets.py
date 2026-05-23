from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import config


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        path, _ = self.samples[index]
        return image, label, path


class UnlabeledImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = [Path(p) for p in image_paths]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, str(path)


def get_transforms(image_size: int = config.IMAGE_SIZE, train: bool = True):
    """Images are stored/read as grayscale.

    Backbone models usually expect 3 channels, so Grayscale(3) repeats the same
    gray channel three times without adding color information.
    """
    if train:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15, fill=255),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def make_imagefolder_loader(root: Path, batch_size: int, train: bool, num_workers: int = 2):
    root = Path(root)
    ds = ImageFolderWithPaths(root, transform=get_transforms(train=train))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
    )
    return ds, loader


def make_dataloaders(dataset_dir: Path, batch_size: int = config.DEFAULT_BATCH_SIZE, num_workers: int = 2):
    dataset_dir = Path(dataset_dir)
    train_ds, train_loader = make_imagefolder_loader(dataset_dir / "train", batch_size, train=True, num_workers=num_workers)
    val_ds, val_loader = make_imagefolder_loader(dataset_dir / "val", batch_size, train=False, num_workers=num_workers)
    test_ds, test_loader = make_imagefolder_loader(dataset_dir / "test", batch_size, train=False, num_workers=num_workers)
    return {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "class_names": train_ds.classes,
    }
