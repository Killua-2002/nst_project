from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import config
from src.image_processing import list_images


class ChromosomeFolderDataset(Dataset):
    """ImageFolder-like dataset, but always loads images as grayscale.

    The tensor shape is [1, H, W]. Models can internally repeat it to 3 channels
    if they use ImageNet-style backbones.
    """

    def __init__(self, root: Path, train: bool = False):
        self.root = Path(root)
        self.train = train
        self.samples: List[Tuple[Path, int]] = []
        for label_idx, cls in enumerate(config.CLASS_NAMES):
            cls_dir = self.root / cls
            for path in list_images(cls_dir):
                self.samples.append((path, label_idx))

        if train:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.75, 1.00)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        img = Image.open(path).convert("L")
        return self.transform(img), label


class UnlabeledChromosomeDataset(Dataset):
    def __init__(self, root: Path, strong: bool = False):
        self.root = Path(root)
        self.paths = list_images(self.root)
        if strong:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.60, 1.00)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        img = Image.open(path).convert("L")
        return self.transform(img), str(path)
