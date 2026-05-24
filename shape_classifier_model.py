
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()
        hidden = max(4, ch // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.net(x)


class ShapeClassifierCCINet(nn.Module):
    """Classification-style model used as a chromosome-shape validator.

    Input: one binary/soft mask [B,1,H,W].
    Output: 2 logits:
        0 = invalid / broken / holey / non-chromosome-like mask
        1 = valid single-chromosome-like shape

    This is intentionally small and Colab-friendly. It follows the CCI-Net idea:
    CNN backbone + SE blocks + multi-scale pooled feature fusion.
    """

    def __init__(self, dropout: float = 0.5):
        super().__init__()

        def block(in_ch, out_ch, p):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                SEBlock(out_ch),
                nn.MaxPool2d(2),
                nn.Dropout2d(p),
            )

        self.s1 = block(1, 24, 0.05)
        self.s2 = block(24, 48, 0.10)
        self.s3 = block(48, 96, 0.20)
        self.s4 = block(96, 160, 0.25)

        self.p2 = nn.Conv2d(48, 96, 1)
        self.p3 = nn.Conv2d(96, 96, 1)
        self.p4 = nn.Conv2d(160, 96, 1)

        self.head = nn.Sequential(
            nn.Linear(96 * 3, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(192, 2),
        )

    def forward(self, x):
        x1 = self.s1(x)
        x2 = self.s2(x1)
        x3 = self.s3(x2)
        x4 = self.s4(x3)

        f2 = F.adaptive_avg_pool2d(self.p2(x2), 1).flatten(1)
        f3 = F.adaptive_avg_pool2d(self.p3(x3), 1).flatten(1)
        f4 = F.adaptive_avg_pool2d(self.p4(x4), 1).flatten(1)
        return self.head(torch.cat([f2, f3, f4], dim=1))


def mask_to_tensor(mask: np.ndarray, device=None) -> torch.Tensor:
    arr = mask.astype(np.float32)
    if arr.max() > 1:
        arr = arr / 255.0
    x = torch.from_numpy(arr)[None, None, :, :]
    if device is not None:
        x = x.to(device)
    return x


@torch.no_grad()
def shape_probability(model: nn.Module | None, mask: np.ndarray, device=None) -> float:
    """Return probability that a mask looks like one valid chromosome.

    If no classifier checkpoint is available, return 0.5 so the pipeline still
    runs using skeleton + segmentation scores.
    """
    if model is None:
        return 0.5
    model.eval()
    x = mask_to_tensor(mask, device=device)
    prob = torch.softmax(model(x), dim=1)[0, 1].item()
    return float(prob)


def load_shape_classifier(path: Path, device, dropout: float = 0.5) -> ShapeClassifierCCINet | None:
    path = Path(path)
    if not path.exists():
        return None
    model = ShapeClassifierCCINet(dropout=dropout).to(device)
    ckpt = torch.load(path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    return model
