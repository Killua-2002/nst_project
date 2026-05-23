from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NUM_CLASSES, DEFAULT_DROPOUT


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, ch, dropout=0.0):
        super().__init__()
        self.c1 = ConvBNReLU(ch, ch, dropout=dropout)
        self.c2 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.c2(self.c1(x)))


class CCINetTeacher(nn.Module):
    """Compact CCI-Net style teacher for A/B/C segmentation.

    It is not a classifier. Output shape is [B,4,H,W] with classes:
    background, A-only, B-only, C-overlap.
    """

    def __init__(self, num_classes=NUM_CLASSES, dropout=DEFAULT_DROPOUT):
        super().__init__()
        self.enc1 = nn.Sequential(ConvBNReLU(1, 32), ConvBNReLU(32, 32))
        self.enc2 = nn.Sequential(ConvBNReLU(32, 64, s=2), ConvBNReLU(64, 64))
        self.enc3 = nn.Sequential(ConvBNReLU(64, 128, s=2), ConvBNReLU(128, 128, dropout=dropout))
        self.enc4 = nn.Sequential(ConvBNReLU(128, 256, s=2), ConvBNReLU(256, 256, dropout=dropout))

        # Multi-scale feature fusion similar in spirit to CCI-Net MFF.
        self.fuse3 = ConvBNReLU(256 + 128, 128, dropout=dropout)
        self.fuse2 = ConvBNReLU(128 + 64, 64)
        self.fuse1 = ConvBNReLU(64 + 32, 32)
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        h, w = x.shape[-2:]
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d3 = F.interpolate(e4, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.fuse3(torch.cat([d3, e3], dim=1))
        d2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.fuse2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.fuse1(torch.cat([d1, e1], dim=1))
        out = self.head(d1)
        return F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)


class WindowAttentionLite(nn.Module):
    """Lightweight Swin-like local attention block.

    Keeps the project Colab-friendly while preserving the intended idea:
    local window context + CNN/FPN features.
    """

    def __init__(self, ch, window_size=7):
        super().__init__()
        self.window_size = window_size
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.norm = nn.BatchNorm2d(ch)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=1)
        # Local context via depth-wise average as a stable approximation.
        k_ctx = F.avg_pool2d(k, kernel_size=self.window_size, stride=1, padding=self.window_size // 2)
        att = torch.sigmoid((q * k_ctx).sum(dim=1, keepdim=True) / (c ** 0.5))
        out = self.proj(v * att)
        return self.norm(out + x)


class StudentSwinResNetFPNV2(nn.Module):
    """Student segmentation model: Swin-like context + ResNet50-FPN-v2 style decoder.

    Output is pixel-level A/B/C segmentation, not image classification.
    """

    def __init__(self, num_classes=NUM_CLASSES, dropout=DEFAULT_DROPOUT):
        super().__init__()
        self.stem = nn.Sequential(ConvBNReLU(1, 32), ConvBNReLU(32, 64, s=2))
        self.layer1 = nn.Sequential(ResidualBlock(64), ResidualBlock(64))
        self.down2 = ConvBNReLU(64, 128, s=2)
        self.layer2 = nn.Sequential(ResidualBlock(128), ResidualBlock(128))
        self.down3 = ConvBNReLU(128, 256, s=2)
        self.layer3 = nn.Sequential(ResidualBlock(256, dropout=dropout), ResidualBlock(256, dropout=dropout))
        self.down4 = ConvBNReLU(256, 512, s=2)
        self.layer4 = nn.Sequential(ResidualBlock(512, dropout=dropout), WindowAttentionLite(512), ResidualBlock(512, dropout=dropout))

        # FPN v2 style lateral + top-down fusion.
        self.lat4 = nn.Conv2d(512, 128, 1)
        self.lat3 = nn.Conv2d(256, 128, 1)
        self.lat2 = nn.Conv2d(128, 128, 1)
        self.lat1 = nn.Conv2d(64, 128, 1)

        self.smooth3 = ConvBNReLU(128, 128)
        self.smooth2 = ConvBNReLU(128, 128)
        self.smooth1 = ConvBNReLU(128, 128)

        self.decoder = nn.Sequential(
            ConvBNReLU(128, 96),
            nn.Dropout2d(dropout),
            ConvBNReLU(96, 64),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        c1 = self.layer1(self.stem(x))     # /2
        c2 = self.layer2(self.down2(c1))   # /4
        c3 = self.layer3(self.down3(c2))   # /8
        c4 = self.layer4(self.down4(c3))   # /16

        p4 = self.lat4(c4)
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p3 = self.smooth3(p3)
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        p2 = self.smooth2(p2)
        p1 = self.lat1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode="nearest")
        p1 = self.smooth1(p1)

        out = self.decoder(p1)
        return F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)


def build_model(name: str, dropout: float = DEFAULT_DROPOUT, num_classes: int = NUM_CLASSES):
    key = name.lower()
    if key in {"teacher", "cci", "cci-net", "ccinet"}:
        return CCINetTeacher(num_classes=num_classes, dropout=dropout)
    if key in {"student", "swin", "swin-resnet-fpn", "swin_resnet_fpn_v2"}:
        return StudentSwinResNetFPNV2(num_classes=num_classes, dropout=dropout)
    raise ValueError(f"Unknown model name: {name}")
