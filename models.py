from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            SEBlock(out_ch),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CCINetTeacherSeg(nn.Module):
    """CCI-Net-inspired teacher for 4-class A/B/C segmentation.

    It uses a CNN backbone, SE attention blocks, multi-scale feature fusion, and a decoder.
    Output classes: 0 background, 1 A-only, 2 B-only, 3 C-overlap.
    """
    def __init__(self, num_classes: int = 4, dropout: float = 0.5, base: int = 16):
        super().__init__()
        self.e1 = ConvBlock(1, base, dropout=0.0)
        self.e2 = ConvBlock(base, base * 2, dropout=0.1)
        self.e3 = ConvBlock(base * 2, base * 4, dropout=dropout)
        self.e4 = ConvBlock(base * 4, base * 8, dropout=dropout)
        self.pool = nn.MaxPool2d(2)

        self.mff1 = nn.Conv2d(base, base, 1)
        self.mff2 = nn.Conv2d(base * 2, base, 1)
        self.mff3 = nn.Conv2d(base * 4, base, 1)
        self.mff4 = nn.Conv2d(base * 8, base, 1)

        self.d3 = ConvBlock(base * 8 + base * 4, base * 4, dropout=dropout)
        self.d2 = ConvBlock(base * 4 + base * 2, base * 2, dropout=0.1)
        self.d1 = ConvBlock(base * 2 + base, base, dropout=0.0)
        self.fuse = ConvBlock(base * 2, base * 2, dropout=0.1)
        self.out = nn.Conv2d(base * 2, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))

        d3 = F.interpolate(e4, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.d3(torch.cat([d3, e3], dim=1))
        d2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.d2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.d1(torch.cat([d1, e1], dim=1))

        # Multi-scale feature fusion like CCI-Net style.
        f1 = self.mff1(e1)
        f2 = F.interpolate(self.mff2(e2), size=e1.shape[-2:], mode="bilinear", align_corners=False)
        f3 = F.interpolate(self.mff3(e3), size=e1.shape[-2:], mode="bilinear", align_corners=False)
        f4 = F.interpolate(self.mff4(e4), size=e1.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.fuse(torch.cat([d1, f1 + f2 + f3 + f4], dim=1))
        return self.out(fused)


class WindowAttentionBlock(nn.Module):
    """Lightweight Swin-style local window attention block for FPN features.

    This version uses local window/depthwise attention gates instead of full MHA so the
    notebook can run reliably on Colab. It keeps the Swin idea of local-window context.
    """
    def __init__(self, dim: int = 32, heads: int = 4, window_size: int = 7, dropout: float = 0.1):
        super().__init__()
        self.local_gate = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=window_size, padding=window_size // 2, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.Sigmoid(),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.proj(x * self.local_gate(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        return F.relu(y + self.skip(x), inplace=True)


class SwinResNetFPNStudentSeg(nn.Module):
    """Student segmentation model: lightweight Swin + ResNet-FPN-v2 inspired.

    The original requested name is kept, but this implementation is intentionally
    lightweight for Google Colab: residual CNN encoder, FPN multi-scale fusion,
    and a local window attention gate similar to Swin's local-context idea.
    """
    def __init__(self, num_classes: int = 4, dropout: float = 0.5, fpn_dim: int = 32):
        super().__init__()
        self.e1 = ConvBlock(1, 32, dropout=0.0)          # H
        self.e2 = ConvBlock(32, 64, dropout=0.1)         # H/2
        self.e3 = ConvBlock(64, 128, dropout=dropout)    # H/4
        self.e4 = ConvBlock(128, 128, dropout=dropout)   # H/8
        self.pool = nn.MaxPool2d(2)

        self.lat1 = nn.Conv2d(32, fpn_dim, 1)
        self.lat2 = nn.Conv2d(64, fpn_dim, 1)
        self.lat3 = nn.Conv2d(128, fpn_dim, 1)
        self.lat4 = nn.Conv2d(128, fpn_dim, 1)
        self.swin_block = WindowAttentionBlock(dim=fpn_dim, heads=4, window_size=7, dropout=0.1)
        self.head = nn.Sequential(
            ConvBlock(fpn_dim * 4, 64, dropout=dropout),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))

        p4 = self.lat4(e4)
        p3 = self.lat3(e3) + F.interpolate(p4, size=e3.shape[-2:], mode="nearest")
        p2 = self.lat2(e2) + F.interpolate(p3, size=e2.shape[-2:], mode="nearest")
        p1 = self.lat1(e1) + F.interpolate(p2, size=e1.shape[-2:], mode="nearest")
        p1 = self.swin_block(p1)

        size = p1.shape[-2:]
        y = torch.cat([
            p1,
            F.interpolate(p2, size=size, mode="bilinear", align_corners=False),
            F.interpolate(p3, size=size, mode="bilinear", align_corners=False),
            F.interpolate(p4, size=size, mode="bilinear", align_corners=False),
        ], dim=1)
        return F.interpolate(self.head(y), size=input_size, mode="bilinear", align_corners=False)


def build_model(name: str, num_classes: int = 4, dropout: float = 0.5) -> nn.Module:
    key = name.lower().strip()
    if key in {"teacher", "cci", "cci-net", "ccinet"}:
        return CCINetTeacherSeg(num_classes=num_classes, dropout=dropout)
    if key in {"student", "swin-resnet-fpn", "swin_resnet_fpn"}:
        return SwinResNetFPNStudentSeg(num_classes=num_classes, dropout=dropout)
    raise ValueError(f"Unknown model name: {name}")
