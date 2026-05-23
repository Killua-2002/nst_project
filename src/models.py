from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

import config


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class ConvSEBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            SEBlock(out_ch),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.block(x)


class CCINetTeacher(nn.Module):
    """Lightweight CCI-Net style teacher.

    It uses CNN backbone + SE attention + multi-scale feature fusion head.
    This is intentionally small enough for Colab while matching the report idea.
    """
    def __init__(self, num_classes: int = config.NUM_CLASSES, dropout: float = config.DEFAULT_DROPOUT):
        super().__init__()
        self.stage1 = ConvSEBlock(3, 32, dropout=0.05)
        self.stage2 = ConvSEBlock(32, 64, dropout=0.10)
        self.stage3 = ConvSEBlock(64, 128, dropout=0.15)
        self.stage4 = ConvSEBlock(128, 256, dropout=0.20)
        self.proj2 = nn.Conv2d(64, 128, 1)
        self.proj3 = nn.Conv2d(128, 128, 1)
        self.proj4 = nn.Conv2d(256, 128, 1)
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        f2 = F.adaptive_avg_pool2d(self.proj2(x2), 1).flatten(1)
        f3 = F.adaptive_avg_pool2d(self.proj3(x3), 1).flatten(1)
        f4 = F.adaptive_avg_pool2d(self.proj4(x4), 1).flatten(1)
        return self.classifier(torch.cat([f2, f3, f4], dim=1))


class ResNet50FPNv2(nn.Module):
    """ResNet50 + simple FPN-v2 style top-down feature pyramid."""
    def __init__(self, pretrained: bool = False, out_channels: int = 256):
        super().__init__()
        weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
        base = tv_models.resnet50(weights=weights)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.lat2 = nn.Conv2d(512, out_channels, 1)
        self.lat3 = nn.Conv2d(1024, out_channels, 1)
        self.lat4 = nn.Conv2d(2048, out_channels, 1)
        self.smooth2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.out_channels = out_channels

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p5 = self.lat4(c5)
        p4 = self.lat3(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lat2(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p3 = self.smooth2(p3)
        p4 = self.smooth3(p4)
        p5 = self.smooth4(p5)
        pooled = [F.adaptive_avg_pool2d(p, 1).flatten(1) for p in [p3, p4, p5]]
        return torch.stack(pooled, dim=0).mean(dim=0)


class SwinResNetFPNStudent(nn.Module):
    """Student = Swin Transformer + ResNet50 FPN v2.

    timm is used for Swin. ResNet50 FPN is implemented above. The two feature
    vectors are concatenated and classified with Dropout=0.5.
    """
    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        dropout: float = config.DEFAULT_DROPOUT,
        pretrained: bool = False,
        swin_name: str = "swin_tiny_patch4_window7_224",
    ):
        super().__init__()
        import timm

        self.swin = timm.create_model(
            swin_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=3,
        )
        swin_dim = self.swin.num_features
        self.fpn = ResNet50FPNv2(pretrained=pretrained, out_channels=256)
        self.classifier = nn.Sequential(
            nn.Linear(swin_dim + 256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        swin_feat = self.swin(x)
        fpn_feat = self.fpn(x)
        return self.classifier(torch.cat([swin_feat, fpn_feat], dim=1))


def build_model(name: str, num_classes: int = config.NUM_CLASSES, dropout: float = config.DEFAULT_DROPOUT, pretrained: bool = False):
    name = name.lower()
    if name in {"teacher", "cci", "cci-net", "ccinet"}:
        return CCINetTeacher(num_classes=num_classes, dropout=dropout)
    if name in {"student", "swin_resnet_fpn", "swin-resnet-fpn"}:
        return SwinResNetFPNStudent(num_classes=num_classes, dropout=dropout, pretrained=pretrained)
    raise ValueError(f"Unknown model name: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
