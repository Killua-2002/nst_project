from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

import config


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


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
    """Light CCI-Net style classifier for teacher model.

    This is a practical implementation of the report idea: backbone + SE blocks +
    multi-scale feature fusion + recognition head.
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES, dropout: float = config.DROPOUT):
        super().__init__()
        self.stage1 = ConvSEBlock(1, 32, dropout=0.05)
        self.stage2 = ConvSEBlock(32, 64, dropout=0.10)
        self.stage3 = ConvSEBlock(64, 128, dropout=0.20)
        self.stage4 = ConvSEBlock(128, 256, dropout=0.25)

        self.proj1 = nn.Conv2d(32, 64, 1)
        self.proj2 = nn.Conv2d(64, 64, 1)
        self.proj3 = nn.Conv2d(128, 64, 1)
        self.proj4 = nn.Conv2d(256, 64, 1)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        if x.shape[1] != 1:
            x = x.mean(dim=1, keepdim=True)
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        target_size = f4.shape[-2:]
        p1 = F.adaptive_avg_pool2d(self.proj1(f1), target_size)
        p2 = F.adaptive_avg_pool2d(self.proj2(f2), target_size)
        p3 = F.adaptive_avg_pool2d(self.proj3(f3), target_size)
        p4 = self.proj4(f4)
        fused = torch.cat([p1, p2, p3, p4], dim=1)
        return self.head(fused)


class SmallCNNFallback(nn.Module):
    def __init__(self, num_classes: int = config.NUM_CLASSES, dropout: float = config.DROPOUT):
        super().__init__()
        self.features = nn.Sequential(
            ConvSEBlock(1, 32, dropout=0.10),
            ConvSEBlock(32, 64, dropout=0.15),
            ConvSEBlock(64, 128, dropout=0.20),
            ConvSEBlock(128, 256, dropout=0.25),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        if x.shape[1] != 1:
            x = x.mean(dim=1, keepdim=True)
        return self.head(self.features(x))


class SwinResNet50FPNv2Student(nn.Module):
    """Student model: Swin Transformer + ResNet50 FPN v2-style fusion.

    Input remains grayscale. Internally it is repeated to 3 channels because the
    torchvision Swin/ResNet backbones expect RGB tensors.
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES, dropout: float = config.DROPOUT, pretrained: bool = False):
        super().__init__()
        try:
            from torchvision.models import ResNet50_Weights, Swin_T_Weights, resnet50, swin_t
            from torchvision.models.feature_extraction import create_feature_extractor
            from torchvision.ops import FeaturePyramidNetwork
        except Exception as exc:
            raise ImportError(
                "SwinResNet50FPNv2Student requires torchvision with swin_t and FeaturePyramidNetwork. "
                "Install requirements.txt or use --model small_cnn."
            ) from exc

        swin_weights = Swin_T_Weights.DEFAULT if pretrained else None
        resnet_weights = ResNet50_Weights.DEFAULT if pretrained else None

        self.swin = swin_t(weights=swin_weights)
        swin_dim = self.swin.head.in_features
        self.swin.head = nn.Identity()

        resnet = resnet50(weights=resnet_weights)
        return_nodes = {
            "layer1": "c2",
            "layer2": "c3",
            "layer3": "c4",
            "layer4": "c5",
        }
        self.resnet_features = create_feature_extractor(resnet, return_nodes=return_nodes)
        self.fpn = FeaturePyramidNetwork(in_channels_list=[256, 512, 1024, 2048], out_channels=256)

        self.classifier = nn.Sequential(
            nn.Linear(swin_dim + 256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        if x.shape[1] == 1:
            x3 = x.repeat(1, 3, 1, 1)
        else:
            x3 = x
        swin_feat = self.swin(x3)
        res_feats = self.resnet_features(x3)
        fpn_feats = self.fpn(res_feats)
        pooled = []
        for key in sorted(fpn_feats.keys()):
            pooled.append(F.adaptive_avg_pool2d(fpn_feats[key], 1).flatten(1))
        fpn_feat = torch.stack(pooled, dim=0).mean(dim=0)
        feat = torch.cat([swin_feat, fpn_feat], dim=1)
        return self.classifier(feat)


def get_model(name: str, num_classes: int = config.NUM_CLASSES, dropout: float = config.DROPOUT, pretrained: bool = False) -> nn.Module:
    name = name.lower().strip()
    if name in {"cci", "cci_net", "teacher"}:
        return CCINetTeacher(num_classes=num_classes, dropout=dropout)
    if name in {"swin_resnet50_fpn_v2", "student", "swin_resnet"}:
        try:
            return SwinResNet50FPNv2Student(num_classes=num_classes, dropout=dropout, pretrained=pretrained)
        except Exception as exc:
            print("[WARNING] Could not build Swin+ResNet50FPNv2 student:", exc)
            print("[WARNING] Falling back to SmallCNNFallback so the pipeline can still run.")
            return SmallCNNFallback(num_classes=num_classes, dropout=dropout)
    if name in {"small_cnn", "fallback"}:
        return SmallCNNFallback(num_classes=num_classes, dropout=dropout)
    raise ValueError(f"Unknown model name: {name}")


def load_checkpoint(model: nn.Module, checkpoint_path: str | None, map_location="cpu") -> nn.Module:
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=map_location)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state, strict=False)
    return model
