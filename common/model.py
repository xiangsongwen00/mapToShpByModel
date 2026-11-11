from __future__ import annotations

import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TinyUNet(nn.Module):
    """A dependency-light U-Net baseline suitable for 3-band binary segmentation."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        b = base_channels
        self.enc1 = DoubleConv(in_channels, b)
        self.enc2 = DoubleConv(b, b * 2)
        self.enc3 = DoubleConv(b * 2, b * 4)
        self.bottleneck = DoubleConv(b * 4, b * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.dec3 = DoubleConv(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.dec2 = DoubleConv(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.dec1 = DoubleConv(b * 2, b)
        self.head = nn.Conv2d(b, 1, 1)

    @staticmethod
    def _join(skip, value):
        if value.shape[-2:] != skip.shape[-2:]:
            value = nn.functional.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([skip, value], dim=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bottleneck = self.bottleneck(self.pool(e3))
        d3 = self.dec3(self._join(e3, self.up3(bottleneck)))
        d2 = self.dec2(self._join(e2, self.up2(d3)))
        d1 = self.dec1(self._join(e1, self.up1(d2)))
        return self.head(d1)


def build_model(config: dict) -> TinyUNet:
    model_name = config.get("name", "tiny_unet")
    if model_name != "tiny_unet":
        raise ValueError(f"不支持的模型: {model_name}")
    return TinyUNet(3, int(config.get("base_channels", 32)))

