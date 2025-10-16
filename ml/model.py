"""Shared model definitions for the grid estimator."""

from __future__ import annotations

import torch
from torch import nn


class SqueezeExcite(nn.Module):
    """Channel attention block used inside residual stages."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduced = max(4, channels // reduction)
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        scale = self.layers(x).unsqueeze(-1).unsqueeze(-1)
        return x * scale


class ResidualBlock(nn.Module):
    """Residual block with optional downsampling and squeeze-excite attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SqueezeExcite(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = out + identity
        return self.relu(out)


class GridEstimatorModel(nn.Module):
    """CNN with residual stages that ingests RGB+edge channels and regresses grid parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            ResidualBlock(64, 96, stride=1),
            ResidualBlock(96, 96, stride=1),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(96, 128, stride=2),
            ResidualBlock(128, 128, stride=1),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 192, stride=2),
            ResidualBlock(192, 192, stride=1),
        )
        self.stage4 = nn.Sequential(
            ResidualBlock(192, 256, stride=2),
            ResidualBlock(256, 256, stride=1),
        )
        self.features = nn.Sequential(
            self.stem,
            self.stage1,
            self.stage2,
            self.stage3,
            self.stage4,
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 160),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(160, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.head(self.features(x))
