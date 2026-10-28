from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import spectral_norm

from .common import ConvBlock, PosCNN, StyleEncoder, TransformerBlock
from .resnet18 import ResNet18


class CharacterClassifier(nn.Module):
    def __init__(self, classes: int, width=64):
        super().__init__()
        self.features = nn.Sequential(ConvBlock(1, width, 2), ConvBlock(width, width * 2, 2), ConvBlock(width * 2, width * 2, 2), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(width * 2, classes)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(image).flatten(1))


class WordOCR(nn.Module):
    """WriteViT ViT recognizer with a ResNet18 spatial patch encoder."""

    def __init__(self, classes: int, width=256, depth=3, heads=8):
        super().__init__()
        self.features = ResNet18(width)
        self.blocks = nn.ModuleList([TransformerBlock(width, heads) for _ in range(depth)])
        self.cpe = PosCNN(width); self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, classes)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feature = self.features(image); batch, channels, height, width = feature.shape
        tokens = feature.flatten(2).transpose(1, 2)
        for index, block in enumerate(self.blocks):
            tokens = block(tokens)
            if index == 0: tokens = self.cpe(tokens, height, width)
        return self.head(self.norm(tokens))


class SNResidualDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1=spectral_norm(nn.Conv2d(in_channels,out_channels,3,1,1)); self.conv2=spectral_norm(nn.Conv2d(out_channels,out_channels,3,1,1)); self.skip=spectral_norm(nn.Conv2d(in_channels,out_channels,1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        residual=nn.functional.avg_pool2d(self.skip(image),2)
        feature=nn.functional.leaky_relu(image,.2); feature=self.conv1(feature); feature=nn.functional.leaky_relu(feature,.2); feature=self.conv2(feature); feature=nn.functional.avg_pool2d(feature,2)
        return feature+residual


class PatchDiscriminator(nn.Module):
    """BigGAN-style spectral residual discriminator for variable-width words."""

    def __init__(self, base=32):
        super().__init__()
        self.net = nn.Sequential(SNResidualDownBlock(1,base),SNResidualDownBlock(base,base*2),SNResidualDownBlock(base*2,base*4))
        self.output=spectral_norm(nn.Conv2d(base*4,1,3,1,1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.output(nn.functional.leaky_relu(self.net(image),.2))


class WriterStyleNetwork(nn.Module):
    def __init__(self, writers: int, style_dim=128):
        super().__init__()
        self.encoder = StyleEncoder(style_dim)
        self.head = nn.Linear(style_dim, writers)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        style = self.encoder(images).mean(1)
        return style, self.head(style)

