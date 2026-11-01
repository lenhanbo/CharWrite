from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import spectral_norm

from .common import ConvBlock, GlobalLayerNorm, PosCNN, StyleEncoder, TransformerBlock
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
        self.input_norm = GlobalLayerNorm()
        self.features = ResNet18(width)
        self.blocks = nn.ModuleList([
            TransformerBlock(width, heads, qkv_bias=True, spectral=True)
            for _ in range(depth)
        ])
        self.cpe = PosCNN(width); self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, classes)
        self.output_norm = GlobalLayerNorm()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feature = self.features(self.input_norm(image)); batch, channels, height, width = feature.shape
        tokens = feature.flatten(2).transpose(1, 2)
        for index, block in enumerate(self.blocks):
            tokens = block(tokens)
            if index == 0: tokens = self.cpe(tokens, height, width)
        return self.output_norm(self.head(self.norm(tokens)))


class SNResidualDownBlock(nn.Module):
    """BigGAN DBlock used by the released WriteViT discriminator."""

    def __init__(self, in_channels: int, out_channels: int, preactivation=False, downsample=True):
        super().__init__()
        self.activation=nn.ReLU(inplace=False); self.preactivation=preactivation; self.downsample=downsample
        self.conv1=spectral_norm(nn.Conv2d(in_channels,out_channels,3,1,1)); self.conv2=spectral_norm(nn.Conv2d(out_channels,out_channels,3,1,1))
        self.learnable_skip=in_channels!=out_channels or downsample
        self.skip=spectral_norm(nn.Conv2d(in_channels,out_channels,1)) if self.learnable_skip else nn.Identity()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feature=self.activation(image) if self.preactivation else image
        feature=self.activation(self.conv1(feature)); feature=self.conv2(feature)
        if self.downsample: feature=nn.functional.avg_pool2d(feature,2)
        residual=image
        if self.preactivation and self.learnable_skip: residual=self.skip(residual)
        if self.downsample: residual=nn.functional.avg_pool2d(residual,2)
        if not self.preactivation and self.learnable_skip: residual=self.skip(residual)
        return feature+residual


class PatchDiscriminator(nn.Module):
    """WriteViT's resolution-16 BigGAN discriminator for variable-width words."""

    def __init__(self, base=64):
        super().__init__()
        self.blocks=nn.Sequential(
            SNResidualDownBlock(1,base,False,True),
            SNResidualDownBlock(base,base*4,True,True),
            SNResidualDownBlock(base*4,base*8,True,True),
            SNResidualDownBlock(base*8,base*8,True,False),
        )
        self.activation=nn.ReLU(inplace=False); self.output=spectral_norm(nn.Linear(base*8,1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feature=self.blocks(image); pooled=self.activation(feature).sum(dim=(2,3))
        return self.output(pooled)


class WriterStyleNetwork(nn.Module):
    def __init__(self, writers: int, style_dim=128, encoder: StyleEncoder | None = None):
        super().__init__()
        self.encoder = encoder if encoder is not None else StyleEncoder(style_dim)
        self.head = nn.Linear(style_dim, writers)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        style = self.encoder(images).mean(1)
        return style, self.head(style)

