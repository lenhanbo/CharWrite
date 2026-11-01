from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm


class ConvBlock(nn.Module):
    """Small classifier-only convolution block (not used by WriteViT)."""

    def __init__(self, in_channels: int, out_channels: int, stride=1, norm=True):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_channels, out_channels, 3, stride, 1)]
        if norm:
            layers.append(nn.GroupNorm(min(8, out_channels), out_channels))
        layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


class GlobalLayerNorm(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(value, value.shape[1:], weight=None, bias=None, eps=1e-5)


class PositionalEncoding(nn.Module):
    def __init__(self, dimension: int, dropout=0.0, max_length=100):
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(torch.arange(0, dimension, 2).float() * (-math.log(10000.0) / dimension))
        encoding = torch.zeros(max_length, dimension)
        encoding[:, 0::2] = torch.sin(position * scale)
        encoding[:, 1::2] = torch.cos(position * scale)
        self.register_buffer("pe", encoding.unsqueeze(0), persistent=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.dropout(tokens + self.pe[:, : tokens.shape[1]])


class PosCNN(nn.Module):
    """Conditional positional encoding (PEG) used by WriteViT."""

    def __init__(self, in_channels: int, embed_dim: int | None = None, stride=1):
        super().__init__()
        embed_dim = embed_dim or in_channels
        self.proj = nn.Conv2d(in_channels, embed_dim, 3, stride, 1, bias=True, groups=embed_dim)
        self.stride = stride

    def forward(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, _, channels = tokens.shape
        feature = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        output = self.proj(feature)
        if self.stride == 1:
            output = output + feature
        return output.flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dimension: int, heads=8, qkv_bias=False, dropout=0.0, spectral=False):
        super().__init__()
        if dimension % heads:
            raise ValueError("dimension must be divisible by heads")
        wrap = spectral_norm if spectral else (lambda layer: layer)
        self.heads = heads
        self.scale = (dimension // heads) ** -0.5
        self.qkv = wrap(nn.Linear(dimension, dimension * 3, bias=qkv_bias))
        self.proj = wrap(nn.Linear(dimension, dimension))
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, length, dimension = tokens.shape
        qkv = self.qkv(tokens).reshape(batch, length, 3, self.heads, dimension // self.heads).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        scores = (query @ key.transpose(-2, -1)) * self.scale
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        weights = self.attn_drop(scores.softmax(-1))
        output = (weights @ value).transpose(1, 2).reshape(batch, length, dimension)
        return self.proj_drop(self.proj(output))


class CrossAttention(nn.Module):
    def __init__(self, query_dim: int, key_dim: int, heads=4, qkv_bias=False, dropout=0.0):
        super().__init__()
        if query_dim % heads:
            raise ValueError("query_dim must be divisible by heads")
        self.query_dim, self.heads = query_dim, heads
        self.scale = (query_dim // heads) ** -0.5
        self.q = nn.Linear(query_dim, query_dim, bias=qkv_bias)
        self.k = nn.Linear(key_dim, query_dim, bias=qkv_bias)
        self.v = nn.Linear(key_dim, query_dim, bias=qkv_bias)
        self.proj = nn.Linear(query_dim, query_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.noise_strength_1 = nn.Parameter(torch.zeros(()))

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch, length, _ = query.shape
        context_length = context.shape[1]
        q = self.q(query).reshape(batch, length, self.heads, -1).permute(0, 2, 1, 3)
        k = self.k(context).reshape(batch, context_length, self.heads, -1).permute(0, 2, 1, 3)
        v = self.v(context).reshape(batch, context_length, self.heads, -1).permute(0, 2, 1, 3)
        weights = self.attn_drop(((q @ k.transpose(-2, -1)) * self.scale).softmax(-1))
        output = (weights @ v).transpose(1, 2).reshape(batch, length, self.query_dim)
        return query + self.proj_drop(self.proj(output))


class MLP(nn.Module):
    def __init__(self, dimension: int, ratio=4.0, dropout=0.0, spectral=False):
        super().__init__()
        wrap = spectral_norm if spectral else (lambda layer: layer)
        hidden = int(dimension * ratio)
        self.fc1 = wrap(nn.Linear(dimension, hidden))
        self.fc2 = wrap(nn.Linear(hidden, dimension))
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.dropout(F.gelu(self.fc1(tokens)))))


class TransformerBlock(nn.Module):
    """The pre-norm encoder block used by WriteViT."""

    def __init__(self, dimension: int, heads=4, mlp_ratio=4.0, dropout=0.0,
                 qkv_bias=True, spectral=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dimension)
        self.attention = Attention(dimension, heads, qkv_bias, dropout, spectral)
        self.norm2 = nn.LayerNorm(dimension)
        self.mlp = MLP(dimension, mlp_ratio, dropout, spectral)

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        tokens = tokens + self.attention(self.norm1(tokens), padding_mask)
        return tokens + self.mlp(self.norm2(tokens))


class CrossBlock(nn.Module):
    """WriteViT block: self attention, norm, cross attention, then MLP."""

    def __init__(self, dimension: int, heads=4, mlp_ratio=4.0, dropout=0.0, qkv_bias=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dimension)
        self.norm2 = nn.LayerNorm(dimension)
        self.norm3 = nn.LayerNorm(dimension)
        self.attention = Attention(dimension, heads, qkv_bias, dropout)
        self.cross_attention = CrossAttention(dimension, dimension, heads, qkv_bias, dropout)
        self.mlp = MLP(dimension, mlp_ratio, dropout)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        query = query + self.attention(self.norm1(query))
        query = self.norm2(query)
        # Preserve the released WriteViT implementation exactly: CrossAttention
        # already contains an inner residual, and CrossBlock adds the outer one.
        query = query + self.cross_attention(query, context)
        return query + self.mlp(self.norm3(query))


class CharacterStyleBlock(nn.Module):
    """The sole CharWriteViT addition: independent per-character style refinement."""

    def __init__(self, dimension: int, heads=4, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.query_norm = nn.LayerNorm(dimension)
        self.context_norm = nn.LayerNorm(dimension)
        self.cross_attention = CrossAttention(dimension, dimension, heads, True, dropout)
        self.mlp_norm = nn.LayerNorm(dimension)
        self.mlp = MLP(dimension, mlp_ratio, dropout)

    def forward(self, characters: torch.Tensor, writer_tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.query_norm(characters)
        characters = characters + (self.cross_attention(normalized, self.context_norm(writer_tokens)) - normalized)
        return characters + self.mlp(self.mlp_norm(characters))


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int,
                 padding=0, norm="none", activation="relu", pad_type="zero"):
        super().__init__()
        pads = {"reflect": nn.ReflectionPad2d, "replicate": nn.ReplicationPad2d, "zero": nn.ZeroPad2d}
        self.pad = pads[pad_type](padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, bias=True)
        self.norm = nn.InstanceNorm2d(out_channels) if norm == "in" else None
        activations = {"relu": nn.ReLU(inplace=False), "lrelu": nn.LeakyReLU(0.2, inplace=False), "tanh": nn.Tanh(), "none": None}
        self.activation = activations[activation]

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        feature = self.conv(self.pad(feature))
        if self.norm is not None:
            feature = self.norm(feature)
        if self.activation is not None:
            feature = self.activation(feature)
        return feature


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            Conv2dBlock(channels, channels, 3, 1, 1, "in", "relu", "reflect"),
            Conv2dBlock(channels, channels, 3, 1, 1, "in", "none", "reflect"),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return feature + self.body(feature)


class StyleEncoder(nn.Module):
    """WriteViT Writer encoder; multiple references are averaged per patch."""

    def __init__(self, style_dim=256, depth=3, heads=4, dropout=0.0):
        super().__init__()
        self.input_norm = GlobalLayerNorm()
        self.patch = nn.Conv2d(1, style_dim, kernel_size=8, stride=4, padding=2)
        self.blocks = nn.ModuleList([
            TransformerBlock(style_dim, heads, 4.0, dropout, qkv_bias=True) for _ in range(depth)
        ])
        self.cpe = PosCNN(style_dim, style_dim)
        self.norm = nn.LayerNorm(style_dim)

    def forward(self, references: torch.Tensor) -> torch.Tensor:
        if references.ndim == 4:
            references = references.unsqueeze(1)
        batch, count, channels, height, width = references.shape
        feature = self.patch(self.input_norm(references.reshape(batch * count, channels, height, width)))
        _, dimension, grid_h, grid_w = feature.shape
        tokens = feature.flatten(2).transpose(1, 2)
        for index, block in enumerate(self.blocks):
            tokens = block(tokens)
            if index == 0:
                tokens = self.cpe(tokens, grid_h, grid_w)
        tokens = self.norm(tokens).reshape(batch, count, grid_h * grid_w, dimension)
        return tokens.mean(1)
