from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride=1, norm=True):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_channels, out_channels, 3, stride, 1)]
        if norm:
            layers.append(nn.GroupNorm(min(8, out_channels), out_channels))
        layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PosCNN(nn.Module):
    """Conditional positional encoding used by WriteViT."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, _, channels = tokens.shape
        feature = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        return (feature + self.proj(feature)).flatten(2).transpose(1, 2)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads=4, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim), nn.Dropout(dropout))

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        normalized = self.norm1(tokens)
        attended, _ = self.attention(normalized, normalized, normalized, key_padding_mask=padding_mask, need_weights=False)
        tokens = tokens + attended
        return tokens + self.mlp(self.norm2(tokens))


class CrossBlock(nn.Module):
    def __init__(self, dim: int, context_dim: int | None = None, heads=4, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        context_dim = context_dim or dim
        self.query_norm = nn.LayerNorm(dim); self.context_norm = nn.LayerNorm(context_dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True, kdim=context_dim, vdim=context_dim)
        self.norm = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim), nn.Dropout(dropout))

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(self.query_norm(query), self.context_norm(context), self.context_norm(context), need_weights=False)
        query = query + attended
        return query + self.mlp(self.norm(query))


class StyleEncoder(nn.Module):
    """WriteViT-style ViT Writer Encoder returning spatial style tokens."""

    def __init__(self, style_dim=256, depth=3, heads=4, dropout=0.0):
        super().__init__()
        self.patch = nn.Conv2d(1, style_dim, kernel_size=8, stride=4, padding=2)
        self.blocks = nn.ModuleList([TransformerBlock(style_dim, heads, dropout=dropout) for _ in range(depth)])
        self.cpe = PosCNN(style_dim)
        self.norm = nn.LayerNorm(style_dim)

    def forward(self, references: torch.Tensor) -> torch.Tensor:
        if references.ndim == 4:
            references = references.unsqueeze(1)
        b, r, c, h, w = references.shape
        features = self.patch(references.reshape(b * r, c, h, w))
        _, channels, grid_h, grid_w = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        for index, block in enumerate(self.blocks):
            tokens = block(tokens)
            if index == 0: tokens = self.cpe(tokens, grid_h, grid_w)
        tokens = self.norm(tokens).reshape(b, r * grid_h * grid_w, channels)
        return tokens

