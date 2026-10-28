from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .common import ConvBlock, CrossBlock, PosCNN, StyleEncoder, TransformerBlock


class GlyphGenerator(nn.Module):
    def __init__(self, vocab_size: int, style_dim=256, hidden_dim=256, char_width=16,
                 unifont_table: torch.Tensor | None = None, depth=5, heads=4, dropout=0.0):
        super().__init__()
        self.char_width = char_width
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.unifont = None
        if unifont_table is not None:
            self.unifont = nn.Embedding.from_pretrained(unifont_table.float(), freeze=True, padding_idx=0)
            self.unifont_projection = nn.Linear(unifont_table.shape[1], hidden_dim)
        self.style_projection = nn.Linear(style_dim, hidden_dim)
        self.cross_blocks = nn.ModuleList([CrossBlock(hidden_dim, hidden_dim, heads, dropout=dropout) for _ in range(depth)])
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4), nn.SiLU(),
            nn.Linear(hidden_dim * 4, 32 * char_width), nn.Tanh(),
        )

    def forward(self, ids: torch.Tensor, style_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        characters = self.embedding(ids)
        if self.unifont is not None:
            characters = characters + self.unifont_projection(self.unifont(ids))
        style = self.style_projection(style_tokens)
        styled = characters
        for block in self.cross_blocks: styled = block(styled, style)
        pixels = self.decoder(torch.cat([characters, styled], dim=-1))
        glyphs = pixels.view(ids.shape[0], ids.shape[1], 1, 32, self.char_width)
        return glyphs, characters, styled


class CharacterContextualizer(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim=128, layers=3, heads=4, dropout=0.1, char_width=16):
        super().__init__()
        self.char_width = char_width
        self.image_encoder = nn.Sequential(
            ConvBlock(1, 32, 2), ConvBlock(32, hidden_dim, 2), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.character = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.position = nn.Parameter(torch.randn(1, 64, hidden_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(hidden_dim, heads, hidden_dim * 4, dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.residual = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.SiLU(), nn.Linear(hidden_dim * 2, 32 * char_width), nn.Tanh())
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, glyphs: torch.Tensor, ids: torch.Tensor, padding_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, length = ids.shape
        encoded = self.image_encoder(glyphs.reshape(b * length, 1, 32, self.char_width)).reshape(b, length, -1)
        tokens = encoded + self.character(ids) + self.position[:, :length]
        context = self.transformer(tokens, src_key_padding_mask=padding_mask)
        correction = self.residual(context).view(b, length, 1, 32, self.char_width)
        images = (glyphs + self.scale.tanh() * correction).clamp(-1, 1)
        return images, context


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3), nn.InstanceNorm2d(channels), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3), nn.InstanceNorm2d(channels),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return F.relu(feature + self.net(feature))


def upsample_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), ConvBlock(in_channels, out_channels), ConvBlock(out_channels, out_channels))


class WriteViTGlyphFusion(nn.Module):
    """WriteViT multi-scale Transformer/CPE/toRGB fusion and CNN decoder."""

    def __init__(self, hidden_dim=256, depth=3, heads=4, dropout=0.0, feature_noise=True, noise_std=1.0):
        super().__init__()
        dims = [hidden_dim, hidden_dim // 2, hidden_dim // 2, hidden_dim // 4]
        self.feature_noise, self.noise_std, self.dims = feature_noise, noise_std, dims
        self.to_rgb1 = nn.Conv2d(dims[0], dims[3], 3, 1, 1)
        self.up1 = upsample_block(dims[0], dims[1])
        self.stage2 = nn.ModuleList([TransformerBlock(dims[1], heads, dropout=dropout) for _ in range(depth)])
        self.cpe2 = PosCNN(dims[1]); self.to_rgb2 = nn.Conv2d(dims[1], dims[3], 3, 1, 1)
        self.up2 = upsample_block(dims[1], dims[2])
        self.stage3 = nn.ModuleList([TransformerBlock(dims[2], heads, dropout=dropout) for _ in range(depth)])
        self.cpe3 = PosCNN(dims[2]); self.to_rgb3 = nn.Conv2d(dims[2], dims[3], 3, 1, 1)
        self.up3 = upsample_block(dims[2], dims[3])
        self.stage4 = nn.ModuleList([TransformerBlock(dims[3], heads, dropout=dropout) for _ in range(depth)])
        self.cpe4 = PosCNN(dims[3])
        self.norm2 = nn.LayerNorm(dims[1]); self.norm3 = nn.LayerNorm(dims[2]); self.norm4 = nn.LayerNorm(dims[3])
        self.decoder = nn.Sequential(
            ResidualConvBlock(dims[3]), ResidualConvBlock(dims[3]),
            nn.Upsample(scale_factor=2, mode="nearest"), ConvBlock(dims[3], max(16, dims[3] // 2)),
            nn.ReflectionPad2d(2), nn.Conv2d(max(16, dims[3] // 2), max(16, dims[3] // 2), 5), nn.InstanceNorm2d(max(16, dims[3] // 2)), nn.ReLU(),
            nn.ReflectionPad2d(3), nn.Conv2d(max(16, dims[3] // 2), 1, 7), nn.Tanh(),
        )

    @staticmethod
    def stack_feature_rows(context_tokens: torch.Tensor, content_tokens: torch.Tensor) -> torch.Tensor:
        if context_tokens.shape != content_tokens.shape:
            raise ValueError("Context and content token shapes must match")
        batch, length, dimension = context_tokens.shape
        # This is the same ordering used by WriteViT: cat [h2, tmp], then view as H=2, W=L.
        fused = torch.cat([context_tokens, content_tokens], dim=1)
        return fused.permute(0, 2, 1).reshape(batch, dimension, 2, length)

    @staticmethod
    def _run_stage(feature: torch.Tensor, blocks: nn.ModuleList, cpe: PosCNN, norm: nn.LayerNorm) -> torch.Tensor:
        batch, channels, height, width = feature.shape
        tokens = feature.flatten(2).transpose(1, 2)
        for index, block in enumerate(blocks):
            tokens = block(tokens)
            if index == 0: tokens = cpe(tokens, height, width)
        return norm(tokens).transpose(1, 2).reshape(batch, channels, height, width)

    def forward(self, context_tokens: torch.Tensor, content_tokens: torch.Tensor,
                noise_seed: int | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        scale1 = self.stack_feature_rows(context_tokens, content_tokens)
        rgb1 = self.to_rgb1(scale1)
        scale2 = self._run_stage(self.up1(scale1), self.stage2, self.cpe2, self.norm2)
        rgb2 = self.to_rgb2(scale2)
        scale3 = self._run_stage(self.up2(scale2), self.stage3, self.cpe3, self.norm3)
        rgb3 = self.to_rgb3(scale3)
        scale4 = self._run_stage(self.up3(scale3), self.stage4, self.cpe4, self.norm4)
        fused = F.interpolate(rgb1, size=scale4.shape[-2:], mode="nearest") + F.interpolate(rgb2, size=scale4.shape[-2:], mode="nearest") + F.interpolate(rgb3, size=scale4.shape[-2:], mode="nearest") + scale4
        if self.feature_noise:
            if noise_seed is None:
                noise = torch.randn_like(fused)
            else:
                generator = torch.Generator(device=fused.device).manual_seed(noise_seed)
                noise = torch.randn(fused.shape, generator=generator, device=fused.device, dtype=fused.dtype)
            fused = fused + self.noise_std * noise
        word = self.decoder(fused)
        return word, {"scale1": scale1, "scale2": scale2, "scale3": scale3, "scale4": scale4, "fused": fused}


class WordMerger(nn.Module):
    def __init__(self, style_dim=128, hidden_dim=128):
        super().__init__()
        self.input = ConvBlock(1, hidden_dim // 2)
        self.body = nn.Sequential(ConvBlock(hidden_dim // 2, hidden_dim), ConvBlock(hidden_dim, hidden_dim), ConvBlock(hidden_dim, hidden_dim // 2))
        self.film = nn.Linear(style_dim, hidden_dim * 2)
        self.output = nn.Conv2d(hidden_dim // 2, 1, 3, 1, 1)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, coarse: torch.Tensor, style_tokens: torch.Tensor) -> torch.Tensor:
        x = self.input(coarse)
        middle = self.body[0](x)
        gamma, beta = self.film(style_tokens.mean(1)).chunk(2, dim=-1)
        middle = middle * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        middle = self.body[1](middle)
        x = self.body[2](middle)
        return (coarse + self.scale.tanh() * self.output(x)).clamp(-1, 1)


class CharWriteViTGenerator(nn.Module):
    def __init__(self, vocab_size: int, style_dim=256, hidden_dim=256, context_layers=3,
                 context_heads=4, dropout=0.1, char_width=16, pad_id=0, unifont_table: torch.Tensor | None = None,
                 multiscale_depth=3, feature_noise=True, feature_noise_std=1.0,
                 use_contextualizer=True, use_merger=False):
        super().__init__()
        self.pad_id, self.char_width = pad_id, char_width
        self.use_contextualizer, self.use_merger = use_contextualizer, use_merger
        self.glyph_style_encoder = StyleEncoder(style_dim)
        self.merge_style_encoder = StyleEncoder(style_dim) if use_merger else None
        self.glyph_generator = GlyphGenerator(vocab_size, style_dim, hidden_dim, char_width, unifont_table, depth=5, heads=context_heads, dropout=dropout)
        self.contextualizer = CharacterContextualizer(vocab_size, hidden_dim, context_layers, context_heads, dropout, char_width)
        self.context_adapter = nn.Linear(hidden_dim, hidden_dim); self.context_gate = nn.Parameter(torch.tensor(1.0)); self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.glyph_fusion = WriteViTGlyphFusion(hidden_dim, multiscale_depth, context_heads, dropout, feature_noise, feature_noise_std)
        self.merger = WordMerger(style_dim, hidden_dim) if use_merger else None

    def forward(self, ids: torch.Tensor, references: torch.Tensor, noise_seed: int | None = None) -> dict[str, torch.Tensor]:
        padding_mask = ids.eq(self.pad_id)
        glyph_style = self.glyph_style_encoder(references)
        merge_style = self.merge_style_encoder(references) if self.merge_style_encoder is not None else None
        raw, content_tokens, styled_tokens = self.glyph_generator(ids, glyph_style)
        if self.use_contextualizer:
            contextual, context_tokens = self.contextualizer(raw, ids, padding_mask)
            main_tokens = self.fusion_norm(styled_tokens + self.context_gate * self.context_adapter(context_tokens))
        else:
            contextual, context_tokens, main_tokens = raw, styled_tokens, styled_tokens
        valid = (~padding_mask)[:, :, None, None, None]
        raw = torch.where(valid, raw, torch.ones_like(raw))
        contextual = torch.where(valid, contextual, torch.ones_like(contextual))
        token_valid = (~padding_mask)[:, :, None]
        content_tokens = torch.where(token_valid, content_tokens, torch.zeros_like(content_tokens))
        context_tokens = torch.where(token_valid, context_tokens, torch.zeros_like(context_tokens))
        main_tokens = torch.where(token_valid, main_tokens, torch.zeros_like(main_tokens))
        coarse, scales = self.glyph_fusion(main_tokens, content_tokens, noise_seed)
        pixel_valid = (~padding_mask).repeat_interleave(self.char_width, dim=1)[:, None, None, :]
        coarse = torch.where(pixel_valid, coarse, torch.ones_like(coarse))
        final = self.merger(coarse, merge_style) if self.merger is not None else coarse
        final = torch.where(pixel_valid, final, torch.ones_like(final))
        return {"raw_glyphs": raw, "contextual_glyphs": contextual, "coarse_word": coarse, "word": final,
                "glyph_style": glyph_style,
                "content_tokens": content_tokens, "context_tokens": context_tokens, **scales}

