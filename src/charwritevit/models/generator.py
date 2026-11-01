from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .common import (
    CharacterStyleBlock,
    Conv2dBlock,
    CrossBlock,
    PosCNN,
    PositionalEncoding,
    ResidualBlock,
    StyleEncoder,
    TransformerBlock,
)


class CharacterStylePath(nn.Module):
    """Refine GNU tokens before WriteViT, with a dedicated character-style path.

    Positional self-attention lets characters see one another. Cross-attention
    injects style from a separate Writer encoder. The resulting token replaces
    the GNU token everywhere it entered the original WriteViT generator.
    """

    def __init__(self, dimension=256, layers=3, heads=4, dropout=0.0, char_width=16):
        super().__init__()
        self.char_width = char_width
        self.position = PositionalEncoding(dimension, dropout, max_length=100)
        self.context_blocks = nn.ModuleList([
            TransformerBlock(dimension, heads, 4.0, dropout, qkv_bias=True)
            for _ in range(layers)
        ])
        self.style_blocks = nn.ModuleList([
            CharacterStyleBlock(dimension, heads, 4.0, dropout)
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(dimension)
        self.glyph_decoder = nn.Sequential(
            nn.Linear(dimension, dimension * 2),
            nn.GELU(),
            nn.Linear(dimension * 2, 32 * char_width),
            nn.Tanh(),
        )

    def forward(self, gnu_tokens: torch.Tensor, style_tokens: torch.Tensor,
                padding_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        refined = self.position(gnu_tokens)
        for context_block, style_block in zip(self.context_blocks, self.style_blocks):
            # Masking is intentionally confined to the added character path.
            # The released WriteViT CrossBlocks below remain mask-free.
            refined = context_block(refined, padding_mask=padding_mask)
            refined = style_block(refined, style_tokens)
            refined = refined.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        refined = self.norm(refined)
        glyphs = self.glyph_decoder(refined).view(
            refined.shape[0], refined.shape[1], 1, 32, self.char_width
        )
        return refined, glyphs


class WriteViTGlyphFusion(nn.Module):
    """Released WriteViT generator after its GNU query embedding."""

    def __init__(self, hidden_dim=256, depth=3, heads=4, dropout=0.0,
                 feature_noise=True, noise_std=1.0):
        super().__init__()
        self.dims = [hidden_dim, hidden_dim // 2, hidden_dim // 2, hidden_dim // 4]
        self.feature_noise = feature_noise
        self.noise_std = noise_std
        self.query_position = PositionalEncoding(hidden_dim, dropout, max_length=100)

        # Original WriteViT has depth + 2 (=5) cross blocks before the 2xL grid.
        self.cross_blocks = nn.ModuleList([
            CrossBlock(hidden_dim, heads, 4.0, dropout, qkv_bias=True)
            for _ in range(depth + 2)
        ])
        self.layer_norm2 = nn.LayerNorm(self.dims[0])
        self.to_rgb1 = nn.Conv2d(self.dims[0], self.dims[3], 3, 1, 1)

        self.up1 = self._upsample(self.dims[0], self.dims[1])
        self.stage2 = nn.ModuleList([
            TransformerBlock(self.dims[1], heads, 4.0, dropout, qkv_bias=True)
            for _ in range(depth)
        ])
        self.norm2 = nn.LayerNorm(self.dims[1])
        self.cpe2 = PosCNN(self.dims[1], self.dims[1])
        self.to_rgb2 = nn.Conv2d(self.dims[1], self.dims[3], 3, 1, 1)

        self.up2 = self._upsample(self.dims[1], self.dims[2])
        self.stage3 = nn.ModuleList([
            TransformerBlock(self.dims[2], heads, 4.0, dropout, qkv_bias=True)
            for _ in range(depth)
        ])
        self.norm3 = nn.LayerNorm(self.dims[2])
        self.cpe3 = PosCNN(self.dims[2], self.dims[2])
        self.to_rgb3 = nn.Conv2d(self.dims[2], self.dims[3], 3, 1, 1)

        self.up3 = self._upsample(self.dims[2], self.dims[3])
        self.stage4 = nn.ModuleList([
            TransformerBlock(self.dims[3], heads, 4.0, dropout, qkv_bias=True)
            for _ in range(depth)
        ])
        self.norm4 = nn.LayerNorm(self.dims[3])
        self.cpe4 = PosCNN(self.dims[3], self.dims[3])

        self.decoder = nn.Sequential(
            ResidualBlock(self.dims[3]),
            ResidualBlock(self.dims[3]),
            nn.Upsample(scale_factor=2),
            Conv2dBlock(self.dims[3], self.dims[3] // 2, 3, 1, 1, "in", "none", "reflect"),
            Conv2dBlock(self.dims[3] // 2, self.dims[3] // 2, 5, 1, 2, "in", "relu", "reflect"),
            Conv2dBlock(self.dims[3] // 2, 1, 7, 1, 3, "none", "tanh", "reflect"),
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @staticmethod
    def _upsample(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv2dBlock(in_channels, out_channels, 3, 1, 1, "in", "none", "reflect"),
            Conv2dBlock(out_channels, out_channels, 3, 1, 1, "in", "relu", "reflect"),
        )

    @staticmethod
    def stack_feature_rows(styled_tokens: torch.Tensor, content_tokens: torch.Tensor) -> torch.Tensor:
        if styled_tokens.shape != content_tokens.shape:
            raise ValueError("styled and content token shapes must match")
        batch, length, dimension = styled_tokens.shape
        return torch.cat([styled_tokens, content_tokens], 1).permute(0, 2, 1).reshape(
            batch, dimension, 2, length
        )

    @staticmethod
    def _run_stage(feature: torch.Tensor, blocks: nn.ModuleList,
                   cpe: PosCNN, norm: nn.LayerNorm) -> torch.Tensor:
        batch, channels, height, width = feature.shape
        tokens = feature.flatten(2).transpose(1, 2)
        for index, block in enumerate(blocks):
            tokens = block(tokens)
            if index == 0:
                tokens = cpe(tokens, height, width)
        return norm(tokens).transpose(1, 2).reshape(batch, channels, height, width)

    def forward(self, writer_tokens: torch.Tensor, refined_queries: torch.Tensor,
                noise_seed: int | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # `refined_queries` replaces query_embed(tgt) in both original branches.
        content_row = refined_queries
        styled = self.query_position(refined_queries)
        for block in self.cross_blocks:
            styled = block(styled, writer_tokens)

        scale1 = self.stack_feature_rows(styled, content_row)
        scale1 = self.layer_norm2(scale1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        rgb1 = self.to_rgb1(scale1)

        scale2 = self._run_stage(self.up1(scale1), self.stage2, self.cpe2, self.norm2)
        rgb2 = self.to_rgb2(scale2)
        scale3 = self._run_stage(self.up2(scale2), self.stage3, self.cpe3, self.norm3)
        rgb3 = self.to_rgb3(scale3)
        scale4 = self._run_stage(self.up3(scale3), self.stage4, self.cpe4, self.norm4)

        fused = (
            F.interpolate(rgb1, size=scale4.shape[-2:], mode="nearest")
            + F.interpolate(rgb2, size=scale4.shape[-2:], mode="nearest")
            + F.interpolate(rgb3, size=scale4.shape[-2:], mode="nearest")
            + scale4
        )
        if self.feature_noise:
            if noise_seed is None:
                noise = torch.randn_like(fused)
            else:
                generator = torch.Generator(device=fused.device).manual_seed(noise_seed)
                noise = torch.randn(fused.shape, generator=generator, device=fused.device, dtype=fused.dtype)
            fused = fused + self.noise_std * noise
        word = self.decoder(fused)
        return word, {
            "scale1": scale1,
            "scale2": scale2,
            "scale3": scale3,
            "scale4": scale4,
            "to_rgb1": rgb1,
            "to_rgb2": rgb2,
            "to_rgb3": rgb3,
            "fused": fused,
        }


class CharWriteViTGenerator(nn.Module):
    def __init__(self, vocab_size: int, style_dim=256, hidden_dim=256, character_style_layers=3,
                 character_style_heads=4, dropout=0.0, char_width=16, pad_id=0,
                 unifont_table: torch.Tensor | None = None, multiscale_depth=3,
                 feature_noise=True, feature_noise_std=1.0):
        super().__init__()
        if style_dim != hidden_dim:
            raise ValueError("WriteViT requires style_dim == hidden_dim")
        self.pad_id = pad_id
        self.char_width = char_width

        if unifont_table is None:
            self.unifont = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)
            self.unifont_projection = nn.Identity()
            self.register_buffer("unifont_pixels", torch.zeros(vocab_size, 256), persistent=False)
        else:
            self.unifont = nn.Embedding.from_pretrained(
                unifont_table.float(), freeze=True, padding_idx=pad_id
            )
            self.unifont_projection = nn.Linear(unifont_table.shape[1], hidden_dim)
            self.register_buffer("unifont_pixels", unifont_table.float().clone(), persistent=False)

        # Deliberately independent style paths, as required by the ablation.
        self.character_style_encoder = StyleEncoder(style_dim, 3, 4, dropout)
        self.writevit_writer_encoder = StyleEncoder(style_dim, 3, 4, dropout)
        self.character_style_path = CharacterStylePath(
            hidden_dim, character_style_layers, character_style_heads, dropout, char_width
        )
        self.glyph_fusion = WriteViTGlyphFusion(
            hidden_dim, multiscale_depth, 4, dropout,
            feature_noise, feature_noise_std,
        )

    def _gnu_glyphs(self, ids: torch.Tensor) -> torch.Tensor:
        pixels = self.unifont_pixels[ids]
        glyphs = 1.0 - 2.0 * pixels.view(ids.shape[0], ids.shape[1], 1, 16, 16)
        return F.interpolate(
            glyphs.flatten(0, 1), size=(32, self.char_width), mode="nearest"
        ).view(ids.shape[0], ids.shape[1], 1, 32, self.char_width)

    def forward(self, ids: torch.Tensor, references: torch.Tensor,
                noise_seed: int | None = None) -> dict[str, torch.Tensor]:
        padding = ids.eq(self.pad_id)
        gnu_tokens = self.unifont_projection(self.unifont(ids))

        character_style = self.character_style_encoder(references)
        refined_tokens, styled_glyphs = self.character_style_path(gnu_tokens, character_style, padding)

        # A second, independent Writer path is used by the untouched WriteViT GAN.
        writer_tokens = self.writevit_writer_encoder(references)
        word, scales = self.glyph_fusion(writer_tokens, refined_tokens, noise_seed)

        valid_glyph = (~padding)[:, :, None, None, None]
        gnu_glyphs = torch.where(valid_glyph, self._gnu_glyphs(ids), torch.ones_like(styled_glyphs))
        styled_glyphs = torch.where(valid_glyph, styled_glyphs, torch.ones_like(styled_glyphs))
        valid_word = (~padding).repeat_interleave(self.char_width, 1)[:, None, None, :]
        word = torch.where(valid_word, word, torch.ones_like(word))

        return {
            "gnu_glyphs": gnu_glyphs,
            "styled_glyphs": styled_glyphs,
            "coarse_word": word,
            "word": word,
            "gnu_tokens": gnu_tokens,
            "refined_tokens": refined_tokens,
            "character_style": character_style,
            "writer_style": writer_tokens,
            **scales,
        }
