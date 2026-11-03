from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .vocab import CharacterVocabulary


def load_unifont_table(path: str | Path, vocabulary: CharacterVocabulary) -> torch.Tensor:
    """Build a vocabulary-aligned frozen glyph table from WriteViT's unifont asset."""
    with Path(path).open("rb") as handle:
        symbols = pickle.load(handle)
    by_codepoint = {int(item["idx"][0]): np.asarray(item["mat"], dtype=np.float32).reshape(-1) for item in symbols}
    sample = next(iter(by_codepoint.values()))
    rows = []
    for token in vocabulary.tokens:
        rows.append(by_codepoint.get(ord(token), np.zeros_like(sample)) if len(token) == 1 else np.zeros_like(sample))
    return torch.from_numpy(np.stack(rows))


class UnifontCharacterDataset(Dataset):
    """Vocabulary-aligned GNU glyphs used as a fixed classifier anchor."""

    def __init__(self, table: torch.Tensor, vocabulary: CharacterVocabulary,
                 repeats: int = 256, char_width: int = 16):
        if repeats < 1:
            raise ValueError("Unifont repeats must be positive")
        if table.shape != (len(vocabulary), 16 * 16):
            raise ValueError(f"Expected Unifont table [{len(vocabulary)},256], got {tuple(table.shape)}")
        self.table = table.float()
        self.labels = torch.arange(3, len(vocabulary), dtype=torch.long)
        self.repeats = repeats
        self.char_width = char_width

    def __len__(self) -> int:
        return self.labels.numel() * self.repeats

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        label = self.labels[index % self.labels.numel()]
        glyph = 1.0 - 2.0 * self.table[label].view(1, 16, 16)
        glyph = F.interpolate(
            glyph.unsqueeze(0), size=(32, self.char_width), mode="nearest"
        ).squeeze(0)
        return glyph, label
