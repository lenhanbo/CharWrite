from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch

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
