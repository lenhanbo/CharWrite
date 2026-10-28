from __future__ import annotations

import random
import string
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset

from .image import pil_to_tensor
from .vocab import CharacterVocabulary


def font_license(font: Path, root: Path) -> Path | None:
    names = ("OFL.txt", "LICENSE.txt", "LICENSE", "UFL.txt")
    directory = font.parent
    while directory == root or root in directory.parents:
        for name in names:
            candidate = directory / name
            if candidate.is_file(): return candidate
        if directory == root: break
        directory = directory.parent
    return None


def discover_fonts(root: str | Path, require_license: bool = True, allow_empty: bool = False) -> list[Path]:
    base = Path(root)
    fonts = sorted([*base.rglob("*.ttf"), *base.rglob("*.otf")])
    if require_license:
        fonts = [font for font in fonts if font_license(font, base) is not None]
    if not fonts and not allow_empty:
        detail = " with an adjacent OFL/LICENSE file" if require_license else ""
        raise FileNotFoundError(f"No .ttf/.otf fonts{detail} found under {root}")
    return fonts


def render_character(char: str, font: ImageFont.FreeTypeFont, size=(16, 32), offset=(0, 0)) -> Image.Image:
    image = Image.new("L", size, 255)
    draw = ImageDraw.Draw(image)
    box = draw.textbbox((0, 0), char, font=font)
    x = (size[0] - (box[2] - box[0])) // 2 - box[0] + offset[0]
    y = (size[1] - (box[3] - box[1])) // 2 - box[1] + offset[1]
    draw.text((x, y), char, font=font, fill=0)
    return image


def render_word(text: str, font: ImageFont.FreeTypeFont, char_width=16, height=32) -> Image.Image:
    size = (char_width * len(text), height)
    image = Image.new("L", size, 255)
    draw = ImageDraw.Draw(image)
    box = draw.textbbox((0, 0), text, font=font)
    x = max(0, (size[0] - (box[2] - box[0])) // 2 - box[0])
    y = (height - (box[3] - box[1])) // 2 - box[1]
    draw.text((x, y), text, font=font, fill=0)
    return image


class SyntheticCharacterDataset(Dataset):
    def __init__(self, fonts: list[Path], vocabulary: CharacterVocabulary, samples=100_000, font_size=(20, 30), characters: str | list[str] | None = None):
        self.fonts, self.vocab, self.samples, self.font_size = fonts, vocabulary, samples, font_size
        requested=self.vocab.tokens[3:] if characters is None else list(dict.fromkeys(characters))
        self.characters=[char for char in requested if char in self.vocab.stoi]
        if not self.fonts: raise ValueError("SyntheticCharacterDataset requires at least one font")
        if not self.characters: raise ValueError("SyntheticCharacterDataset requires at least one vocabulary character")

    def __len__(self): return self.samples

    def __getitem__(self, index):
        rng = random.Random(index)
        char = rng.choice(self.characters); path = rng.choice(self.fonts)
        font = ImageFont.truetype(str(path), rng.randint(*self.font_size))
        image = render_character(char, font, offset=(rng.randint(-2, 2), rng.randint(-2, 2)))
        return pil_to_tensor(image, (16, 32)), self.vocab.stoi[char]


class SyntheticWordDataset(Dataset):
    def __init__(self, fonts: list[Path], vocabulary: CharacterVocabulary, samples=100_000, max_length=12):
        self.fonts, self.vocab, self.samples, self.max_length = fonts, vocabulary, samples, max_length
        self.characters = [c for c in self.vocab.tokens[3:] if c not in "#&()*+"]

    def __len__(self): return self.samples

    def __getitem__(self, index):
        rng = random.Random(index); length = rng.randint(1, self.max_length)
        text = "".join(rng.choice(self.characters) for _ in range(length))
        font_path = rng.choice(self.fonts); font = ImageFont.truetype(str(font_path), rng.randint(20, 30))
        glyphs = torch.stack([pil_to_tensor(render_character(c, font), (16, 32)) for c in text])
        word = pil_to_tensor(render_word(text, font), (16 * length, 32))
        references = torch.stack([pil_to_tensor(render_word("Style", font, 16, 32), (80, 32)) for _ in range(2)])
        return {"glyphs": glyphs, "word": word, "ids": torch.tensor(self.vocab.encode(text)), "references": references, "text": text}


def collate_synthetic(batch: list[dict], pad_id=0) -> dict:
    max_len = max(x["ids"].numel() for x in batch); b = len(batch)
    ids = torch.full((b, max_len), pad_id, dtype=torch.long)
    glyphs = torch.ones((b, max_len, 1, 32, 16)); words = torch.ones((b, 1, 32, max_len * 16))
    refs = torch.stack([x["references"] for x in batch]); lengths=[]
    for i, x in enumerate(batch):
        n=x["ids"].numel(); lengths.append(n); ids[i,:n]=x["ids"]; glyphs[i,:n]=x["glyphs"]; words[i,:,:,:n*16]=x["word"]
    return {"ids":ids,"glyphs":glyphs,"word":words,"references":refs,"lengths":torch.tensor(lengths),"text":[x["text"] for x in batch]}

