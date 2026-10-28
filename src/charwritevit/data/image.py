from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps


def load_grayscale(path: str | Path) -> Image.Image:
    return Image.open(path).convert("L")


def canonical_word(image: Image.Image, length: int, height: int = 32, char_width: int = 16) -> torch.Tensor:
    width = max(char_width, char_width * length)
    fitted = ImageOps.contain(image.convert("L"), (width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (width, height), 255)
    canvas.paste(fitted, ((width - fitted.width) // 2, (height - fitted.height) // 2))
    array = np.asarray(canvas, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).unsqueeze(0)


def pil_to_tensor(image: Image.Image, size: tuple[int, int]) -> torch.Tensor:
    canvas = ImageOps.fit(image.convert("L"), size, method=Image.Resampling.LANCZOS)
    array = np.asarray(canvas, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).unsqueeze(0)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    array = ((tensor.detach().cpu().squeeze().clamp(-1, 1) + 1) * 127.5).byte().numpy()
    return Image.fromarray(array, mode="L")

