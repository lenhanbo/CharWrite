from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps


def load_grayscale(path: str | Path) -> Image.Image:
    return Image.open(path).convert("L")


def canonicalize_polarity(image: Image.Image) -> Image.Image:
    """Return black ink on a white background regardless of source polarity.

    EMNIST/MNIST commonly stores bright ink on a dark background, while IAM
    and our font renderer use dark ink on a bright background.  The median of
    the outer border is a robust estimate of the background for both sources.
    """
    grayscale=image.convert("L")
    array=np.asarray(grayscale,dtype=np.uint8)
    if array.size==0:
        return grayscale
    border=np.concatenate((array[0,:],array[-1,:],array[:,0],array[:,-1]))
    return ImageOps.invert(grayscale) if float(np.median(border))<127.5 else grayscale


def canonical_word(image: Image.Image, length: int, height: int = 32, char_width: int = 16) -> torch.Tensor:
    width = max(char_width, char_width * length)
    fitted = ImageOps.contain(canonicalize_polarity(image), (width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (width, height), 255)
    canvas.paste(fitted, ((width - fitted.width) // 2, (height - fitted.height) // 2))
    array = np.asarray(canvas, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).unsqueeze(0)


def pil_to_tensor(image: Image.Image, size: tuple[int, int]) -> torch.Tensor:
    canvas = ImageOps.fit(canonicalize_polarity(image), size, method=Image.Resampling.LANCZOS)
    array = np.asarray(canvas, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).unsqueeze(0)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    array = ((tensor.detach().cpu().squeeze().clamp(-1, 1) + 1) * 127.5).byte().numpy()
    return Image.fromarray(array, mode="L")

