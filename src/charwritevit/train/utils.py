from __future__ import annotations

import contextlib
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_context(device: torch.device, enabled=True):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled and device.type == "cuda")


def make_scaler(device: torch.device, enabled=True):
    return torch.amp.GradScaler(device.type, enabled=enabled and device.type == "cuda")


def save_checkpoint(path: str | Path, **state) -> None:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True); torch.save(state, target)


def load_checkpoint(path: str | Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device, weights_only=False)


def optimizer_step(loss, optimizer, scaler, accumulation_steps: int, step: int):
    scaler.scale(loss / accumulation_steps).backward()
    if (step + 1) % accumulation_steps == 0:
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)

