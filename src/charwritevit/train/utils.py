from __future__ import annotations

import contextlib
import math
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
        scale_before=scaler.get_scale()
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        return scaler.get_scale() >= scale_before
    return False


def make_warmup_cosine_scheduler(optimizer, total_steps: int, warmup_steps: int = 0, min_lr_ratio: float = 0.01):
    if total_steps < 1: raise ValueError("total_steps must be positive")
    if not 0 <= warmup_steps < total_steps: raise ValueError("warmup_steps must be in [0, total_steps)")
    if not 0 <= min_lr_ratio <= 1: raise ValueError("min_lr_ratio must be in [0, 1]")

    def multiplier(current_step: int) -> float:
        if warmup_steps and current_step < warmup_steps:
            return (current_step + 1) / warmup_steps
        progress=(current_step-warmup_steps)/max(1,total_steps-warmup_steps)
        progress=min(1.0,max(0.0,progress))
        cosine=0.5*(1.0+math.cos(math.pi*progress))
        return min_lr_ratio+(1.0-min_lr_ratio)*cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer,multiplier)

