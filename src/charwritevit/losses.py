from __future__ import annotations

import torch
import torch.nn.functional as F


def edge_map(image: torch.Tensor) -> torch.Tensor:
    dx = image[..., :, 1:] - image[..., :, :-1]
    dy = image[..., 1:, :] - image[..., :-1, :]
    return F.pad(dx.abs(), (0, 1, 0, 0)) + F.pad(dy.abs(), (0, 0, 0, 1))


def hinge_discriminator(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    return F.relu(1 - real).mean() + F.relu(1 + fake).mean()


def hinge_generator(fake: torch.Tensor) -> torch.Tensor:
    return -fake.mean()


def glyph_classification_loss(classifier, glyphs: torch.Tensor, ids: torch.Tensor, pad_id=0) -> torch.Tensor:
    mask = ids.ne(pad_id)
    selected = glyphs[mask]
    if selected.numel() == 0:
        return glyphs.sum() * 0
    return F.cross_entropy(classifier(selected), ids[mask])


def reconstruction_loss(prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ink_weight = 1 + 3 * ((target + 1) * 0.5 < 0.8).float()
    pixel = (ink_weight * (prediction - target).abs()).mean()
    edge = F.l1_loss(edge_map(prediction), edge_map(target))
    return pixel, edge

