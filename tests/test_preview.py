from pathlib import Path

import torch

from charwritevit.train.preview import MetricHistory, save_classifier_preview


def test_classifier_history_and_preview(tmp_path: Path):
    history = MetricHistory(tmp_path)
    history.append({"epoch": 0, "train_loss": 1.0, "val_loss": 1.2})
    history.append({"epoch": 1, "train_loss": 0.8, "val_loss": 0.9})
    history.plot("curves.png")
    images = torch.randn(4, 1, 32, 16)
    labels = torch.tensor([0, 1, 2, 1])
    logits = torch.randn(4, 3)
    save_classifier_preview(images, labels, logits, ["a", "b", "c"], tmp_path / "preview.png")
    assert (tmp_path / "metrics.jsonl").is_file()
    assert (tmp_path / "curves.png").is_file()
    assert (tmp_path / "preview.png").is_file()
