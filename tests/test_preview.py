from pathlib import Path

import torch

from charwritevit.train.preview import MetricHistory, save_classifier_preview, save_generator_preview


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


def test_generator_preview_writes_characters_scales_and_final_word(tmp_path: Path):
    batch={
        "ids":torch.tensor([[0,1,2],[1,2,0]]),
        "lengths":torch.tensor([3,2]),
        "image":torch.randn(2,1,32,48).clamp(-1,1),
        "references":torch.randn(2,1,1,32,48).clamp(-1,1),
    }
    output={
        "gnu_glyphs":torch.randn(2,3,1,32,16).clamp(-1,1),
        "styled_glyphs":torch.randn(2,3,1,32,16).clamp(-1,1),
        "scale1":torch.randn(2,8,2,3),
        "scale2":torch.randn(2,4,4,6),
        "scale3":torch.randn(2,4,8,12),
        "scale4":torch.randn(2,2,16,24),
        "fused":torch.randn(2,2,16,24),
        "word":torch.randn(2,1,32,48).clamp(-1,1),
    }
    classifier=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(32*16,3))
    save_generator_preview(batch,output,classifier,["a","b","c"],tmp_path,limit=2)
    for sample in range(2):
        for suffix in ("characters","scales","word"):
            assert (tmp_path/f"sample_{sample:02d}_{suffix}.png").is_file()
