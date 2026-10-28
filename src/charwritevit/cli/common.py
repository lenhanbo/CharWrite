from __future__ import annotations

import argparse
from pathlib import Path

import torch

from charwritevit.config import load_config
from charwritevit.data.vocab import CharacterVocabulary
from charwritevit.data.unifont import load_unifont_table
from charwritevit.models.generator import CharWriteViTGenerator


def training_parser(description: str):
    parser=argparse.ArgumentParser(description=description); parser.add_argument("--config",required=True); parser.add_argument("--output",required=True); parser.add_argument("--resume")
    parser.add_argument("--epochs",type=int); parser.add_argument("--max-steps",type=int)
    return parser


def build_generator(cfg, vocab: CharacterVocabulary):
    model=cfg["model"]; data=cfg["data"]
    unifont_path=model.get("unifont_path"); table=load_unifont_table(unifont_path,vocab) if unifont_path and Path(unifont_path).is_file() else None
    return CharWriteViTGenerator(len(vocab), int(model.get("style_dim",128)), int(model.get("hidden_dim",128)),
        int(model.get("context_layers",3)), int(model.get("context_heads",4)), float(model.get("dropout",.1)),
        int(data.get("char_width",16)), vocab.pad_id, table,
        int(model.get("multiscale_depth",3)), bool(model.get("feature_noise",True)), float(model.get("feature_noise_std",1.0)),
        bool(model.get("use_contextualizer",True)), bool(model.get("use_merger",True)))


def load_vocab(cfg):
    path=Path(cfg["data"]["prepared_dir"])/"vocab.json"
    return CharacterVocabulary.load(path) if path.is_file() else CharacterVocabulary(cfg["model"]["alphabet"])

