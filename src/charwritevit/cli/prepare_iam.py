from __future__ import annotations

import argparse
from pathlib import Path

from charwritevit.config import load_config
from charwritevit.data.iam import build_writer_splits, find_forms_file, parse_forms, parse_manifest, save_splits
from charwritevit.data.vocab import CharacterVocabulary


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--manifest")
    args = parser.parse_args(); cfg = load_config(args.config); data = cfg["data"]
    root = Path(data["iam_root"]); forms = parse_forms(find_forms_file(root))
    manifest = Path(args.manifest) if args.manifest else root / "linux_gt.txt"
    if not manifest.is_file():
        manifest = root / "train_gt.txt"
    records = parse_manifest(manifest, root, forms)
    if manifest.name == "train_gt.txt" and (root / "val_gt.txt").is_file():
        records += parse_manifest(root / "val_gt.txt", root, forms)
        records = list({r["word_id"]: r for r in records}.values())
    allowed = set(cfg["model"]["alphabet"]); max_len = int(data.get("max_word_length", 24))
    records = [r for r in records if 0 < len(r["text"]) <= max_len and set(r["text"]) <= allowed and Path(r["path"]).is_file()]
    splits = build_writer_splits(records, tuple(data.get("split", [0.8, 0.1, 0.1])), int(cfg.get("seed", 42)))
    output = Path(data["prepared_dir"]); save_splits(splits, output)
    CharacterVocabulary(cfg["model"]["alphabet"]).save(output / "vocab.json")
    print((output / "summary.json").read_text())


if __name__ == "__main__": main()

