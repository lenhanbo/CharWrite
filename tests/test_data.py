import json
from pathlib import Path

from charwritevit.data.iam import build_writer_splits, parse_forms, parse_manifest
from charwritevit.data.vocab import CharacterVocabulary


def test_forms_manifest_and_writer_disjoint_split(tmp_path: Path):
    (tmp_path / "forms.txt").write_text("# header\na01-000u 000 1\na01-001u 001 1\n", encoding="utf-8")
    (tmp_path / "words").mkdir()
    manifest = tmp_path / "gt.txt"
    manifest.write_text("words/a01-000u-00-00.png\tA\nwords/a01-001u-00-00.png\tB\n", encoding="utf-8")
    mapping = parse_forms(tmp_path / "forms.txt")
    rows = parse_manifest(manifest, tmp_path, mapping)
    assert [row["writer_id"] for row in rows] == ["000", "001"]
    splits = build_writer_splits(rows, (0.5, 0.0, 0.5), seed=3)
    assert {x["writer_id"] for x in splits["train"]}.isdisjoint({x["writer_id"] for x in splits["test"]})


def test_vocabulary_roundtrip(tmp_path: Path):
    vocab = CharacterVocabulary("aA!")
    assert vocab.decode(vocab.encode("aA!")) == "aA!"
    vocab.save(tmp_path / "vocab.json")
    assert CharacterVocabulary.load(tmp_path / "vocab.json").tokens == vocab.tokens

