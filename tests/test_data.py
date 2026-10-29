import json
from pathlib import Path

from PIL import Image

from charwritevit.data.iam import build_writer_splits, load_writer_mapping, parse_forms, parse_forms_by_writer, parse_iam_words, parse_manifest
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


def test_official_words_and_xml_recover_writer_metadata(tmp_path: Path):
    (tmp_path/"xml").mkdir(); (tmp_path/"xml"/"a01-000u.xml").write_text(
        '<form id="a01-000u" writer-id="007"></form>',encoding="utf-8")
    image_dir=tmp_path/"words"/"a01"/"a01-000u"; image_dir.mkdir(parents=True)
    Image.new("L",(20,10),255).save(image_dir/"a01-000u-00-00.png")
    words=tmp_path/"words.txt"; words.write_text(
        "# header\na01-000u-00-00 ok 154 1 2 20 10 NN hello\n",encoding="utf-8")
    mapping=load_writer_mapping(tmp_path); rows=parse_iam_words(words,tmp_path,mapping)
    assert mapping=={"a01-000u":"007"}
    assert rows[0]["writer_id"]=="007" and rows[0]["text"]=="hello"
    assert Path(rows[0]["path"]).is_file()


def test_writer_folder_form_tree_recovers_mapping(tmp_path: Path):
    form_dir=tmp_path/"forms"/"007"; form_dir.mkdir(parents=True)
    Image.new("L",(20,10),255).save(form_dir/"a01-000u.png")
    assert parse_forms_by_writer(tmp_path/"forms")=={"a01-000u":"007"}


def test_folder_can_be_used_as_explicit_pseudo_writer(tmp_path: Path):
    image_dir=tmp_path/"words"/"a01"/"a01-000u"; image_dir.mkdir(parents=True)
    Image.new("L",(20,10),255).save(image_dir/"a01-000u-00-00.png")
    words=tmp_path/"words.txt"; words.write_text(
        "a01-000u-00-00 ok 154 1 2 20 10 NN hello\n",encoding="utf-8")
    mapping=load_writer_mapping(tmp_path,allow_folder_as_writer=True)
    rows=parse_iam_words(words,tmp_path,mapping,folder_as_writer=True)
    assert mapping=={} and rows[0]["writer_id"]=="a01-000u"

