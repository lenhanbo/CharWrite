from __future__ import annotations

import json
import random
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path
import torch
from torch.utils.data import Dataset

from .image import canonical_word
from .vocab import CharacterVocabulary


def parse_forms(path: str | Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            raise ValueError(f"Malformed forms.txt row: {raw!r}")
        mapping[fields[0]] = fields[1]
    if not mapping:
        raise ValueError("forms.txt contains no form-to-writer mappings")
    return mapping


def find_forms_file(root: str | Path) -> Path:
    base = Path(root)
    candidates = [base / "ascii" / "forms.txt", base / "forms.txt", base.parent / "ascii" / "forms.txt"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("IAM forms.txt was not found")


def parse_xml_writer_mapping(root: str | Path) -> dict[str,str]:
    """Recover form-to-writer metadata from official IAM form XML files."""
    mapping={}
    for path in Path(root).rglob("*.xml"):
        try: xml_root=ET.parse(path).getroot()
        except ET.ParseError: continue
        for element in xml_root.iter():
            if element.tag.rsplit("}",1)[-1] != "form": continue
            form_id=element.attrib.get("id") or element.attrib.get("form-id")
            writer=element.attrib.get("writer-id") or element.attrib.get("writer_id") or element.attrib.get("writer")
            if form_id and writer: mapping[str(form_id)]=str(writer)
    if not mapping:
        raise FileNotFoundError("IAM writer metadata is missing: provide forms.txt or form XML files containing id and writer-id")
    return mapping


def parse_forms_by_writer(root: str | Path) -> dict[str,str]:
    """Recover form-to-writer mapping from writer_folder/form_id.png trees or ZIPs."""
    base=Path(root); mapping={}

    def add(parts: tuple[str,...]) -> None:
        if len(parts)<2 or not parts[-1].lower().endswith(".png"): return
        form_id=Path(parts[-1]).stem; writer=parts[-2]
        previous=mapping.get(form_id)
        if previous is not None and previous!=writer: raise ValueError(f"Form {form_id} occurs under writers {previous} and {writer}")
        mapping[form_id]=writer

    if base.exists():
        for image in base.rglob("*.png"): add(tuple(image.relative_to(base).parts))
        for archive in base.rglob("*.zip"):
            with zipfile.ZipFile(archive) as handle:
                for name in handle.namelist(): add(tuple(Path(name).parts))
    if not mapping: raise FileNotFoundError(f"No writer_folder/form_id.png mapping found under {base}")
    return mapping


def load_writer_mapping(root: str | Path, forms_by_writer_root: str | Path | None = None,
                        allow_folder_as_writer: bool = False) -> dict[str,str]:
    try: return parse_forms(find_forms_file(root))
    except FileNotFoundError:
        try: return parse_xml_writer_mapping(root)
        except FileNotFoundError:
            if forms_by_writer_root:
                try: return parse_forms_by_writer(forms_by_writer_root)
                except FileNotFoundError: pass
            if allow_folder_as_writer: return {}
            raise FileNotFoundError("IAM writer metadata is missing: attach forms.txt/XML or a writer_folder/form_id.png dataset")


def find_words_file(root: str | Path) -> Path:
    base=Path(root); candidates=[base/"ascii"/"words.txt",base/"words.txt"]
    for candidate in candidates:
        if candidate.is_file(): return candidate
    matches=list(base.rglob("words.txt"))
    if matches: return matches[0]
    raise FileNotFoundError("Expected linux_gt.txt/train_gt.txt or official IAM ascii/words.txt")


def parse_iam_words(path: str | Path, root: str | Path, form_to_writer: dict[str,str],
                    folder_as_writer: bool = False) -> list[dict]:
    """Parse official IAM words.txt and resolve its standard word-image tree."""
    base=Path(root); word_roots=[]
    for candidate in [base/"words",base,*[p for p in base.rglob("words") if p.is_dir()]]:
        if candidate not in word_roots: word_roots.append(candidate)
    records=[]
    for raw in Path(path).read_text(encoding="utf-8",errors="replace").splitlines():
        line=raw.strip()
        if not line or line.startswith("#"): continue
        fields=line.split(maxsplit=8)
        if len(fields)<9: raise ValueError(f"Malformed IAM words.txt row: {raw!r}")
        word_id,status,text=fields[0],fields[1],fields[8]
        if status.lower()!="ok": continue
        form_id="-".join(word_id.split("-")[:2])
        relative=Path(word_id.split("-")[0])/form_id/f"{word_id}.png"
        candidates=[directory/relative for directory in word_roots]
        image_path=next((candidate for candidate in candidates if candidate.is_file()),candidates[0])
        writer=form_to_writer.get(form_id)
        if writer is None and folder_as_writer: writer=image_path.parent.name
        if writer is None: raise KeyError(f"No writer mapping for IAM form {form_id}")
        records.append({"path":str(image_path.resolve()),"text":text,"word_id":word_id,
                        "form_id":form_id,"writer_id":writer})
    return records


def parse_manifest(path: str | Path, root: str | Path, form_to_writer: dict[str, str],
                   folder_as_writer: bool = False) -> list[dict]:
    records: list[dict] = []
    base = Path(root)
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if "\t" not in raw:
            continue
        relative, text = raw.split("\t", 1)
        word_id = Path(relative).stem
        form_id = "-".join(word_id.split("-")[:2])
        image_path = base / Path(relative)
        writer=form_to_writer.get(form_id)
        if writer is None and folder_as_writer: writer=image_path.parent.name
        if writer is None: raise KeyError(f"No writer mapping for IAM form {form_id}")
        records.append({
            "path": str(image_path.resolve()), "text": text, "word_id": word_id,
            "form_id": form_id, "writer_id": writer,
        })
    return records


def build_writer_splits(records: list[dict], ratios=(0.8, 0.1, 0.1), seed: int = 42) -> dict[str, list[dict]]:
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to one")
    by_writer: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_writer[record["writer_id"]].append(record)
    writers = sorted(by_writer)
    random.Random(seed).shuffle(writers)
    n_train = round(len(writers) * ratios[0])
    n_val = round(len(writers) * ratios[1])
    groups = {"train": writers[:n_train], "val": writers[n_train:n_train+n_val], "test": writers[n_train+n_val:]}
    return {name: [item for writer in ids for item in by_writer[writer]] for name, ids in groups.items()}


def save_splits(splits: dict[str, list[dict]], output: str | Path) -> None:
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    for name, records in splits.items():
        (target / f"{name}.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    summary = {name: {"samples": len(rows), "writers": len({r["writer_id"] for r in rows})} for name, rows in splits.items()}
    (target / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


class IAMWordDataset(Dataset):
    def __init__(self, manifest: str | Path, vocabulary: CharacterVocabulary, height=32, char_width=16,
                 max_length=24, references=4):
        self.records = json.loads(Path(manifest).read_text(encoding="utf-8"))
        self.records = [r for r in self.records if 0 < len(r["text"]) <= max_length and Path(r["path"]).is_file()]
        self.vocabulary, self.height, self.char_width, self.references = vocabulary, height, char_width, references
        self.by_writer: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(self.records):
            self.by_writer[record["writer_id"]].append(index)
        self.records = [r for r in self.records if len(self.by_writer[r["writer_id"]]) > 1]
        self.by_writer.clear()
        for index, record in enumerate(self.records):
            self.by_writer[record["writer_id"]].append(index)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        target = canonical_word(ImageOpen(record["path"]), len(record["text"]), self.height, self.char_width)
        choices = [i for i in self.by_writer[record["writer_id"]] if i != index]
        different_form = [i for i in choices if self.records[i]["form_id"] != record["form_id"]]
        pool = different_form or choices
        rng = random.Random(index + random.randrange(1 << 20))
        selected = [rng.choice(pool) for _ in range(self.references)]
        refs = []
        for ref_index in selected:
            ref = self.records[ref_index]
            refs.append(canonical_word(ImageOpen(ref["path"]), min(len(ref["text"]), 24), self.height, self.char_width))
        return {"image": target, "text": record["text"], "ids": torch.tensor(self.vocabulary.encode(record["text"])),
                "references": refs, "writer_id": record["writer_id"], "word_id": record["word_id"]}


def ImageOpen(path: str):
    from PIL import Image
    with Image.open(path) as image:
        return image.convert("L").copy()


def collate_iam(batch: list[dict], pad_id: int = 0) -> dict:
    max_length = max(item["ids"].numel() for item in batch)
    ids = torch.full((len(batch), max_length), pad_id, dtype=torch.long)
    images = torch.ones((len(batch), 1, 32, max_length * 16))
    ref_width = max(ref.shape[-1] for item in batch for ref in item["references"])
    references = torch.ones((len(batch), len(batch[0]["references"]), 1, 32, ref_width))
    lengths = []
    for i, item in enumerate(batch):
        length = item["ids"].numel(); lengths.append(length)
        ids[i, :length] = item["ids"]
        images[i, :, :, :item["image"].shape[-1]] = item["image"]
        for j, ref in enumerate(item["references"]): references[i, j, :, :, :ref.shape[-1]] = ref
    return {"image": images, "ids": ids, "lengths": torch.tensor(lengths), "references": references,
            "text": [x["text"] for x in batch], "writer_id": [x["writer_id"] for x in batch]}

