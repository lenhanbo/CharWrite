from __future__ import annotations

import json
from pathlib import Path


class CharacterVocabulary:
    PAD = "<pad>"
    UNK = "<unk>"
    BLANK = "<blank>"

    def __init__(self, characters: str | list[str]):
        chars = list(dict.fromkeys(characters))
        self.tokens = [self.PAD, self.UNK, self.BLANK, *chars]
        self.stoi = {token: idx for idx, token in enumerate(self.tokens)}

    @property
    def pad_id(self) -> int:
        return self.stoi[self.PAD]

    @property
    def blank_id(self) -> int:
        return self.stoi[self.BLANK]

    def encode(self, text: str) -> list[int]:
        unk = self.stoi[self.UNK]
        return [self.stoi.get(char, unk) for char in text]

    def decode(self, ids: list[int]) -> str:
        special = {self.PAD, self.UNK, self.BLANK}
        return "".join(self.tokens[idx] for idx in ids if self.tokens[idx] not in special)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.tokens, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharacterVocabulary":
        tokens = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(tokens[3:])

    def __len__(self) -> int:
        return len(self.tokens)

