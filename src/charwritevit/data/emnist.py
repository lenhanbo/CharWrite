from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .image import pil_to_tensor
from .vocab import CharacterVocabulary


class EMNISTCharacters(Dataset):
    """EMNIST ByClass with its storage transpose corrected and labels mapped to our vocabulary."""

    def __init__(self, root: str, vocabulary: CharacterVocabulary, train=True, download=False):
        from torchvision.datasets import EMNIST
        self.base = EMNIST(root=root, split="byclass", train=train, download=download)
        self.vocabulary = vocabulary
        self.indices = []
        for index, target in enumerate(self.base.targets.tolist()):
            char = str(self.base.classes[target])
            if len(char) == 1 and char in vocabulary.stoi:
                self.indices.append(index)

    def __len__(self): return len(self.indices)

    def __getitem__(self, index):
        image, target = self.base[self.indices[index]]
        image = image.transpose(Image.Transpose.TRANSPOSE)
        char = str(self.base.classes[target])
        return pil_to_tensor(image, (16, 32)), self.vocabulary.stoi[char]


class KaggleEMNISTCSV(Dataset):
    """Random-access adapter for crawford/emnist without loading its CSV into RAM."""

    def __init__(self, csv_path: str | Path, mapping_path: str | Path, vocabulary: CharacterVocabulary, cache_dir: str | Path):
        self.csv_path=Path(csv_path); self.vocabulary=vocabulary; self._handle=None
        mapping={}
        for line in Path(mapping_path).read_text(encoding="utf-8").splitlines():
            fields=line.split()
            if len(fields)>=2: mapping[int(fields[0])]=chr(int(fields[1]))
        self.mapping=mapping
        cache=Path(cache_dir); cache.mkdir(parents=True,exist_ok=True); offset_path=cache/f"{self.csv_path.name}.offsets.npy"
        if offset_path.is_file():
            self.offsets=np.load(offset_path,mmap_mode="r")
        else:
            offsets=[]
            with self.csv_path.open("rb") as handle:
                position=handle.tell(); line=handle.readline()
                while line:
                    try: int(line.split(b",",1)[0])
                    except ValueError: pass
                    else: offsets.append(position)
                    position=handle.tell(); line=handle.readline()
            self.offsets=np.asarray(offsets,dtype=np.int64); np.save(offset_path,self.offsets)

    def __len__(self): return len(self.offsets)

    def __getstate__(self):
        state=self.__dict__.copy(); state["_handle"]=None; return state

    def __getitem__(self,index):
        if self._handle is None: self._handle=self.csv_path.open("rb")
        self._handle.seek(int(self.offsets[index])); values=np.fromstring(self._handle.readline().decode("ascii"),dtype=np.uint8,sep=",")
        label=int(values[0]); char=self.mapping[label]
        image=Image.fromarray(values[1:].reshape(28,28).T,mode="L")
        return pil_to_tensor(image,(16,32)),self.vocabulary.stoi[char]


def build_emnist_dataset(root: str | Path, vocabulary: CharacterVocabulary, train=True, download=False, cache_dir: str | Path | None = None) -> Dataset:
    base=Path(root); split="train" if train else "test"
    csv_files=list(base.rglob(f"emnist-byclass-{split}.csv")) if base.exists() else []
    mapping_files=list(base.rglob("emnist-byclass-mapping.txt")) if base.exists() else []
    if csv_files and mapping_files:
        cache_dir=cache_dir or Path(".cache/emnist-csv")
        return KaggleEMNISTCSV(csv_files[0],mapping_files[0],vocabulary,cache_dir)
    return EMNISTCharacters(str(base),vocabulary,train=train,download=download)

