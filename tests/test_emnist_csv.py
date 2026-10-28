from pathlib import Path

from charwritevit.data.emnist import KaggleEMNISTCSV
from charwritevit.data.vocab import CharacterVocabulary


def test_kaggle_csv_adapter_builds_random_access_index(tmp_path: Path):
    mapping=tmp_path/"emnist-byclass-mapping.txt"; mapping.write_text("0 65\n",encoding="utf-8")
    pixels=[0]*784; pixels[3]=255
    csv=tmp_path/"emnist-byclass-train.csv"; csv.write_text("label,"+",".join(f"p{i}" for i in range(784))+"\n"+",".join(map(str,[0,*pixels]))+"\n",encoding="ascii")
    dataset=KaggleEMNISTCSV(csv,mapping,CharacterVocabulary("A"),tmp_path/"cache")
    image,label=dataset[0]
    assert len(dataset)==1
    assert image.shape==(1,32,16)
    assert label==3
