import torch

from charwritevit.data.unifont import UnifontCharacterDataset
from charwritevit.data.vocab import CharacterVocabulary


def test_unifont_classifier_anchor_has_expected_shape_labels_and_repeats():
    vocab=CharacterVocabulary("A ")
    table=torch.zeros(len(vocab),256)
    table[vocab.stoi["A"],5*16+7]=1
    dataset=UnifontCharacterDataset(table,vocab,repeats=2,char_width=16)
    assert len(dataset)==4
    image,label=dataset[0]
    assert image.shape==(1,32,16)
    assert int(label)==vocab.stoi["A"]
    assert image.min()==-1 and image.max()==1
    assert int(dataset[2][1])==vocab.stoi["A"]
