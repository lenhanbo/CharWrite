from pathlib import Path

import pytest
from PIL import ImageFont

from charwritevit.data.synthetic import SyntheticCharacterDataset, discover_fonts, render_character, render_word
from charwritevit.data.vocab import CharacterVocabulary


def test_rendering_contract():
    font = ImageFont.load_default()
    assert render_character("A", font).size == (16, 32)
    assert render_word("Ab!", font).size == (48, 32)


def test_optional_font_discovery_can_return_empty(tmp_path):
    assert discover_fonts(tmp_path,allow_empty=True)==[]
    with pytest.raises(FileNotFoundError):
        discover_fonts(tmp_path)


def test_synthetic_character_dataset_can_target_punctuation(tmp_path):
    font=tmp_path/"font.ttf"; font.touch()
    dataset=SyntheticCharacterDataset([font],CharacterVocabulary("A!?"),samples=7,characters="!?")
    assert dataset.characters==["!","?"]
    assert len(dataset)==7

