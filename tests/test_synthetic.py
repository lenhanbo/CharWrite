from pathlib import Path

import pytest
from PIL import ImageFont

from charwritevit.data.synthetic import render_character, render_word


def test_rendering_contract():
    font = ImageFont.load_default()
    assert render_character("A", font).size == (16, 32)
    assert render_word("Ab!", font).size == (48, 32)

