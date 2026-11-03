import numpy as np
import torch
from PIL import Image

from charwritevit.data.image import canonicalize_polarity, pil_to_tensor


def _glyph(background: int, ink: int) -> Image.Image:
    array=np.full((32,16),background,dtype=np.uint8)
    array[8:24,6:10]=ink
    return Image.fromarray(array,mode="L")


def test_opposite_source_polarities_become_black_ink_on_white():
    iam_like=_glyph(255,0)
    emnist_like=_glyph(0,255)
    normalized_iam=np.asarray(canonicalize_polarity(iam_like))
    normalized_emnist=np.asarray(canonicalize_polarity(emnist_like))
    assert normalized_iam[0,0]==255 and normalized_iam[16,8]==0
    assert np.array_equal(normalized_iam,normalized_emnist)


def test_tensor_contract_is_white_positive_and_ink_negative():
    iam_tensor=pil_to_tensor(_glyph(255,0),(16,32))
    emnist_tensor=pil_to_tensor(_glyph(0,255),(16,32))
    assert iam_tensor[0,0,0]>0.99
    assert iam_tensor[0,16,8]<-0.99
    assert torch.allclose(iam_tensor,emnist_tensor)


def test_square_character_is_contained_and_padded_instead_of_cropped():
    array=np.full((28,28),255,dtype=np.uint8)
    array[6:22,1:4]=0
    array[6:22,24:27]=0
    tensor=pil_to_tensor(Image.fromarray(array,mode="L"),(16,32))
    assert tensor[0,:6].min()>0.9
    assert tensor[0,26:].min()>0.9
    assert tensor[0,10:22,:3].min()<0
    assert tensor[0,10:22,-3:].min()<0
