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
