import torch

from charwritevit.cli.evaluate import _inception_images


def test_inception_images_pad_consistently_and_convert_to_rgb():
    words=torch.full((2,1,32,48),-1.0)
    converted=_inception_images(words,64)
    assert converted.shape == (2,3,32,64)
    assert converted.dtype == words.dtype
    assert torch.all(converted[...,:48] == 0)
    assert torch.all(converted[...,48:] == 1)


def test_inception_images_crop_to_fixed_width():
    words=torch.zeros(1,1,32,80)
    converted=_inception_images(words,64)
    assert converted.shape == (1,3,32,64)
    assert torch.all(converted == 0.5)
