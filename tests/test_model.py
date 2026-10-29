import torch

from charwritevit.models.auxiliary import CharacterClassifier
from charwritevit.models.generator import CharWriteViTGenerator, WriteViTGlyphFusion
from charwritevit.losses import glyph_classification_loss


def tiny_model():
    return CharWriteViTGenerator(12, style_dim=32, hidden_dim=32, context_layers=1, context_heads=4, char_width=16, multiscale_depth=1, feature_noise=False)


def test_generator_shapes_and_single_concat():
    model = tiny_model()
    ids = torch.tensor([[3, 4, 5], [6, 7, 0]])
    refs = torch.randn(2, 2, 1, 32, 64)
    out = model(ids, refs)
    assert out["raw_glyphs"].shape == (2, 3, 1, 32, 16)
    assert out["contextual_glyphs"].shape == (2, 3, 1, 32, 16)
    assert out["coarse_word"].shape == (2, 1, 32, 48)
    assert out["word"].shape == out["coarse_word"].shape
    assert out["scale1"].shape == (2, 32, 2, 3)
    assert out["scale2"].shape == (2, 16, 4, 6)
    assert out["scale3"].shape == (2, 16, 8, 12)
    assert out["scale4"].shape == (2, 8, 16, 24)
    assert torch.all(out["raw_glyphs"][1, 2] == 1)


def test_style_encoders_are_independent():
    model = CharWriteViTGenerator(12, style_dim=32, hidden_dim=32, context_layers=1, context_heads=4, char_width=16, multiscale_depth=1, feature_noise=False, use_merger=True)
    glyph_ids = {id(p) for p in model.glyph_style_encoder.parameters()}
    merge_ids = {id(p) for p in model.merge_style_encoder.parameters()}
    assert glyph_ids.isdisjoint(merge_ids)


def test_writevit_fusion_stacks_context_over_content():
    context = torch.tensor([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]])
    content = torch.tensor([[[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]]])
    grid = WriteViTGlyphFusion.stack_feature_rows(context, content)
    assert grid.shape == (1, 2, 2, 3)
    assert torch.equal(grid[0, :, 0, :], context[0].T)
    assert torch.equal(grid[0, :, 1, :], content[0].T)


def test_seeded_feature_noise_is_reproducible():
    model = CharWriteViTGenerator(12, style_dim=32, hidden_dim=32, context_layers=1, context_heads=4, char_width=16, multiscale_depth=1, feature_noise=True).eval()
    ids = torch.tensor([[3, 4]])
    refs = torch.randn(1, 1, 1, 32, 48)
    with torch.inference_mode():
        first = model(ids, refs, noise_seed=7)["word"]
        second = model(ids, refs, noise_seed=7)["word"]
        different = model(ids, refs, noise_seed=8)["word"]
    assert torch.equal(first, second)
    assert not torch.equal(first, different)


def test_word_loss_reaches_glyph_context_and_multiscale_path():
    model = tiny_model()
    ids = torch.tensor([[3, 4]])
    refs = torch.randn(1, 1, 1, 32, 48)
    output = model(ids, refs)
    (output["word"].mean() + output["raw_glyphs"].mean()).backward()
    assert model.glyph_generator.embedding.weight.grad is not None
    assert model.context_adapter.weight.grad is not None
    assert model.glyph_fusion.to_rgb1.weight.grad is not None


def test_contextualizer_decodes_direct_glyphs_without_pixel_residual():
    model=tiny_model()
    assert not hasattr(model.contextualizer,"scale")
    assert not hasattr(model.contextualizer,"residual")
    assert hasattr(model.contextualizer,"glyph_decoder")


def test_frozen_classifier_still_sends_gradient_to_glyphs():
    classifier = CharacterClassifier(12, width=8).requires_grad_(False)
    glyphs = torch.randn(1, 2, 1, 32, 16, requires_grad=True)
    ids = torch.tensor([[3, 4]])
    loss = glyph_classification_loss(classifier, glyphs, ids)
    loss.backward()
    assert glyphs.grad is not None
    assert all(p.grad is None for p in classifier.parameters())

