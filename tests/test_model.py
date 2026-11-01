import torch

from charwritevit.models.auxiliary import CharacterClassifier
from charwritevit.models.generator import CharWriteViTGenerator, WriteViTGlyphFusion
from charwritevit.losses import glyph_classification_loss


def tiny_model():
    return CharWriteViTGenerator(12, style_dim=32, hidden_dim=32, character_style_layers=1, character_style_heads=4, char_width=16, multiscale_depth=1, feature_noise=False)


def test_generator_shapes_and_single_concat():
    model = tiny_model()
    ids = torch.tensor([[3, 4, 5], [6, 7, 0]])
    refs = torch.randn(2, 2, 1, 32, 64)
    out = model(ids, refs)
    assert out["gnu_glyphs"].shape == (2, 3, 1, 32, 16)
    assert out["styled_glyphs"].shape == (2, 3, 1, 32, 16)
    assert out["coarse_word"].shape == (2, 1, 32, 48)
    assert out["word"].shape == out["coarse_word"].shape
    assert out["scale1"].shape == (2, 32, 2, 3)
    assert out["scale2"].shape == (2, 16, 4, 6)
    assert out["scale3"].shape == (2, 16, 8, 12)
    assert out["scale4"].shape == (2, 8, 16, 24)
    assert torch.all(out["gnu_glyphs"][1, 2] == 1)


def test_style_encoders_are_independent():
    model = CharWriteViTGenerator(12, style_dim=32, hidden_dim=32, character_style_layers=1, character_style_heads=4, char_width=16, multiscale_depth=1, feature_noise=False)
    character_ids = {id(p) for p in model.character_style_encoder.parameters()}
    writevit_ids = {id(p) for p in model.writevit_writer_encoder.parameters()}
    assert character_ids.isdisjoint(writevit_ids)


def test_writevit_fusion_stacks_context_over_content():
    context = torch.tensor([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]])
    content = torch.tensor([[[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]]])
    grid = WriteViTGlyphFusion.stack_feature_rows(context, content)
    assert grid.shape == (1, 2, 2, 3)
    assert torch.equal(grid[0, :, 0, :], context[0].T)
    assert torch.equal(grid[0, :, 1, :], content[0].T)


def test_seeded_feature_noise_is_reproducible():
    model = CharWriteViTGenerator(12, style_dim=32, hidden_dim=32, character_style_layers=1, character_style_heads=4, char_width=16, multiscale_depth=1, feature_noise=True).eval()
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
    (output["word"].mean() + output["styled_glyphs"].mean()).backward()
    assert model.unifont.weight.grad is not None
    assert model.character_style_path.context_blocks[0].attention.qkv.weight.grad is not None
    assert model.character_style_path.style_blocks[0].cross_attention.q.weight.grad is not None
    assert model.glyph_fusion.to_rgb1.weight.grad is not None


def test_character_refiner_has_positional_context_and_glyph_decoder():
    model=tiny_model()
    assert hasattr(model.character_style_path,"position")
    assert hasattr(model.character_style_path,"context_blocks")
    assert hasattr(model.character_style_path,"style_blocks")
    assert hasattr(model.character_style_path,"glyph_decoder")


def test_refined_tokens_replace_both_writevit_gnu_inputs():
    model=tiny_model().eval()
    ids=torch.tensor([[3,4]])
    refs=torch.randn(1,1,1,32,48)
    captured={}
    handle=model.glyph_fusion.register_forward_pre_hook(
        lambda module,args: captured.update({"refined":args[1]})
    )
    with torch.inference_mode(): output=model(ids,refs,noise_seed=1)
    handle.remove()
    assert torch.equal(captured["refined"],output["refined_tokens"])


def test_character_refiner_masks_padding_without_changing_valid_tokens():
    model=tiny_model().eval()
    style=torch.randn(1,5,32)
    valid=torch.randn(1,2,32)
    padded=torch.cat([valid,torch.randn(1,2,32)],dim=1)
    with torch.inference_mode():
        short,_=model.character_style_path(valid,style,torch.zeros(1,2,dtype=torch.bool))
        long,_=model.character_style_path(padded,style,torch.tensor([[False,False,True,True]]))
    assert torch.allclose(short,long[:,:2],atol=1e-5,rtol=1e-5)
    assert torch.count_nonzero(long[:,2:]) == 0


def test_frozen_classifier_still_sends_gradient_to_glyphs():
    classifier = CharacterClassifier(12, width=8).requires_grad_(False)
    glyphs = torch.randn(1, 2, 1, 32, 16, requires_grad=True)
    ids = torch.tensor([[3, 4]])
    loss = glyph_classification_loss(classifier, glyphs, ids)
    loss.backward()
    assert glyphs.grad is not None
    assert all(p.grad is None for p in classifier.parameters())


def test_joint_classifier_step_uses_detached_query_aligned_cells():
    model=tiny_model()
    classifier=CharacterClassifier(12,width=8)
    ids=torch.tensor([[3,4]])
    output=model(ids,torch.randn(1,1,1,32,48))
    cells=torch.cat([output["gnu_glyphs"].detach().flatten(0,1),output["styled_glyphs"].detach().flatten(0,1)])
    labels=ids.flatten().repeat(2)
    torch.nn.functional.cross_entropy(classifier(cells),labels).backward()
    assert classifier.head.weight.grad is not None
    assert all(parameter.grad is None for parameter in model.parameters())


def test_real_unifont_table_is_frozen_but_projection_trains():
    table=torch.randint(0,2,(12,256),dtype=torch.float32)
    model=CharWriteViTGenerator(12,style_dim=32,hidden_dim=32,character_style_layers=1,
                                character_style_heads=4,multiscale_depth=1,
                                feature_noise=False,unifont_table=table)
    output=model(torch.tensor([[3,4]]),torch.randn(1,1,1,32,48))
    output["word"].mean().backward()
    assert model.unifont.weight.requires_grad is False
    assert model.unifont.weight.grad is None
    assert model.unifont_projection.weight.grad is not None

