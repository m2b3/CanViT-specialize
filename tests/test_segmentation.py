"""Tests for SegmentationProbe."""

import torch
from canvit_probes import SegmentationProbe


def test_forward_shape() -> None:
    probe = SegmentationProbe(embed_dim=768, num_classes=150, dropout=0.0)
    out = probe(torch.randn(2, 32, 32, 768))
    assert out.shape == (2, 150, 32, 32)


def test_forward_no_ln() -> None:
    probe = SegmentationProbe(embed_dim=768, num_classes=150, dropout=0.0, use_ln=False)
    out = probe(torch.randn(2, 16, 16, 768))
    assert out.shape == (2, 150, 16, 16)


def test_predict_upsample() -> None:
    probe = SegmentationProbe(embed_dim=768, num_classes=150, dropout=0.0)
    out = probe.predict(torch.randn(1, 8, 8, 768), target_size=(512, 512))
    assert out.shape == (1, 150, 512, 512)


def test_embed_dim_mismatch() -> None:
    probe = SegmentationProbe(embed_dim=768, num_classes=150)
    try:
        probe(torch.randn(1, 8, 8, 384))
        assert False, "Should have raised"
    except AssertionError:
        pass


def test_state_dict_roundtrip() -> None:
    probe = SegmentationProbe(embed_dim=1024, num_classes=150, dropout=0.1)
    sd = probe.state_dict()
    probe2 = SegmentationProbe(embed_dim=1024, num_classes=150, dropout=0.1)
    probe2.load_state_dict(sd)
    x = torch.randn(1, 32, 32, 1024)
    probe.eval(); probe2.eval()
    assert torch.allclose(probe(x), probe2(x))


def test_from_pretrained() -> None:
    """Integration: load real probe from HuggingFace."""
    probe = SegmentationProbe.from_pretrained("canvit/probe-ade20k-40k-s512-c32-in21k")
    assert probe.embed_dim == 1024
    assert probe.num_classes == 150
    out = probe(torch.randn(1, 32, 32, 1024))
    assert out.shape == (1, 150, 32, 32)
