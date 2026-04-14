"""Smoke tests for loss functions and upsample_preds."""

import torch

from canvit_specialize.training.ade20k.loss import ce_loss, upsample_preds


def test_upsample_preds_noop():
    """If already correct size, return as-is."""
    preds = torch.randint(0, 150, (2, 32, 32))
    result = upsample_preds(preds, 32, 32)
    assert torch.equal(result, preds)


def test_upsample_preds_upscale():
    preds = torch.randint(0, 150, (2, 8, 8))
    result = upsample_preds(preds, 32, 32)
    assert result.shape == (2, 32, 32)
    assert result.dtype == torch.long


def test_ce_loss_basic():
    logits = torch.randn(2, 150, 8, 8)
    masks = torch.randint(0, 150, (2, 8, 8))
    loss = ce_loss(logits, masks)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_ce_loss_mask_resize():
    """Masks get resized to match logits when sizes differ."""
    logits = torch.randn(2, 150, 8, 8)
    masks = torch.randint(0, 150, (2, 32, 32))
    loss = ce_loss(logits, masks)
    assert loss.ndim == 0
