"""Tests for IoUAccumulator — pure CPU, no dataset needed."""

import torch
from canvit_probes.metrics import IoUAccumulator


def test_perfect_predictions() -> None:
    acc = IoUAccumulator(num_classes=3, ignore_index=255, device=torch.device("cpu"))
    targets = torch.tensor([[0, 1, 2, 0]])
    acc.update(targets, targets)
    assert acc.compute() > 0.99


def test_reset_zeros_counters() -> None:
    acc = IoUAccumulator(num_classes=3, ignore_index=255, device=torch.device("cpu"))
    acc.update(torch.tensor([[0, 1, 2]]), torch.tensor([[0, 1, 2]]))
    assert acc.compute() > 0.99
    acc.reset()
    assert acc.intersection.sum().item() == 0
    assert acc.union.sum().item() == 0


def test_ignore_index() -> None:
    acc = IoUAccumulator(num_classes=3, ignore_index=255, device=torch.device("cpu"))
    preds = torch.tensor([[0, 1, 2, 0]])
    targets = torch.tensor([[0, 1, 255, 0]])  # last-1 ignored
    acc.update(preds, targets)
    miou = acc.compute()
    assert miou > 0.5  # not perfect because class 2 has no valid targets
