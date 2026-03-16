# CanViT-probes

Probe definitions, datasets, metrics, and training for CanViT frozen-feature evaluation.

## Installation

```bash
# Probe loading only (minimal deps: torch + huggingface-hub):
uv add canvit-probes

# With training support:
uv add "canvit-probes[train]"
```

## Usage

```python
from canvit_probes import SegmentationProbe

# Load from HuggingFace Hub
probe = SegmentationProbe.from_pretrained("canvit/probe-ade20k-40k-s512-c32-in21k")

# Forward: [B, H, W, D] spatial features → [B, num_classes, H, W] logits
logits = probe(features)
```

## Available probes

All probes are on HuggingFace under the `canvit/` organization (private).
Browse at https://huggingface.co/canvit or list via API:
```python
from huggingface_hub import HfApi
[m.id for m in HfApi().list_models(author="canvit") if "probe" in m.id]
```

## Architecture

```bash
uv run pypatree
```

## Related repos

| Repo | Role |
|------|------|
| [CanViT-PyTorch](https://github.com/m2b3/CanViT-PyTorch) | Core model |
| [CanViT-eval](https://github.com/m2b3/CanViT-eval) | Evaluation (uses probes) |
| [CanViT-pretrain](https://github.com/m2b3/CanViT-pretrain) | Model pretraining |
| [CanViT-Toward-AVFMs](https://github.com/m2b3/CanViT-Toward-AVFMs) | Paper |
