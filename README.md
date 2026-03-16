# CanViT-probes

Probe definitions and training for CanViT frozen-feature evaluation.

## Installation

```bash
# Just probe loading (minimal deps: torch + huggingface-hub):
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

# With bilinear upsample to target resolution
logits = probe.predict(features, target_size=(512, 512))
```

## Available probes on HuggingFace

ADE20K segmentation (150 classes, LN→BN→Dropout→Conv1×1):

| Probe | Features | Grid | mIoU |
|-------|----------|------|------|
| `canvit/probe-ade20k-40k-s512-c8-in21k` | CanViT canvas | 8×8 | 29.3% |
| `canvit/probe-ade20k-40k-s512-c16-in21k` | CanViT canvas | 16×16 | 35.2% |
| `canvit/probe-ade20k-40k-s512-c32-in21k` | CanViT canvas | 32×32 | 38.5% |
| `canvit/probe-ade20k-40k-s1024-c64-in21k` | CanViT canvas | 64×64 | 39.6% |
| `canvit/probe-ade20k-40k-dv3b-{128..512}px` | DINOv3 ViT-B | varies | 28.9–46.9% |
| `canvit/probe-ade20k-40k-dv3s-{128..512}px` | DINOv3 ViT-S | varies | 25.3–43.2% |

## Architecture

```
canvit_probes/
    segmentation.py    ← SegmentationProbe (LN → BN → Dropout → Conv1×1)
    (future: depth.py, etc.)
```

## Related repos

| Repo | Role |
|------|------|
| [CanViT-PyTorch](https://github.com/m2b3/CanViT-PyTorch) | Core model |
| [CanViT-eval](https://github.com/m2b3/CanViT-eval) | Evaluation (uses probes) |
| [CanViT-Toward-AVFMs](https://github.com/m2b3/CanViT-Toward-AVFMs) | Paper |
