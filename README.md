# CanViT-specialize

Training loops for [CanViT](https://github.com/m2b3/CanViT-PyTorch) downstream probes and finetuning. The probe and finetuning *architectures* live in `canvit_pytorch.probes` / `canvit_pytorch.CanViTForImageClassification`; this repo holds the training side (data loaders, IoU metrics, training loops, HF push scripts).

## Install

```bash
uv add "canvit-specialize @ git+https://github.com/m2b3/CanViT-specialize.git"
```

The base `uv sync` pulls `torch==2.9.0` from PyPI, which on Linux x86_64 is the CUDA-12.8 build (correct for H100 / RTX 4090). For TPU finetuning, see the [`gcp_in1k_clf_ft/README.md`](canvit_specialize/training/gcp_in1k_clf_ft/README.md).

## Using a pre-trained probe

```python
from canvit_pytorch import SegmentationProbe
probe = SegmentationProbe.from_pretrained("canvit/probe-ade20k-40k-s1024-c64-in21k")
logits = probe(features)  # [B, H, W, D] → [B, num_classes, H, W]
```

For the fused **CanViT + probe** pair (recommended for downstream eval), see `canvit_pytorch.CanViTForSemanticSegmentation`.

## Training

`COMET_API_KEY`, `COMET_WORKSPACE`, and `ADE20K_ROOT` must be set before training.

```bash
cp .envrc.example .envrc && direnv allow
# Edit .envrc to point at your dataset / Comet workspace.
```

### ADE20K segmentation probe (frozen backbone)

```bash
uv run python -m canvit_specialize.training.ade20k train \
  --scene-size 1024 --canvas-grid 64
```

### DINOv3 baseline probe

```bash
uv run python -m canvit_specialize.training.ade20k train-dinov3-probe
```

### IN1k classification finetuning on GCP TPU v6e

See [`canvit_specialize/training/gcp_in1k_clf_ft/README.md`](canvit_specialize/training/gcp_in1k_clf_ft/README.md).

## Available probes

Probes live under [`canvit/`](https://huggingface.co/canvit) on HuggingFace.

- [`canvit/canvit-ade20k-segmentation-probes-pytorch`](https://huggingface.co/collections/canvit/canvit-ade20k-segmentation-probes-pytorch) — CanViT canvas probes at multiple `(scene_size, canvas_grid)` points.
- [`canvit/dinov3-ade20k-segmentation-probes-pytorch`](https://huggingface.co/collections/canvit/dinov3-ade20k-segmentation-probes-pytorch) — DINOv3 baseline probes.

The paper's headline ADE20K mIoU uses [`canvit/probe-ade20k-40k-s1024-c64-in21k`](https://huggingface.co/canvit/probe-ade20k-40k-s1024-c64-in21k).

## Citation

```bibtex
@article{berreby2026canvit,
  title={CanViT: Toward Active-Vision Foundation Models},
  author={Berreby, Yoha{\"i}-Eliel and Du, Sabrina and Durand, Audrey and Krishna, B. Suresh},
  year={2026},
  eprint={2603.22570},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2603.22570}
}
```

## License

MIT. See [LICENSE](LICENSE) for details.
