# CanViT-specialize

Trainer, datasets, and metrics for CanViT downstream-task probes + LP-FT fine-tuning. The probe *architecture* lives upstream in `canvit_pytorch.probes` (inference-only consumers don't need this package); this repo holds the *training* loops.

## Installation

Not on PyPI. Install via git:

```bash
# As a dep in another project's pyproject.toml:
# [tool.uv.sources]
# canvit-specialize = { git = "https://github.com/m2b3/CanViT-specialize.git" }

# Or directly in an ad-hoc venv:
uv add "canvit-specialize @ git+https://github.com/m2b3/CanViT-specialize.git"
```

### torch wheel per target

The base `uv sync` pulls `torch==2.9.0` from PyPI, which on Linux x86_64 IS
the CUDA-12.8 build — correct for Nibi H100 and crockett RTX 4090.

- **GCP TPU v6e (IN1K finetune, `--group gcp-in1k-finetune`)**: `pyproject.toml`
  group-scopes the `pytorch-cpu` index so the TPU group installs
  `torch==2.9.0+cpu` / `torchvision==0.24.0+cpu`. No nvidia_* cu12 satellites,
  no `libtorch_cuda.so`. The `setup_tpu.sh` invokes
  `uv sync --group gcp-in1k-finetune` directly — no post-sync reinstall
  needed.
- **Nibi H100 / crockett RTX 4090 (ADE20K probe training, default deps)**:
  no action needed. If a specific box needs a different CUDA version
  (e.g. cu121 for an older driver), set `UV_INDEX_URL` at sync time:
  ```bash
  UV_INDEX_URL=https://download.pytorch.org/whl/cu121 uv sync
  ```
- **macOS dev**: PyPI default is already CPU-only on Darwin — no special
  handling.

PyPI's `torch==2.9.0` for Linux is byte-identical to `torch==2.9.0+cu128`
(verified via `sha256sum libtorch_cpu.so`); the `+cu128` suffix is a
dist-info label only. The truly CPU-only wheel lives exclusively at
`https://download.pytorch.org/whl/cpu`.

## Using a pre-trained probe

```python
from canvit_pytorch import SegmentationProbe
probe = SegmentationProbe.from_pretrained("canvit/probe-ade20k-40k-s1024-c64-in21k")
logits = probe(features)  # [B, H, W, D] → [B, num_classes, H, W]
```

For the fused **CanViT + probe** pair (one HF artifact, recommended for downstream eval), see `canvit_pytorch.CanViTForSemanticSegmentation`.

## Training

### Required environment variables

- `COMET_API_KEY` — required; creates `comet_ml.Experiment` (no fallback).
- `ADE20K_ROOT` — required even if you also pass `--ade20k-root`. `tyro` evaluates the config's `default_factory` regardless of CLI overrides.

Non-interactive `ssh remote "…"` does not source `~/.zshrc` — pass env vars inline.

### ADE20K segmentation probe (frozen backbone)

```bash
export COMET_API_KEY=$(cat ~/comet_api_key.txt)
export ADE20K_ROOT=/path/to/ADEChallengeData2016

uv run python -m canvit_specialize.training.ade20k train \
  --scene-size 1024 --canvas-grid 64
```

### DINOv3 baseline probe

```bash
uv run python -m canvit_specialize.training.ade20k train-dinov3-probe
```

### IN1K classification finetuning on GCP TPU v6e

Training code + SkyPilot launcher live in `canvit_specialize/training/gcp_in1k_clf_ft/`.
Deps: `uv sync --group gcp-in1k-finetune` (TPU-VM only — pulls `torch_xla[tpu]` + `tfrecord`).
End-to-end workflow (train → verify → `scripts/push_finetuned.py` to HF) is documented
in [`canvit_specialize/training/gcp_in1k_clf_ft/README.md`](canvit_specialize/training/gcp_in1k_clf_ft/README.md).

### Smoke testing (validate env / import / dataset extraction without burning a long job)

There is no separate smoke sbatch. Smoke tests are short parameterizations of
the main training sbatch:

```bash
# Quick GPU smoke (5 train steps, no validation, ~5 min wallclock)
sbatch --time=00:15:00 slurm/train_ade20k_canvit.sbatch \
  --scene-size 512 --canvas-grid 32 \
  --max-steps 5 --val-every 99999 --warmup-steps 1 \
  --batch-size 2 --num-workers 2

# CPU env validation only (no GPU consumed; submit to CPU partition):
salloc --time=00:15:00 --account=def-skrishna_cpu --cpus-per-task=4 --mem=16G
# then in the allocation:
cd ~/scratch/canvit-specialize && cp .envrc.nibi .envrc && source slurm/setup.sh
uv run python -c "from canvit_pytorch import SegmentationProbe; \
  p = SegmentationProbe.from_pretrained('canvit/probe-ade20k-40k-s512-c32-in21k'); \
  print(f'OK, {sum(x.numel() for x in p.parameters()):,} params')"
```

## Where training runs

| Machine | Purpose | Notes |
|---------|---------|-------|
| **Nibi** (H100, SLURM) | Production probe training | ADE20K at `$SLURM_TMPDIR/ADEChallengeData2016`, submit via `sbatch` |
| **Crockett** (RTX 4090) | Quick iteration, smoke tests | ADE20K at `/datasets/ADE20k/ADEChallengeData2016`, run `nohup` directly |

On Nibi, fetch the dataset into node-local NVMe via the SLURM prolog (already
handled for existing jobs). On crockett, one GPU process at a time —
DataLoader workers (CPU-only) are fine.

## Available probes

Probes live under the `canvit/` HuggingFace organization — mostly public,
browsable at https://huggingface.co/canvit. List via API:

```python
from huggingface_hub import HfApi
[m.id for m in HfApi().list_models(author="canvit") if "probe" in m.id]
```

Curated collections:

- `canvit/canvit-ade20k-segmentation-probes-pytorch` — CanViT canvas probes at multiple `(scene_size, canvas_grid)` points.
- `canvit/dinov3-ade20k-segmentation-probes-pytorch` — DINOv3 baseline probes for passive comparison.

The paper's headline ADE20K mIoU uses `canvit/probe-ade20k-40k-s1024-c64-in21k` (public). See the paper itself for the number.

## Architecture

```bash
uv run pypatree
```

## Related repos

| Repo | Role |
|------|------|
| [CanViT-PyTorch](https://github.com/m2b3/CanViT-PyTorch) (public, canonical) | Core model (`canvit_pytorch` package), probe architecture (`canvit_pytorch.probes`) |
| [CanViT-eval](https://github.com/m2b3/CanViT-eval) | Evaluation (uses probes) |
| [CanViT-pretrain](https://github.com/m2b3/CanViT-pretrain) | Model pretraining |
| [CanViT-Toward-AVFMs](https://github.com/m2b3/CanViT-Toward-AVFMs) | Paper |
