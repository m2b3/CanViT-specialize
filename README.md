# CanViT-probes

Probe definitions, datasets, metrics, and training for CanViT downstream evaluation.

## Installation

```bash
uv add canvit-specialize
```

## Using a pre-trained probe

The probe **architecture** lives in `canvit_pytorch.probes` (graduated
there so inference-only consumers don't need canvit-specialize' training
deps). This package contains the **trainer** + dataset + IoU metric.

```python
from canvit_pytorch import SegmentationProbe

# Load from HuggingFace Hub (see "Available probes" below)
probe = SegmentationProbe.from_pretrained("canvit/probe-ade20k-40k-s1024-c64-in21k")

# Forward: [B, H, W, D] spatial features → [B, num_classes, H, W] logits
logits = probe(features)
```

For the fused **CanViT + probe** pair (one HF artifact, the recommended
form for downstream eval), see `canvit_pytorch.CanViTForSemanticSegmentation`.

## Training

### Required environment variables

Both must be set in the launching shell — they are not optional and CLI flags
do NOT bypass them:

- `COMET_API_KEY` — used to create the `comet_ml.Experiment` (no fallback).
- `ADE20K_ROOT` — used by the `Config.ade20k_root` `default_factory`. **`tyro`
  evaluates the default factory even when `--ade20k-root` is provided as a CLI
  override**, so passing the path on the command line does not avoid the env
  check.

When launching via `ssh remote "..."` (non-interactive), `~/.zshrc` etc. are
not sourced — pass the env vars inline before the command.

### ADE20K segmentation probe (frozen backbone)

```bash
export COMET_API_KEY=$(cat ~/comet_api_key.txt)
export ADE20K_ROOT=/path/to/ADEChallengeData2016

uv run python -m canvit_specialize.training.ade20k train \
  --scene-size 1024 --canvas-grid 64 \
  --batch-size 16 --max-steps 40000 \
  --warmup-steps 1500 --peak-lr 3e-4
```

### ADE20K full fine-tuning (LP-FT)

Initialize from a converged frozen probe, then unfreeze the CanViT backbone
and continue training jointly (Kumar et al. ICLR 2022):

```bash
uv run python -m canvit_specialize.training.ade20k train \
  --scene-size 1024 --canvas-grid 64 \
  --finetune --init-probe-repo canvit/probe-ade20k-40k-s1024-c64-in21k \
  --batch-size 16 --max-steps 40000 \
  --warmup-steps 1500 --peak-lr 2.5e-5 --grad-clip 1.0 --weight-decay 1e-4
```

Fine-tuning differs from frozen-probe training in three ways:
- **Lower LR** (backbone is sensitive; 2.5e-5 vs 3e-4 for fresh probe).
- **Lower weight decay** (1e-4 vs 1e-3) — aggressive WD can destabilize the
  pretrained backbone.
- **Finite grad clipping** (1.0 vs `inf`) — prevents catastrophic updates.

Single feature type only (`canvas_hidden` by default) — the backbone is shared,
so multi-probe fine-tuning would double-step it.

### DINOv3 baseline probe

```bash
uv run python -m canvit_specialize.training.ade20k train-dinov3-probe \
  --scene-size 512 --teacher-repo facebook/dinov3-vitb16-pretrain-lvd1689m
```

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

All probes are on HuggingFace under the `canvit/` organization (private).
Browse at https://huggingface.co/canvit or list via API:

```python
from huggingface_hub import HfApi
[m.id for m in HfApi().list_models(author="canvit") if "probe" in m.id]
```

The headline ADE20K mIoU in the paper (45.9%) uses `canvit/probe-ade20k-40k-s1024-c64-in21k`.

## Architecture

```bash
uv run pypatree
```

## Related repos

| Repo | Role |
|------|------|
| [CanViT-PyTorch-Next](https://github.com/yberreby/CanViT-PyTorch-Next) (private) | Core model (`canvit_pytorch` package) |
| [CanViT-eval](https://github.com/m2b3/CanViT-eval) | Evaluation (uses probes) |
| [CanViT-pretrain](https://github.com/m2b3/CanViT-pretrain) | Model pretraining |
| [CanViT-Toward-AVFMs](https://github.com/m2b3/CanViT-Toward-AVFMs) | Paper |
