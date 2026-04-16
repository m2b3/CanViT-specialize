# CanViT IN1K classification finetuning (GCP TPU v6e)

This package contains the training code used to finetune CanViT on ImageNet-1K on GCP TPU v6e. It produced the flagship IN1K-finetuned checkpoint reported in the paper (C2F t=12 = 84.51% top-1).

## Status

- **Origin:** Migrated from `~/code/lamarck-infra/demo-pytorch/` on 2026-04-16 to consolidate training code in one place (this repo). SkyPilot launcher remains in `lamarck-infra/demo-pytorch/sky-train-imagenet.yaml`.
- **Hardware:** TPU v6e-4 slice on GCP. Does NOT run on CPU or CUDA (uses `torch_xla` SPMD).
- **Checkpoint on HF:** `canvit/canvitb16-add-vpe-finetune-g128px-s512px-in1k-2026-04-06`.

## Files

| File | LOC | Role |
|------|-----|------|
| `train_imagenet.py` | 575 | Entry point. Training loop, SPMD sharding, chunked BPTT, validation, checkpointing, Comet logging. Run via `python -m canvit_specialize.training.gcp_in1k_clf_ft.train_imagenet …`. |
| `shared.py` | 286 | Constants (`CANVAS_GRID`, `IMAGENET_MEAN/STD`), TFRecord dataloaders, classifier loading via `CanViTForImageClassification.from_pretrained_with_probe`. |
| `training_utils.py` | 100 | Pure utilities: `ValLoader`, checkpointing, LR schedule lambda, early stopping. No XLA dependency (testable on CPU). |
| `viz.py` | 68 | Comet validation viz: top-5 prediction bar charts per sample. |

## Dependencies

This code is NOT included in the base canvit-specialize install. Requires the `gcp-in1k-finetune` optional dependency group:

```bash
# Inside canvit-specialize (on the TPU VM):
uv sync --group gcp-in1k-finetune
```

Extra deps it pulls: `torch_xla[tpu]==2.9.0` (Linux only), `tfrecord` (for IN1K TFRecord format), `datasets` (unused at runtime but retained for compat; drop if confirmed unused).

## Required environment

- **COMET_API_KEY** — optional. If unset, training runs without Comet logging.
- **HF_TOKEN** — required. Pulls the pretrained `canvit/canvitb16-add-vpe-pretrain-*` checkpoint and the DINOv3 probe (`yberreby/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe`) from Hugging Face Hub.
- **SKYPILOT_TASK_ID** — set automatically by SkyPilot. Used in the checkpoint directory name so jobs across spot recoveries share the same state.
- **GCS bucket naming convention:** `lamarck-<gcp-region>`. Training auto-detects the VM's region via the GCE metadata server and mounts `gs://lamarck-<region>` via `gcsfuse`. ImageNet data must exist at `gs://lamarck-<region>/datasets/imagenet/` (TFRecord format, `train-*` and `validation-*` shards).
- **Checkpoint bucket:** pinned to `gs://lamarck-us-central1` via SkyPilot `file_mounts` (MOUNT_CACHED), so checkpoints survive `EAGER_NEXT_REGION` failover to other regions.

## Launch workflow

Canonical launch command (from `lamarck-infra/`):

```bash
export COMET_API_KEY=$(cat ~/.config/comet_api_key.txt)
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
sky jobs launch demo-pytorch/sky-train-imagenet.yaml -y \
  --secret COMET_API_KEY --secret HF_TOKEN
```

The yaml sets up the TPU VM, clones this repo, runs `uv sync --group gcp-in1k-finetune`, and invokes `python -m canvit_specialize.training.gcp_in1k_clf_ft.train_imagenet …` with HP env vars.

Dev iteration on a warm cluster:

```bash
sky launch demo-pytorch/sky-train-imagenet.yaml -c tpu -i 30 --yes \
  --secret COMET_API_KEY --secret HF_TOKEN
sky exec tpu demo-pytorch/sky-train-imagenet.yaml
```

## Flagship hyperparameters (used for the published checkpoint)

| Param | Value | Note |
|-------|-------|------|
| Batch size | 256 | Linear-scaled LR anchor |
| Epochs | 20 | Early stop possible via `--early-stop-delta` |
| Peak LR | 2.5e-5 | Scales linearly with batch size |
| Weight decay | 1e-4 | WD=0.01 causes divergence in multi-glimpse regime |
| Warmup | 25 000 steps (5 epochs) | Linear → cosine to 0 |
| Label smoothing | 0.1 | `torch.nn.functional.cross_entropy(label_smoothing=…)` |
| N glimpses | 4 | F-IID (start-full-scene, then IID random) |
| Chunk size | 4 | Full BPTT (`chunk_size == n_glimpses`) |
| Gradient clipping | 1.0 | Max-norm |
| Min viewpoint scale | 0.05 | Matches pretraining `p(s) ∝ (1 − s)` with `s_min = 0.05` |
| Mixed precision | **OFF** | `torch.autocast(bf16)` caused 1000× gnorm explosion through full BPTT on TPU XLA. XLA uses bf16 internally for matmuls regardless. |

## Anonymization for NeurIPS submission

Before packaging into the anonymized code zip:

- Scrub `COMET_WORKSPACE = "m2b3-ava"` (train_imagenet.py) and `COMET_PROJECT = "canvit-in1k-finetune"` → placeholders or env-driven.
- Scrub `gs://lamarck-<region>` in the sky yaml → `gs://<YOUR_BUCKET>/datasets/imagenet/` placeholders.
- Scrub `PROBE_REPO = "yberreby/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe"` (shared.py) if the `yberreby/*` namespace is identifying.
- Scrub `canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02` (shared.py) if the `canvit/` org is identifying (the org name is intentionally generic but verify).

## TPU v6e-4 region options (from sky yaml, ordered by spot price)

- `gcp/asia-southeast1` — $1.24/hr spot (4 × $0.31/chip)
- `gcp/us-central1` — $2.16/hr spot (4 × $0.54/chip)
- `gcp/us-east5` — $4.88/hr spot (capacity fallback)

`EAGER_NEXT_REGION` failover handles spot preemption across these three regions. Checkpoint bucket is pinned to `us-central1` so state is durable across failover. Expect a cross-region GCS read penalty on job recovery, which is acceptable given spot availability.

## Not included here

- `bench_*.py`, `profile_step.py`, `test_*.py`, `sweep_optuna.py`, `diagnose_gradnorm.py`, `multi_glimpse.py`, `comet_check.py`, `migrate_and_push_finetuned.py`, `verify_finetuned_in1k.py` — all stay in `lamarck-infra/demo-pytorch/` as exploratory / one-off utilities, not reproducibility-critical.
- `demo-jax/` — separate JAX pathway, not part of the flagship training recipe.
- `push_to_hub.py` — HF publishing of finetuned checkpoints. Candidate for later migration if user wants a standard publishing flow (analogous to canvit-specialize's `scripts/push_probes.py`).
