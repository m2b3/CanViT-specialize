# CanViT IN1k classification finetuning (GCP TPU v6e)

Training code for the flagship IN1k-finetuned checkpoint. See the paper for the headline number.

- **Checkpoint:** [canvit/canvitb16-add-vpe-finetune-g128px-s512px-in1k-2026-04-06](https://huggingface.co/canvit/canvitb16-add-vpe-finetune-g128px-s512px-in1k-2026-04-06)
- **Hardware:** TPU v6e-4 via `torch_xla` SPMD. Does NOT run on CPU or CUDA.

## Files

| File | Role |
|------|------|
| `train_imagenet.py` | Entry point: SPMD training loop, chunked BPTT, validation, checkpointing, Comet. |
| `shared.py` | TFRecord dataloader, `load_classifier()`, constants. |
| `training_utils.py` | `ValLoader`, checkpointing, LR schedule, early stopping. XLA-free (CPU-testable). |
| `viz.py` | Comet validation viz. |
| `sky-train-imagenet.yaml` | SkyPilot launcher (**source of truth for resources, HPs, regions**). |
| `setup_tpu.sh` | TPU VM setup: uv + Python + ldconfig + gcsfuse + cached `uv sync`. |

## Dependency group

```bash
uv sync --group gcp-in1k-finetune
```
All version pins live in `pyproject.toml`. The `torch==2.9.0` pin is load-bearing: `torch_xla[tpu]==2.9.0` declares no torch dep in its wheel metadata, so without an explicit pin uv resolves to the latest `torch>=2.0.0` from `pytorch-cu128`, which breaks `_XLAC.so`'s ABI (`undefined symbol: _wrap_outputs`).

## Prerequisites (one-time, launch laptop)

1. `pip install skypilot[gcp]` + `sky check gcp` green.
2. `gcloud auth application-default login` (OAuth; re-run when expired).
3. `~/.sky/config.yaml` with `gcp.remote_identity: SERVICE_ACCOUNT`. Worker VMs then authenticate via the attached GCE service account — no SA key, no expiry. (SkyPilot rejects this as a per-task field, so it MUST be at user-level.)
4. Comet API key at `~/.config/comet_api_key.txt` (600). Optional at training time; required for `push_finetuned.py` → Comet HP augmentation.
5. HuggingFace token at `~/.cache/huggingface/token` (`huggingface-cli login`). Needs write scope for push.
6. GCS buckets following the `${GCS_BUCKET_PREFIX}-${REGION}` convention:
   - `gs://${GCS_BUCKET_PREFIX}-<region>/datasets/imagenet/` — TFRecord shards, per region you want to train in.
   - `gs://${GCS_BUCKET_PREFIX}-us-central1/` — checkpoint bucket (pinned to us-central1 to survive EAGER_NEXT_REGION failover). Edit the `file_mounts.source` in `sky-train-imagenet.yaml` to match.

   Pass the prefix at launch with `--env GCS_BUCKET_PREFIX=your-prefix`.

## Secrets

```bash
export COMET_API_KEY=$(cat ~/.config/comet_api_key.txt)
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
sky [jobs] launch … --secret COMET_API_KEY --secret HF_TOKEN
```
`--secret` reads from the launch shell and injects into the VM env. Never written to yaml, git, or sky logs.

## Workflow

### Train (production)
```bash
sky jobs launch canvit_specialize/training/gcp_in1k_clf_ft/sky-train-imagenet.yaml -y \
  --secret COMET_API_KEY --secret HF_TOKEN
```

### Dev cluster (interactive iteration)
```bash
# Cheaper TPU spec for import/setup checks. sky prints live spot price.
sky launch canvit_specialize/training/gcp_in1k_clf_ft/sky-train-imagenet.yaml -c tpu-dev -i 30 --yes \
  --gpus tpu-v6e-1:1 --secret COMET_API_KEY --secret HF_TOKEN

ssh tpu-dev                                              # interactive
sky exec tpu-dev canvit_specialize/.../sky-train-imagenet.yaml   # re-sync workdir + re-run
sky stop tpu-dev     # pause
sky down tpu-dev     # terminate
```

### Publish
```bash
uv run python scripts/push_finetuned.py \
  --checkpoint /path/to/best.pt \
  --pretrained-repo canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02 \
  --probe-repo yberreby/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe \
  --canvas-grid 32 \
  --repo-id canvit/<new-name> --public
```

## Hyperparameters

**Defaults live in `sky-train-imagenet.yaml`** (`envs:` block). `uv run python -m canvit_specialize.training.gcp_in1k_clf_ft.train_imagenet --help` for the full tyro CLI.

### Scaling across TPU slices

Defaults in `sky-train-imagenet.yaml` target v6e-4. On a different slice, scale `BATCH_SIZE` and `LR` linearly with chip count (the code does NOT auto-scale); override via `--env BATCH_SIZE=… --env LR=…`. Running the v6e-4 defaults on a smaller slice will OOM at XLA compile.

### Non-obvious choices

- **`chunk_size == n_glimpses`** — full BPTT; chunked BPTT drops final-step gradient quality.
- **`min_viewpoint_scale=0.05`** — matches pretraining `p(s) ∝ (1-s)` truncation.
- **Mixed precision OFF (fp32)** — `torch.autocast("xla", bf16)` caused ~1000× gnorm explosion through full BPTT; throughput was identical without it (XLA uses bf16 internally for matmuls).

