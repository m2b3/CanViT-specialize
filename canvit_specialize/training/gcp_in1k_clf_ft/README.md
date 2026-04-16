# CanViT IN1K classification finetuning (GCP TPU v6e)

Training code used to finetune CanViT on ImageNet-1K on GCP TPU v6e. Produces the flagship IN1K-finetuned checkpoint reported in the paper (C2F t=12 = 84.51% top-1).

- **Checkpoint on HF:** [canvit/canvitb16-add-vpe-finetune-g128px-s512px-in1k-2026-04-06](https://huggingface.co/canvit/canvitb16-add-vpe-finetune-g128px-s512px-in1k-2026-04-06).
- **Hardware:** TPU v6e-4 (`torch_xla` SPMD). Does NOT run on CPU or CUDA.
- **Origin:** Moved from `~/code/lamarck-infra/demo-pytorch/` on 2026-04-16. One home for training code (this repo); lamarck-infra is now just the gcloud/sky ergonomics wrapper (not even — everything is here too).

## Files

| File | Role |
|------|------|
| `train_imagenet.py` | Entry point: training loop, SPMD sharding, chunked BPTT, validation, checkpointing, Comet logging. Invoked via `python -m canvit_specialize.training.gcp_in1k_clf_ft.train_imagenet …`. |
| `shared.py` | Constants (`CANVAS_GRID`, `IMAGENET_MEAN/STD`), TFRecord dataloaders, `load_classifier()` wrapping `CanViTForImageClassification.from_pretrained_with_probe`. |
| `training_utils.py` | Pure utilities: `ValLoader`, checkpointing, LR schedule lambda, early stopping. No XLA dep (testable on CPU). |
| `viz.py` | Comet validation viz: top-5 prediction bar charts. |
| `sky-train-imagenet.yaml` | SkyPilot launcher (managed job or warm cluster). |
| `setup_tpu.sh` | TPU VM environment setup: uv + Python 3.12 + ldconfig + gcsfuse + uv cache restore + `uv sync --group gcp-in1k-finetune`. |

## Dependency group

Pulled via:
```bash
uv sync --group gcp-in1k-finetune
```
Extras: `torch_xla[tpu]==2.9.0` (Linux only), `tfrecord` (IN1K TFRecord decode). Base install stays lean for users who only want ADE20K probe training.

## Required environment

- **COMET_API_KEY** — optional at training time (the run logs without Comet if unset), but required for `scripts/push_finetuned.py` to augment HF config.json.
- **HF_TOKEN** — required. Pulls pretrained `canvit/canvitb16-add-vpe-pretrain-*` + DINOv3 probe `yberreby/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe`.
- **`gs://lamarck-<region>`** bucket convention. Training auto-detects the VM's region via GCE metadata server and mounts `gs://lamarck-<region>/datasets/imagenet/` via `gcsfuse`. Override with `LAMARCK_GCS_BUCKET=gs://your-bucket`.
- **Checkpoint bucket:** pinned to `gs://lamarck-us-central1` via the yaml's `file_mounts` (MOUNT_CACHED), so state survives `EAGER_NEXT_REGION` spot failover.

## End-to-end workflow

### 1. Train

**Managed job (production, spot recovery):**
```bash
cd ~/code/CanViT-specialize
export COMET_API_KEY=$(cat ~/.config/comet_api_key.txt)
export HF_TOKEN=$(cat ~/.cache/huggingface/token)

sky jobs launch canvit_specialize/training/gcp_in1k_clf_ft/sky-train-imagenet.yaml -y \
  --secret COMET_API_KEY --secret HF_TOKEN
```

**Dev iteration on a warm cluster (tight feedback loop):**
```bash
# Launch once, stays up.
sky launch canvit_specialize/training/gcp_in1k_clf_ft/sky-train-imagenet.yaml -c tpu-dev -i 30 --yes \
  --secret COMET_API_KEY --secret HF_TOKEN

# Re-run after local code changes (syncs workdir).
sky exec tpu-dev canvit_specialize/training/gcp_in1k_clf_ft/sky-train-imagenet.yaml

# Pause to save $ (auto-stops after 30 min idle via `-i 30`).
sky stop tpu-dev

# Fully terminate.
sky down tpu-dev
```

### 2. Verify locally (optional)

Pull the checkpoint off the MOUNT_CACHED bucket or off HF, load with `CanViTForImageClassification.from_pretrained_with_probe`, run an IN1K val pass via `canvit-eval`. The paper-side eval pipeline (`canvit-eval batch --tasks in1k-clf`) is the canonical way to get timestep-wise top-1 numbers.

### 3. Publish to HuggingFace

```bash
cd ~/code/CanViT-specialize
export COMET_API_KEY=$(cat ~/.config/comet_api_key.txt)
export HF_TOKEN=$(cat ~/.cache/huggingface/token)

uv run python scripts/push_finetuned.py \
  --checkpoint /path/to/best.pt \
  --pretrained-repo canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02 \
  --probe-repo yberreby/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe \
  --canvas-grid 32 \
  --repo-id canvit/<your-new-repo-name> \
  --public
```

This remaps training-checkpoint keys to the `CanViTForImageClassification` layout, strips pretraining-only prefixes (`scene_cls_head.*`, `scene_patches_head.*`, `cls_standardizers.*`, `scene_standardizers.*`), sanity-forwards a random scene, pushes, and (if `COMET_API_KEY` is set) augments config.json with training HPs from Comet.

## Flagship hyperparameters

Defaults in `sky-train-imagenet.yaml` match the published `canvit/canvitb16-add-vpe-finetune-g128px-s512px-in1k-2026-04-06` checkpoint:

| Param | Value | Note |
|-------|-------|------|
| Batch size | 256 | Linear-scaled LR anchor |
| Epochs | 20 | Early stop possible via `--early-stop-delta` |
| Peak LR | 2.5e-5 | Scales linearly with batch size |
| Weight decay | 1e-4 | WD=0.01 causes divergence in multi-glimpse regime |
| Warmup | 25 000 steps (5 epochs) | Linear → cosine to 0 |
| Label smoothing | 0.1 | `F.cross_entropy(label_smoothing=…)` |
| N glimpses | 4 | F-IID (t=0 full-scene, then IID random) |
| Chunk size | 4 | Full BPTT (`chunk_size == n_glimpses`) |
| Gradient clipping | 1.0 | Max-norm |
| Min viewpoint scale | 0.05 | Matches pretraining `p(s) ∝ (1 − s)` with `s_min = 0.05` |
| Mixed precision | **OFF (fp32)** | `torch.autocast(bf16)` caused 1000× gnorm explosion through full BPTT on TPU XLA. XLA uses bf16 internally for matmuls regardless. |

## TPU v6e-4 regions (from sky yaml, ordered by spot price)

- `gcp/asia-southeast1` — ~$1.24/hr spot (4 × $0.31/chip)
- `gcp/us-central1` — ~$2.16/hr spot (4 × $0.54/chip)
- `gcp/us-east5` — ~$4.88/hr spot (capacity fallback)

`EAGER_NEXT_REGION` failover handles spot preemption. Checkpoint bucket pinned to `us-central1` so state is durable across failover; expect a cross-region GCS read penalty on recovery.

## Anonymization checklist for NeurIPS code zip

Scrub before packaging:
- `shared.py` — `PROBE_REPO = "yberreby/…"` if the `yberreby/*` namespace is identifying. The `canvit/*` namespace is intentionally generic.
- `train_imagenet.py` — `COMET_WORKSPACE = "m2b3-ava"`, `COMET_PROJECT = "canvit-in1k-finetune"`. Replace with neutral placeholders or make env-driven.
- `sky-train-imagenet.yaml` — `gs://lamarck-<region>` convention. Replace with `gs://<YOUR_BUCKET>/datasets/imagenet/` placeholders + a note about the region-autodetect pattern.
- `setup_tpu.sh` — `LAMARCK_GCS_BUCKET` env var name. Rename to `GCS_CACHE_BUCKET` or similar.

## Cross-refs

- `scripts/push_finetuned.py` — HF publish flow for a new finetuned checkpoint.
- `scripts/upload_utils.py` — shared HF utilities (`upload_probe_to_hub`, `augment_hf_config_with_comet`, `pull_comet_params`).
- Top-level `README.md` — canvit-specialize installation and ADE20K probe training.
