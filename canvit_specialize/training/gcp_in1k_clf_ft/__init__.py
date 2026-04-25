"""CanViT IN1k classification finetuning on GCP TPU v6e.

SPMD training on a TPU v6e-4 slice. The SkyPilot launcher
`sky-train-imagenet.yaml` (co-located) is the canonical entry point;
requires `uv sync --group gcp-in1k-finetune`. See `./README.md` for setup.
"""
