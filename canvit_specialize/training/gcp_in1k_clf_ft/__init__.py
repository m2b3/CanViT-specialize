"""CanViT IN1K classification finetuning on GCP TPU v6e.

This package holds the training code that originally lived in `lamarck-infra/demo-pytorch/`
(train_imagenet.py, shared.py, training_utils.py, viz.py). It was migrated here 2026-04-16
so the reproducibility-critical training logic lives alongside other CanViT task-specific
training (ade20k/, in1k/ DINOv3 probes) rather than in a separate private infra repo.

**Hardware constraint:** This code uses `torch_xla` SPMD and requires a TPU v6e slice to
run. It will not run on CPU or CUDA. The SkyPilot launcher (`lamarck-infra/demo-pytorch/
sky-train-imagenet.yaml`) remains the canonical way to launch training.

**Dependency:** Requires the `gcp-in1k-finetune` optional dependency group:
    uv sync --group gcp-in1k-finetune

See ./README.md for setup, environment variables, bucket layout, and launch commands.
"""
