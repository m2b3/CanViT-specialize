"""CanViT IN1k classification finetuning on GCP TPU v6e.

SPMD training on a TPU v6e-4 slice; does NOT run on CPU or CUDA. The SkyPilot
launcher ``sky-train-imagenet.yaml`` (co-located) is the canonical entry point.

Requires the ``gcp-in1k-finetune`` optional dependency group::

    uv sync --group gcp-in1k-finetune

See ``./README.md`` for setup, environment variables, bucket layout, and the
end-to-end train → verify → push workflow.
"""
