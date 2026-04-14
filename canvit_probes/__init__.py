"""CanViT probes: training, datasets, and segmentation/IoU metrics.

The probe **architectures** themselves (e.g. :class:`SegmentationProbe`)
are first-class citizens of canvit-pytorch (`canvit_pytorch.probes`) so
they can be composed into HF model wrappers and consumed by inference-only
clients without canvit-probes' training-only dependencies.

This package contains the **training side**: data loaders (ADE20K), IoU
metrics, training loops, and HuggingFace push scripts.
"""
