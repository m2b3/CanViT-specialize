"""CanViT probes: lightweight heads trained on frozen features.

The core export is SegmentationProbe — loadable from HuggingFace Hub.
Training code is in canvit_probes.training (requires [train] extras).
"""

from canvit_probes.segmentation import SegmentationProbe

__all__ = ["SegmentationProbe"]
