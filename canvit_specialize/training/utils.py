"""Training utilities: viewpoint generation."""

from typing import Literal

import torch
from canvit_pytorch import Viewpoint
from canvit_pytorch.policies import coarse_to_fine_viewpoints, random_viewpoints

ViewpointPolicyName = Literal["coarse_to_fine", "random", "full_then_random"]


def make_viewpoints(
    policy: ViewpointPolicyName,
    batch_size: int,
    device: torch.device,
    n_viewpoints: int,
    *,
    min_scale: float = 0.05,
    max_scale: float = 1.0,
    start_with_full_scene: bool = True,
) -> list[Viewpoint]:
    if policy == "coarse_to_fine":
        return coarse_to_fine_viewpoints(batch_size, device, n_viewpoints)
    start_full = policy == "full_then_random" or start_with_full_scene
    return random_viewpoints(
        batch_size, device, n_viewpoints,
        min_scale=min_scale, max_scale=max_scale,
        start_with_full_scene=start_full,
    )
