"""Naming conventions for DINOv3 IN1K probe artifacts."""

from pathlib import Path


def model_name_from_repo(repo: str) -> str:
    """'facebook/dinov3-vitb16-pretrain-lvd1689m' → 'dinov3_vitb16'."""
    slug = repo.split("/")[-1]
    assert "-pretrain" in slug, f"Expected '-pretrain' in repo slug: {repo}"
    name = "_".join(slug.split("-pretrain")[0].split("-"))
    assert name.startswith("dinov3_"), f"Parsed name {name!r} doesn't look like a DINOv3 model (from {repo})"
    return name


def features_split_dir(features_dir: Path, *, model_repo: str, image_size: int, split: str) -> Path:
    return features_dir / model_name_from_repo(model_repo) / str(image_size) / split
