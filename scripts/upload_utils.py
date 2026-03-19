"""Shared HuggingFace Hub upload logic for probe push scripts."""

import json
import logging
import tempfile
from pathlib import Path

from huggingface_hub import HfApi
from safetensors.torch import save_file
from torch import Tensor

log = logging.getLogger(__name__)


def upload_probe_to_hub(
    *,
    state_dict: dict[str, Tensor],
    config: dict,
    repo_id: str,
    private: bool = True,
) -> str:
    """Upload a probe checkpoint to HuggingFace Hub. Returns the repo URL."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath / "config.json").write_text(
            json.dumps(config, indent=2, default=str)
        )
        save_file(state_dict, tmppath / "model.safetensors")

        api = HfApi()
        api.create_repo(repo_id, private=private, exist_ok=True)
        api.upload_folder(folder_path=tmpdir, repo_id=repo_id)

    url = f"https://huggingface.co/{repo_id}"
    log.info("Pushed to %s", url)
    return url
