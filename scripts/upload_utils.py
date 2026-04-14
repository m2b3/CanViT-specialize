"""Shared HuggingFace Hub upload logic for probe + finetuned-backbone push scripts."""

import json
import logging
import os
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
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
    """Upload a probe checkpoint to HuggingFace Hub. Returns the repo URL.

    Refuses to serialize tensors into config.json — the previous
    `default=str` behavior would silently coerce nested tensors to
    truncated repr strings, producing a useless config and dropping the
    actual weights. Caller must pass a JSON-clean config.
    """
    _assert_json_clean(config)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath / "config.json").write_text(json.dumps(config, indent=2))
        save_file(state_dict, tmppath / "model.safetensors")

        api = HfApi()
        api.create_repo(repo_id, private=private, exist_ok=True)
        api.upload_folder(folder_path=tmpdir, repo_id=repo_id)

    url = f"https://huggingface.co/{repo_id}"
    log.info("Pushed to %s", url)
    return url


def _assert_json_clean(obj: object, path: str = "config") -> None:
    """Refuse non-JSON-native types early. Prevents the `default=str` foot-gun
    that previously could str-coerce torch.Tensor values into config.json."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _assert_json_clean(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _assert_json_clean(v, f"{path}[{i}]")
    elif obj is None or isinstance(obj, (bool, int, float, str)):
        return
    else:
        raise TypeError(
            f"Non-JSON value at {path}: type={type(obj).__name__}. "
            f"Filter or stringify explicitly before passing to upload_probe_to_hub."
        )


def pull_comet_params(experiment_key: str) -> dict[str, str]:
    """Pull deduplicated hyperparameters from a Comet experiment.

    Used to augment HF config.json with training provenance, matching the
    pattern in lamarck-infra/demo-pytorch/push_to_hub.py — grep there for
    `pull_comet_params` or the `api.get_experiment_by_key` call.
    """
    import comet_ml
    api = comet_ml.API()
    exp = api.get_experiment_by_key(experiment_key)
    assert exp is not None, f"Comet experiment {experiment_key!r} not found"
    raw = {p["name"]: p["valueCurrent"] for p in exp.get_parameters_summary()}
    # Comet logs both dash- and underscore-cased duplicates — keep one form.
    return {k.replace("-", "_"): v for k, v in raw.items()}


def augment_hf_config_with_comet(
    repo_id: str,
    comet_experiment_key: str,
    extra: dict | None = None,
) -> None:
    """Download config.json from `repo_id`, add `training` block with Comet
    HPs (and optional `extra` fields), re-upload. Skips silently if
    COMET_API_KEY is not set in the environment."""
    if not os.environ.get("COMET_API_KEY"):
        log.info("Skipping Comet metadata augmentation (no COMET_API_KEY in env)")
        return
    log.info("Pulling training HPs from Comet experiment %s...", comet_experiment_key)
    cfg_path = hf_hub_download(repo_id, "config.json")
    cfg = json.loads(Path(cfg_path).read_text())
    cfg["training"] = {
        "comet_experiment_key": comet_experiment_key,
        "params": pull_comet_params(comet_experiment_key),
        **(extra or {}),
    }
    HfApi().upload_file(
        path_or_fileobj=json.dumps(cfg, indent=2).encode(),
        path_in_repo="config.json",
        repo_id=repo_id,
        commit_message="Add training metadata from Comet",
    )
    log.info("config.json on %s augmented with Comet params", repo_id)
