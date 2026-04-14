"""Push ADE20K segmentation probes to HuggingFace Hub.

Three modes:
  single    — push one probe with an explicit repo_id
  batch     — push every probe under a directory, auto-deriving repo_ids
  retrofit  — refresh the model card + collection note for probes already on
              HF (reads config.json from the Hub; does NOT require a .pt)

Each push (single/batch):
  1. Uploads model.safetensors + config.json (probe weights + training metadata)
  2. Uploads README.md (minimal model card, HP table derived from metadata)
  3. Upserts the repo into the CanViT ADE20K probe collection with a
     canonical note — no parenthetical drift.

Usage:
    uv run python scripts/push_probes.py single \
        --probe PATH --repo-id canvit/probe-ade20k-... [--public] [--dry-run]

    uv run python scripts/push_probes.py batch \
        --probe-dir PATH [--owner canvit] [--public] [--dry-run]

    uv run python scripts/push_probes.py retrofit \
        --repo-ids canvit/probe-ade20k-... canvit/probe-ade20k-... [--dry-run]
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from huggingface_hub import HfApi, hf_hub_download

from canvit_pytorch.probes import SegmentationProbe
from scripts.upload_utils import (
    json_sanitize,
    upload_model_card,
    upload_probe_to_hub,
    upsert_collection_item,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# HF collection slugs are stable identifiers; safe to hardcode.
CANVAS_PROBE_COLLECTION = "canvit/canvit-ade20k-segmentation-probes-69d550b66add770c509bb77a"
DINOV3_PROBE_COLLECTION = "canvit/dinov3-ade20k-segmentation-probes-69d59b1eb69bbb3422f49b4f"

# Batch-mode repo naming: short id per base-model repo. Full name lives in config.json.
# Old + current flagship CanViT-B names resolve to the same weights (same HF rename).
_MODEL_SHORT: dict[str, str] = {
    "facebook/dinov3-vitb16-pretrain-lvd1689m": "dv3b",
    "facebook/dinov3-vits16-pretrain-lvd1689m": "dv3s",
    "canvit/canvit-vitb16-pretrain-512px-in21k": "in21k",
    "canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02": "in21k",
}


# ---------- Checkpoint loading ----------

def _extract_state_dict_and_metadata(raw: dict) -> tuple[dict, dict]:
    """Supports canvas (new + legacy) and DINOv3 probe checkpoint formats."""
    if "feat_type" in raw:
        sd = raw["probe_state_dict"]
        meta = {k: v for k, v in raw.items() if k != "probe_state_dict"}
        return sd, meta
    if "probe_state_dicts" in raw:
        sd = raw["probe_state_dicts"]["canvas_hidden"]
        meta = {k: v for k, v in raw.items() if k != "probe_state_dicts"}
        meta["feat_type"] = "canvas_hidden"
        meta["_legacy_format"] = True
        return sd, meta
    if "probe_state_dict" in raw:
        sd = raw["probe_state_dict"]
        meta = {k: v for k, v in raw.items() if k != "probe_state_dict"}
        meta["feat_type"] = "dinov3_spatial"
        return sd, meta
    raise AssertionError(f"Unknown probe checkpoint format. Keys: {sorted(raw.keys())}")


def _assert_not_finetune(raw: dict, name: str) -> None:
    """LP-FT checkpoints carry full CanViT weights; pushing via this script
    would silently drop them. Refuse until a paired-publish path exists."""
    if "model_state_dict" in raw or raw.get("config", {}).get("finetune") is True:
        raise NotImplementedError(
            f"{name} is an LP-FT checkpoint with full CanViT weights. "
            f"push_probes.py only handles standalone probe heads; pushing "
            f"would silently drop the CanViT state_dict."
        )


def load_probe(pt_path: Path) -> tuple[SegmentationProbe, dict]:
    """Return (probe, metadata). Probe weights strict-loaded and validated."""
    raw = torch.load(pt_path, map_location="cpu", weights_only=False)
    _assert_not_finetune(raw, pt_path.name)
    sd, meta = _extract_state_dict_and_metadata(raw)

    embed_dim = sd["conv.weight"].shape[1]
    num_classes = sd["conv.weight"].shape[0]
    use_ln = "ln.weight" in sd
    dropout = meta.get("config", {}).get("dropout")
    assert dropout is not None, f"dropout missing in {pt_path}"

    probe = SegmentationProbe(
        embed_dim=embed_dim, num_classes=num_classes,
        dropout=dropout, use_ln=use_ln,
    )
    result = probe.load_state_dict(sd, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    return probe, meta


# ---------- Batch-mode repo naming ----------

def derive_repo_id(owner: str, meta: dict, probe_name: str) -> str:
    cfg = meta.get("config", {})
    steps_k = cfg["max_steps"] // 1000

    if meta.get("feat_type") == "dinov3_spatial":
        short = _MODEL_SHORT[cfg["model"]]
        return f"{owner}/probe-ade20k-{steps_k}k-{short}-{cfg['resolution']}px"

    short = _MODEL_SHORT.get(cfg["model_repo"])
    assert short is not None, (
        f"Unknown model_repo {cfg['model_repo']!r} — add to _MODEL_SHORT."
    )
    return f"{owner}/probe-ade20k-{steps_k}k-s{cfg['scene_size']}-c{cfg['canvas_grid']}-{short}"


# ---------- Canonical collection note ----------

def canvas_probe_note(cfg: dict) -> str:
    return f"CanViT, {cfg['scene_size']}px scene, {cfg['canvas_grid']}×{cfg['canvas_grid']} canvas grid"


def dinov3_probe_note(cfg: dict) -> str:
    m = re.search(r"dinov3-vit(b|s|l|h)(\d+)", cfg["model"])
    assert m is not None, f"Cannot parse DINOv3 variant from {cfg['model']!r}"
    variant = m.group(1).upper()
    return f"DINOv3 ViT-{variant}/{m.group(2)}, {cfg['resolution']}px input"


# ---------- Model card ----------

_CARD_TEMPLATE = """\
---
license: mit
library_name: canvit-pytorch
pipeline_tag: image-segmentation
tags:
  - canvit
  - active-vision
  - ade20k
  - segmentation-probe
datasets:
  - scene_parse_150
base_model: {base_model}
---

# {title}

Linear segmentation probe on the canvas features of
[{base_model}](https://huggingface.co/{base_model}).

- **Paper**: [arXiv:2603.22570](https://arxiv.org/abs/2603.22570)
- **Training code**: [github.com/m2b3/CanViT-specialize](https://github.com/m2b3/CanViT-specialize)

## Usage

```bash
uv add "canvit-pytorch @ git+https://github.com/m2b3/CanViT-PyTorch.git"
```

```python
import torch
from canvit_pytorch.probes import SegmentationProbe

probe = SegmentationProbe.from_pretrained("{repo_id}").eval()

# [B, H, W, D] canvas features from a CanViT forward pass
features = torch.randn(1, {grid}, {grid}, {embed_dim})
with torch.inference_mode():
    logits = probe(features)    # [B, num_classes, H, W]
assert logits.shape == (1, {num_classes}, {grid}, {grid})
```

## Training

Architecture: `LayerNorm → Dropout → BatchNorm → Conv1×1`.

{hp_table}
"""


def _policy_label(cfg: dict) -> str:
    """R-IID / F-IID per paper, from `train_start_full`.

    Probe training currently hardcodes the "random" policy family in
    train_canvit.py; under that assumption, the only axis that distinguishes
    R-IID from F-IID is whether t=0 is full-scene. If a non-random training
    policy is ever added, revisit this helper.
    """
    return "F-IID" if cfg.get("train_start_full") else "R-IID"


def _precision_label(cfg: dict) -> str:
    return "bf16 (AMP)" if cfg.get("amp") else "fp32"


def _canvas_hp_table(cfg: dict) -> str:
    scene, grid = cfg["scene_size"], cfg["canvas_grid"]
    scale_lo, scale_hi = cfg["aug_scale_range"]
    rows = [
        ("Scene size", f"{scene} px"),
        ("Canvas grid", f"{grid} × {grid}"),
        ("Glimpse size", f"{cfg['glimpse_px']} px"),
        ("Timesteps (T)", str(cfg["n_timesteps"])),
        ("Training policy", _policy_label(cfg)),
        ("Optimizer", "AdamW"),
        ("Peak LR", f"{cfg['peak_lr']:g}"),
        ("Weight decay", f"{cfg['weight_decay']:g}"),
        ("LR schedule", f"{cfg['warmup_steps']:,}-step warmup → cosine decay"),
        ("Batch size", str(cfg["batch_size"])),
        ("Max steps", f"{cfg['max_steps']:,}"),
        ("Dropout", str(cfg["dropout"])),
        ("Augmentation", f"RandomResizedCrop scale [{scale_lo:g}, {scale_hi:g}] + HFlip"),
        ("Precision", _precision_label(cfg)),
    ]
    lines = ["| Hyperparameter | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    return "\n".join(lines)


def build_canvas_card(
    repo_id: str, *, embed_dim: int, num_classes: int, cfg: dict,
) -> str:
    scene, grid = cfg["scene_size"], cfg["canvas_grid"]
    return _CARD_TEMPLATE.format(
        base_model=cfg["model_repo"],
        title=f"ADE20K Segmentation Probe — canvas {grid}×{grid} @ {scene}px scene",
        repo_id=repo_id,
        grid=grid,
        embed_dim=embed_dim,
        num_classes=num_classes,
        hp_table=_canvas_hp_table(cfg),
    )


# ---------- Publish pipeline ----------

def publish_probe(
    probe: SegmentationProbe, meta: dict, repo_id: str,
    *, public: bool, dry_run: bool,
) -> None:
    feat_type = meta.get("feat_type")
    cfg = meta.get("config", {})
    embed_dim = probe.embed_dim
    num_classes = probe.num_classes

    if feat_type == "dinov3_spatial":
        # DINOv3 probe card is not yet templated; uploads-only path preserved.
        collection, card, note = DINOV3_PROBE_COLLECTION, None, dinov3_probe_note(cfg)
    else:
        collection = CANVAS_PROBE_COLLECTION
        card = build_canvas_card(repo_id, embed_dim=embed_dim, num_classes=num_classes, cfg=cfg)
        note = canvas_probe_note(cfg)

    hf_config = {
        "embed_dim": embed_dim,
        "num_classes": num_classes,
        "dropout": probe.dropout_p,
        "use_ln": probe.use_ln,
        "metadata": json_sanitize(meta),
    }

    log.info("  → %s  visibility=%s", repo_id, "public" if public else "private")
    log.info("     note:       %s", note)
    log.info("     model card: %s", "yes" if card is not None else "no (DINOv3 — TODO)")
    if dry_run:
        return

    upload_probe_to_hub(
        state_dict=probe.state_dict(), config=hf_config,
        repo_id=repo_id, private=not public,
    )
    if card is not None:
        upload_model_card(repo_id=repo_id, card_text=card)
    upsert_collection_item(collection, repo_id, note=note)


# ---------- Entry points ----------

@dataclass
class Single:
    """Push one probe checkpoint with an explicit repo_id."""
    probe: Path
    repo_id: str
    public: bool = False
    dry_run: bool = False


@dataclass
class Batch:
    """Push every probe under `probe_dir`, auto-deriving repo_ids."""
    probe_dir: Path
    owner: str = "canvit"
    public: bool = False
    dry_run: bool = False


@dataclass
class Retrofit:
    """Refresh model card + collection note for probes already on HF.

    Reads config.json from each repo, re-sanitizes it (older pushes
    wrote bare `Infinity` tokens — invalid JSON, rejected by strict
    parsers including HF's own config viewer), regenerates the card and
    collection note with the current template, and uploads anything
    that changed. Idempotent in the sense of: the huggingface_hub
    client skips empty commits when the uploaded content is identical
    to HEAD, so re-running against a fresh repo is a no-op.
    """
    repo_ids: list[str]
    dry_run: bool = False


def _find_best_pt(d: Path) -> Path:
    candidates = list(d.glob("*best*miou*.pt"))
    assert len(candidates) >= 1, f"No best checkpoint in {d}"
    def _miou(p: Path) -> float:
        m = re.search(r"miou([\d.]+)", p.name)
        return float(m.group(1)) if m else 0.0
    return max(candidates, key=_miou)


def _run_single(args: Single) -> None:
    assert args.probe.exists(), f"Not found: {args.probe}"
    probe, meta = load_probe(args.probe)
    publish_probe(probe, meta, args.repo_id,
                  public=args.public, dry_run=args.dry_run)


def _run_batch(args: Batch) -> None:
    assert args.probe_dir.is_dir(), f"Not a directory: {args.probe_dir}"
    probe_dirs = sorted(d for d in args.probe_dir.iterdir() if d.is_dir())
    log.info("%s %d probe directories in %s",
             "DRY RUN:" if args.dry_run else "Pushing", len(probe_dirs), args.probe_dir)

    for d in probe_dirs:
        try:
            best_pt = _find_best_pt(d)
        except AssertionError as e:
            log.info("  SKIP %s: %s", d.name, e)
            continue
        probe, meta = load_probe(best_pt)
        cfg = meta.get("config", {})
        if cfg.get("max_steps") != 40000:
            log.info("  SKIP %s (max_steps=%s, want 40000)", d.name, cfg.get("max_steps"))
            continue
        repo_id = derive_repo_id(args.owner, meta, d.name)
        publish_probe(probe, meta, repo_id,
                      public=args.public, dry_run=args.dry_run)
    log.info("Done.")


def _run_retrofit(args: Retrofit) -> None:
    api = HfApi()
    for repo_id in args.repo_ids:
        log.info("Retrofitting %s", repo_id)
        raw_text = Path(hf_hub_download(repo_id, "config.json")).read_text()
        cfg_json = json.loads(raw_text)
        meta = cfg_json.get("metadata", {})
        train_cfg = meta.get("config", {})
        feat_type = meta.get("feat_type", "canvas_hidden")

        if feat_type == "dinov3_spatial":
            log.info("  (DINOv3 probe: retrofit template not yet implemented, skipping)")
            continue

        card = build_canvas_card(
            repo_id,
            embed_dim=cfg_json["embed_dim"],
            num_classes=cfg_json["num_classes"],
            cfg=train_cfg,
        )
        note = canvas_probe_note(train_cfg)
        # Re-sanitize: old pushes wrote bare `Infinity`; make it strict-JSON.
        fresh_cfg_text = json.dumps(json_sanitize(cfg_json), indent=2)
        cfg_needs_rewrite = fresh_cfg_text != raw_text

        log.info("  note: %s", note)
        log.info("  card: %d chars", len(card))
        log.info("  config.json rewrite: %s", "yes (sanitized)" if cfg_needs_rewrite else "no (clean)")
        if args.dry_run:
            continue

        upload_model_card(repo_id=repo_id, card_text=card)
        upsert_collection_item(CANVAS_PROBE_COLLECTION, repo_id, note=note)
        if cfg_needs_rewrite:
            api.upload_file(
                path_or_fileobj=fresh_cfg_text.encode(),
                path_in_repo="config.json",
                repo_id=repo_id,
                commit_message="Sanitize config.json (Infinity → \"inf\", Paths → str)",
            )
    log.info("Done.")


def main() -> None:
    args = tyro.cli(Single | Batch | Retrofit)
    if isinstance(args, Single):
        _run_single(args)
    elif isinstance(args, Batch):
        _run_batch(args)
    else:
        _run_retrofit(args)


if __name__ == "__main__":
    main()
