"""ADE20K canvas probe training.

Trains segmentation probes on frozen CanViT features:
- ONE probe per feature type, shared weights across timesteps (anytime decoding)
- Training: loss averaged across timesteps, single backward pass
- Eval: mIoU computed per timestep, logged as curves

Training protocol aligned with DINOv3's linear probing (Appendix D.1).
Note: We use whole-image inference, not sliding window (intentional simplification).
"""

import logging
import os
import time
from dataclasses import asdict
from pathlib import Path

import comet_ml
import torch
import torch.nn as nn
import tyro
from canvit_pytorch import CanViTForPretrainingHFHub
from canvit_pytorch.teacher import load_teacher
from torch import Tensor
from tqdm import tqdm

from canvit_probes.training.ade20k.viz import log_viz

from canvit_probes.datasets.ade20k import IGNORE_LABEL, NUM_CLASSES
from canvit_probes.training.ade20k.eval_utils import eval_probe_on_batch
from canvit_probes import SegmentationProbe
from canvit_probes.metrics import IoUAccumulator
from canvit_probes.training.utils import make_viewpoints

from canvit_probes.training.ade20k.common import make_ade20k_loaders, make_amp_ctx, make_optimizer_and_scheduler
from canvit_probes.training.ade20k.config import CANVAS_FEATURES, CanvasFeatureType, Config, get_feature_dims
from canvit_probes.training.ade20k.features import CanvasFeatures, extract_canvas_features
from canvit_probes.training.ade20k.loss import ce_loss, upsample_preds
from canvit_probes.training.ade20k.state import ProbeState

log = logging.getLogger(__name__)


def _make_probe(name: str, dim: int, cfg: Config, device: torch.device, *, use_ln: bool) -> ProbeState:
    head = SegmentationProbe(embed_dim=dim, num_classes=NUM_CLASSES, dropout=cfg.dropout, use_ln=use_ln).to(device)
    opt, scheduler = make_optimizer_and_scheduler(
        head.parameters(), lr=cfg.peak_lr, weight_decay=cfg.weight_decay,
        max_steps=cfg.max_steps, warmup_steps=cfg.warmup_steps, warmup_lr_ratio=cfg.warmup_lr_ratio,
    )
    return ProbeState(name, head, opt, scheduler)


def _save_probe_checkpoint(
    run_dir: Path,
    feat_type: CanvasFeatureType,
    probe: ProbeState,
    step: int,
    cfg: Config,
    *,
    is_best: bool,
) -> Path:
    t_last = cfg.n_timesteps - 1
    miou = probe.best_last_miou
    prefix = f"{feat_type}_best_t{t_last}_miou{miou:.4f}_step{step}" if is_best else f"{feat_type}_final_step{step}"
    filename = f"{prefix}.pt"
    path = run_dir / filename
    tmp_path = run_dir / f".{filename}.tmp"
    data = {
        "step": step,
        "feat_type": feat_type,
        "probe_state_dict": probe.head.state_dict(),
        "best_mious_per_t": probe.best_mious,
        "config": asdict(cfg),
    }
    run_dir.mkdir(parents=True, exist_ok=True)

    if is_best:
        for old in run_dir.glob(f"{feat_type}_best_*.pt"):
            old.unlink()

    torch.save(data, tmp_path)
    tmp_path.rename(path)
    log.info(f"Saved checkpoint: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def train(cfg: Config) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_float32_matmul_precision("high")
    device = torch.device(cfg.device)

    log.info("=" * 60)
    log.info("ADE20K Canvas Probe Training")
    log.info("=" * 60)
    log.info(f"Model: {cfg.model_repo}")
    log.info(f"Features: {cfg.features}")
    log.info(f"Timesteps: {cfg.n_timesteps}")
    log.info(f"Viewpoint: scale=[{cfg.min_vp_scale}, {cfg.max_vp_scale}], train_start_full={cfg.train_start_full}")
    log.info(f"Training: BS={cfg.batch_size}, steps={cfg.max_steps}, LR={cfg.peak_lr}")

    # Model
    log.info("Loading model...")
    model = CanViTForPretrainingHFHub.from_pretrained(cfg.model_repo).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log.info(f"  params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    teacher = load_teacher(cfg.teacher_repo, device)
    log.info(f"  teacher: {cfg.teacher_repo}, dim={teacher.embed_dim}")

    patch_size = model.backbone.patch_size_px
    canvas_grid = cfg.canvas_grid if cfg.canvas_grid is not None else cfg.scene_size // patch_size
    cfg.canvas_grid = canvas_grid  # resolve for logging/checkpointing
    log.info(f"  scene: {cfg.scene_size}px, canvas: {canvas_grid}x{canvas_grid}, glimpse: {cfg.glimpse_px}px")

    # Probes
    dims = get_feature_dims(model.canvas_dim, teacher.embed_dim)
    probes: dict[CanvasFeatureType, ProbeState] = {
        feat: _make_probe(feat, dims[feat], cfg, device, use_ln=CANVAS_FEATURES[feat].needs_ln)
        for feat in cfg.features
    }
    for feat, probe in probes.items():
        probe.init_best_mious(cfg.n_timesteps)
        log.info(f"  probe[{feat}]: dim={dims[feat]}, params={sum(p.numel() for p in probe.head.parameters()):,}")

    # IoU metrics
    val_iou = {
        feat: [IoUAccumulator(NUM_CLASSES, IGNORE_LABEL, device) for _ in range(cfg.n_timesteps)]
        for feat in cfg.features
    }
    train_iou = {
        feat: [IoUAccumulator(NUM_CLASSES, IGNORE_LABEL, device) for _ in range(cfg.n_timesteps)]
        for feat in cfg.features
    }

    # Data
    train_loader, val_loader = make_ade20k_loaders(cfg)

    # Comet
    exp = comet_ml.Experiment(project_name=cfg.comet_project, workspace=cfg.comet_workspace)
    feats_str = "+".join(cfg.features)
    model_slug = cfg.model_repo.split("/")[-1]
    ts = time.strftime("%Y-%m-%d-%H%M%S-%Z")
    exp_name = f"canvit_{model_slug}_{feats_str}_{cfg.n_timesteps}t_{cfg.glimpse_px}g_s{cfg.scene_size}_c{canvas_grid}_{ts}"
    exp.set_name(exp_name)
    exp.log_parameters(asdict(cfg))
    exp.add_tag("canvas-probe")
    exp.add_tag(model_slug)
    exp.add_tag(f"s{cfg.scene_size}_c{canvas_grid}")
    log.info(f"Comet: {cfg.comet_workspace}/{cfg.comet_project}/{exp.get_key()} ({exp_name})")

    job_id = os.environ.get("SLURM_JOB_ID", "local")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dirname = f"{model_slug}_{timestamp}_{job_id}_{exp.get_key()[:8]}"
    run_dir = cfg.probe_ckpt_dir / dirname if cfg.probe_ckpt_dir else None
    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Checkpoints: {run_dir}")

    amp_ctx = make_amp_ctx(cfg.amp, device)

    log.info("=" * 60)
    log.info("Starting training...")

    step = 0
    train_iter = iter(train_loader)
    pbar = tqdm(total=cfg.max_steps, desc="Training")
    val_viz_batch: tuple[Tensor, Tensor, CanvasFeatures] | None = None

    while step < cfg.max_steps:
        try:
            images, masks = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, masks = next(train_iter)
        images, masks = images.to(device), masks.to(device)

        # === Validation ===
        if step % cfg.val_every == 0:
            val_start = time.perf_counter()
            for p in probes.values():
                p.head.eval()
            for feat in cfg.features:
                for m in val_iou[feat]:
                    m.reset()

            first_val_batch = True
            with torch.no_grad():
                for vi, vm in val_loader:
                    vi, vm = vi.to(device), vm.to(device)
                    B = vi.shape[0]
                    viewpoints = make_viewpoints(
                        "random", B, device, cfg.n_timesteps,
                        min_scale=cfg.min_vp_scale, max_scale=cfg.max_vp_scale,
                        start_with_full_scene=True,
                    )
                    with amp_ctx:
                        val_feats = extract_canvas_features(
                            model=model, images=vi,
                            canvas_grid=canvas_grid, glimpse_px=cfg.glimpse_px,
                            viewpoints=viewpoints,
                        )
                    if first_val_batch:
                        val_viz_batch = (vi, vm, val_feats)
                        first_val_batch = False
                    for feat_type in cfg.features:
                        for t in range(cfg.n_timesteps):
                            feat_t = val_feats.get(feat_type, t)
                            eval_probe_on_batch(
                                probes[feat_type].head, feat_t, vm, val_iou[feat_type][t],
                            )

            for feat_type in cfg.features:
                mious = [val_iou[feat_type][t].compute() for t in range(cfg.n_timesteps)]
                improved = probes[feat_type].update_best(mious)

                for t, miou in enumerate(mious):
                    exp.log_metric(f"{feat_type}/val_miou_t{t}", miou, step=step)
                    exp.log_metric(f"{feat_type}/best_val_miou_t{t}", probes[feat_type].best_mious[t], step=step)
                exp.log_curve(f"{feat_type}/val_miou_curve", x=list(range(cfg.n_timesteps)), y=mious, step=step)

                if improved and run_dir:
                    _save_probe_checkpoint(run_dir, feat_type, probes[feat_type], step, cfg, is_best=True)

            val_time = time.perf_counter() - val_start
            log.info(f"Step {step}: validation took {val_time:.1f}s")
            exp.log_metric("timing/val_seconds", val_time, step=step)

        # === Training step ===
        for p in probes.values():
            p.head.train()

        B = images.shape[0]
        viewpoints = make_viewpoints(
            "random", B, device, cfg.n_timesteps,
            min_scale=cfg.min_vp_scale, max_scale=cfg.max_vp_scale,
            start_with_full_scene=cfg.train_start_full,
        )
        with amp_ctx:
            feats = extract_canvas_features(
                model=model, images=images,
                canvas_grid=canvas_grid, glimpse_px=cfg.glimpse_px,
                viewpoints=viewpoints,
            )

        for feat_type in cfg.features:
            probe = probes[feat_type]
            probe.optimizer.zero_grad()
            logits_list = [probe.head(feats.get(feat_type, t).float()) for t in range(cfg.n_timesteps)]
            losses = [ce_loss(logits, masks) for logits in logits_list]
            loss = torch.stack(losses).mean()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(probe.head.parameters(), cfg.grad_clip)
            probe.optimizer.step()
            probe.scheduler.step()
            probe.accumulate(loss, grad_norm)

            with torch.no_grad():
                for t, logits in enumerate(logits_list):
                    preds = logits.detach().argmax(1)
                    preds_up = upsample_preds(preds, masks.shape[1], masks.shape[2])
                    train_iou[feat_type][t].update(preds_up, masks)

        # === Visualization ===
        if step % cfg.viz_every == 0:
            viz_start = time.perf_counter()
            for p in probes.values():
                p.head.eval()
            with torch.no_grad():
                log_viz(exp, step, probes, feats, images, masks, cfg.viz_samples, cfg.n_timesteps, split="train")
                if val_viz_batch is not None:
                    v_img, v_mask, v_feats = val_viz_batch
                    log_viz(exp, step, probes, v_feats, v_img, v_mask, cfg.viz_samples, cfg.n_timesteps, split="val")
            viz_time = time.perf_counter() - viz_start
            log.info(f"Step {step}: viz took {viz_time:.1f}s")
            exp.log_metric("timing/viz_seconds", viz_time, step=step)

        step += 1
        pbar.update(1)

        # === Logging ===
        if step % cfg.log_every == 0:
            lr = list(probes.values())[0].scheduler.get_last_lr()[0]
            log_dict: dict[str, float] = {"lr": float(lr)}
            for name, p in probes.items():
                avg_loss, avg_grad = p.get_and_reset()
                log_dict[f"{name}/loss"] = avg_loss
                log_dict[f"{name}/grad_norm"] = avg_grad

            log_curves = (step % cfg.val_every == 0)
            for feat_type in cfg.features:
                mious = [train_iou[feat_type][t].compute() for t in range(cfg.n_timesteps)]
                log_dict[f"{feat_type}/train_miou_mean"] = sum(mious) / len(mious)
                if log_curves:
                    exp.log_curve(f"{feat_type}/train_miou_curve", x=list(range(cfg.n_timesteps)), y=mious, step=step)
                for m in train_iou[feat_type]:
                    m.reset()

            exp.log_metrics(log_dict, step=step)

    pbar.close()

    if run_dir:
        for feat_type, probe in probes.items():
            _save_probe_checkpoint(run_dir, feat_type, probe, step, cfg, is_best=False)

    log.info("=" * 60)
    log.info("Training complete. Best val mIoU per timestep:")
    for name, p in probes.items():
        for t, v in enumerate(p.best_mious):
            exp.log_metric(f"best/{name}_t{t}", v)
        log.info(f"  {name}: t0={p.best_mious[0]:.4f} ... t{p.n_timesteps-1}={p.best_last_miou:.4f}")
    log.info("=" * 60)


def main() -> None:
    cfg = tyro.cli(Config)
    train(cfg)
