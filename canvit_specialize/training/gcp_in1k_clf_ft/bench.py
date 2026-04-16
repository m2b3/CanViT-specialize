"""Throughput decomposition benchmarks for IN1K finetuning pipeline.

Three modes, each measures ONE axis in isolation. All import production code
(canvit_specialize.training.gcp_in1k_clf_ft.shared + canvit_pytorch) — never
copy-pasted — so the numbers reflect the exact code path training uses.

Usage (on v6e-4 SPMD):
    uv run python -m canvit_specialize.training.gcp_in1k_clf_ft.bench \
        --mode {dataloader,fwd,fwd_bwd} \
        --data-dir ~/gcs/datasets/imagenet \
        --batch-size 256 --n-glimpses 4 --num-workers 32 \
        --warmup-steps 20 --measure-steps 200
"""

import argparse
import logging
import time

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO, force=True)
log = logging.getLogger("bench")

import numpy as np
import torch
import torch.nn.functional as F
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from canvit_pytorch import Viewpoint
from torch_xla.distributed.spmd import Mesh

from .shared import CANVAS_GRID, make_multi_glimpse_dataloader, load_classifier


def bench_dataloader(args) -> None:
    """Pure dataloader throughput — no device transfer, no model."""
    loader = make_multi_glimpse_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_glimpses=args.n_glimpses,
        split="train",
        min_viewpoint_scale=args.min_viewpoint_scale,
    )
    it = iter(loader)

    log.info("dataloader: warmup %d batches", args.warmup_steps)
    for _ in range(args.warmup_steps):
        next(it)

    log.info("dataloader: measuring %d batches (B=%d, N=%d, workers=%d)",
             args.measure_steps, args.batch_size, args.n_glimpses, args.num_workers)
    t0 = time.perf_counter()
    total_scenes = 0
    for _ in range(args.measure_steps):
        glimpses, _labels, _vpc, _vps = next(it)
        total_scenes += glimpses.shape[1]
    elapsed = time.perf_counter() - t0

    sps = total_scenes / elapsed
    gps = sps * args.n_glimpses
    log.info("DATALOADER: %d scenes in %.1fs | %.0f sc/s | %.0f gl/s",
             total_scenes, elapsed, sps, gps)


def _setup_spmd() -> tuple[torch.device, Mesh | None]:
    n_devices = xr.global_runtime_device_count()
    if n_devices > 1:
        xr.use_spmd()
        mesh = Mesh(np.arange(n_devices), (n_devices,), ('data',))
        log.info("SPMD: %d devices, mesh=(%d,)", n_devices, n_devices)
    else:
        mesh = None
    return torch_xla.device(), mesh


def _make_dummy_batch(*, B: int, N: int, device, mesh: Mesh | None):
    """Synthetic batch with the SAME shape/dtype as the real dataloader."""
    glimpses = torch.randn(N, B, 3, 128, 128, device=device)
    labels = torch.randint(0, 1000, (B,), device=device)
    vp_centers = torch.randn(N, B, 2, device=device) * 0.5
    vp_scales = torch.rand(N, B, device=device) * 0.9 + 0.1
    if mesh is not None:
        xs.mark_sharding(glimpses, mesh, (None, 'data', None, None, None))
        xs.mark_sharding(labels, mesh, ('data',))
    return glimpses, labels, vp_centers, vp_scales


def _fwd_only(*, clf, batch_size, device, mesh) -> None:
    glimpses, _labels, vp_centers, vp_scales = _make_dummy_batch(
        B=batch_size, N=4, device=device, mesh=mesh)
    state = clf.init_state(batch_size=batch_size, canvas_grid_size=CANVAS_GRID)
    state.canvas = state.canvas.to(device)
    state.recurrent_cls = state.recurrent_cls.to(device)
    if mesh is not None:
        xs.mark_sharding(state.canvas, mesh, ('data', None, None))
        xs.mark_sharding(state.recurrent_cls, mesh, ('data', None, None))
    for g in range(4):
        vp = Viewpoint(centers=vp_centers[g], scales=vp_scales[g])
        _logits, state = clf(glimpse=glimpses[g], state=state, viewpoint=vp)


def _fwd_bwd(*, clf, optimizer, batch_size, device, mesh) -> None:
    glimpses, labels, vp_centers, vp_scales = _make_dummy_batch(
        B=batch_size, N=4, device=device, mesh=mesh)
    state = clf.init_state(batch_size=batch_size, canvas_grid_size=CANVAS_GRID)
    state.canvas = state.canvas.to(device)
    state.recurrent_cls = state.recurrent_cls.to(device)
    if mesh is not None:
        xs.mark_sharding(state.canvas, mesh, ('data', None, None))
        xs.mark_sharding(state.recurrent_cls, mesh, ('data', None, None))
    chunk_loss = torch.zeros((), device=device)
    for g in range(4):
        vp = Viewpoint(centers=vp_centers[g], scales=vp_scales[g])
        logits, state = clf(glimpse=glimpses[g], state=state, viewpoint=vp)
        chunk_loss = chunk_loss + F.cross_entropy(logits, labels)
    (chunk_loss / 4).backward()
    torch_xla.sync()
    optimizer.step()
    optimizer.zero_grad()
    torch_xla.sync()


def bench_compute(args, *, with_backward: bool) -> None:
    device, mesh = _setup_spmd()
    clf = load_classifier(device=device)
    optimizer = torch.optim.AdamW(clf.parameters(), lr=2.5e-5) if with_backward else None

    fn = (lambda: _fwd_bwd(clf=clf, optimizer=optimizer, batch_size=args.batch_size,
                           device=device, mesh=mesh)) if with_backward else \
         (lambda: _fwd_only(clf=clf, batch_size=args.batch_size, device=device, mesh=mesh))

    mode = "fwd_bwd" if with_backward else "fwd"
    log.info("%s: warmup %d steps (includes XLA compile)", mode, args.warmup_steps)
    t_warm = time.perf_counter()
    for _ in range(args.warmup_steps):
        fn()
    torch_xla.sync(wait=True)
    log.info("%s: warmup done [%.1fs]", mode, time.perf_counter() - t_warm)

    log.info("%s: measuring %d steps (B=%d, N=4)", mode, args.measure_steps, args.batch_size)
    t0 = time.perf_counter()
    for _ in range(args.measure_steps):
        fn()
    torch_xla.sync(wait=True)
    elapsed = time.perf_counter() - t0

    scenes = args.measure_steps * args.batch_size
    sps = scenes / elapsed
    gps = sps * 4
    log.info("%s: %d scenes in %.1fs | %.0f sc/s | %.0f gl/s",
             mode.upper(), scenes, elapsed, sps, gps)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["dataloader", "fwd", "fwd_bwd"], required=True)
    p.add_argument("--data-dir", default="~/gcs/datasets/imagenet")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-glimpses", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=32)
    p.add_argument("--min-viewpoint-scale", type=float, default=0.05)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--measure-steps", type=int, default=200)
    args = p.parse_args()
    if args.mode == "dataloader":
        bench_dataloader(args)
    elif args.mode == "fwd":
        bench_compute(args, with_backward=False)
    else:
        bench_compute(args, with_backward=True)


if __name__ == "__main__":
    main()
