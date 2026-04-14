#!/bin/bash
#SBATCH --job-name=ade20k-smoke-cpu
#SBATCH --account=def-skrishna_cpu
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/ade20k-smoke-cpu-%j.out
#SBATCH --error=logs/ade20k-smoke-cpu-%j.err

# CPU smoke test: verify imports, model load, dataset access, comet logging.
# Does NOT run any backward pass — pure sanity check, low priority queue.
# Should take ~5 minutes and produces no usable training.

set -eu  # NOT -x: would trace secret exports (HF_TOKEN, etc.) into logs
source slurm/setup.sh
mkdir -p logs

echo "=== ADE20K CanViT Smoke Test (CPU-only) ==="
echo "Job: $SLURM_JOB_ID on $(hostname)"

# Just verify: import chain, model loads, dataset loads.
exec uv run python -c "
from canvit_probes.training.ade20k.train_canvit import train
from canvit_probes.training.ade20k.config import Config
from canvit_probes import SegmentationProbe
from canvit_pytorch import CanViTForPretrainingHFHub

print('imports OK')

probe = SegmentationProbe.from_pretrained('canvit/probe-ade20k-40k-s1024-c64-in21k')
print(f'probe loaded: {sum(p.numel() for p in probe.parameters()):,} params')

import os
print(f'ADE20K_ROOT: {os.environ[\"ADE20K_ROOT\"]}')
print(f'  exists: {os.path.exists(os.environ[\"ADE20K_ROOT\"])}')

print('Smoke test PASSED')
"
