#!/bin/bash
#SBATCH --job-name=ade20k-canvit
#SBATCH --account=def-skrishna_gpu
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/ade20k-canvit-%j.out
#SBATCH --error=logs/ade20k-canvit-%j.err

# CanViT ADE20K probe training (frozen backbone OR full fine-tuning).
#
# Frozen-backbone probe training (default):
#   sbatch slurm/train_ade20k_canvit.sbatch \
#     --scene-size 1024 --canvas-grid 64 --batch-size 16 --max-steps 40000
#
# Full fine-tuning (LP-FT) — start from converged frozen probe.
# Note: --weight-decay 1e-4 (10x lower than frozen probe default 1e-3) — high WD
# can destabilize the backbone during FT (per IN1K FT precedent).
#   sbatch slurm/train_ade20k_canvit.sbatch \
#     --scene-size 1024 --canvas-grid 64 --batch-size 16 --max-steps 40000 \
#     --finetune --init-probe-repo canvit/probe-ade20k-40k-s1024-c64-in21k \
#     --peak-lr 2.5e-5 --grad-clip 1.0 --weight-decay 1e-4

set -eux
source slurm/setup.sh
mkdir -p logs

echo "=== ADE20K CanViT Probe Training ==="
echo "Job: $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

exec uv run python -m canvit_probes.training.ade20k train "$@"
