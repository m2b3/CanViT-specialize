#!/bin/bash
#SBATCH --job-name=ade20k-canvit
#SBATCH --account=def-skrishna_gpu
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/ade20k-canvit-%j.out
#SBATCH --error=logs/ade20k-canvit-%j.err

# CanViT ADE20K probe training (frozen backbone).
#
#   sbatch slurm/train_ade20k_canvit.sbatch \
#     --scene-size 1024 --canvas-grid 64 --batch-size 16 --max-steps 40000
#
# LP-FT (--finetune) is supported by the CLI but no audited example is provided —
# the hyperparameters used for any full-fine-tuning run must be chosen by the
# operator based on the checkpoint they're initializing from.

set -eu  # NOT -x: would trace secret exports (HF_TOKEN, etc.) into logs
source slurm/setup.sh
mkdir -p logs

echo "=== ADE20K CanViT Probe Training ==="
echo "Job: $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

exec uv run python -m canvit_specialize.training.ade20k train "$@"
