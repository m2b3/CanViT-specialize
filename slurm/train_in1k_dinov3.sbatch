#!/bin/bash
#SBATCH --job-name=cls-probe
#SBATCH --account=rrg-skrishna_gpu
#SBATCH --time=0-01:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/cls-probe-%j.out
#SBATCH --error=logs/cls-probe-%j.err

# IN1K classification probe training with Optuna HP search.
# Expects extracted CLS features to exist at $FEATURES_DIR.
#
# Usage:
#   sbatch slurm/train_in1k_dinov3.sbatch
#   sbatch slurm/train_in1k_dinov3.sbatch --model-repo facebook/dinov3-vitb16-pretrain-lvd1689m
#   sbatch slurm/train_in1k_dinov3.sbatch --n-trials 200

set -eu
source slurm/setup.sh
mkdir -p logs

echo "=== IN1K Classification Probe Training (Optuna) ==="
echo "Job: $SLURM_JOB_ID on $(hostname)"

exec uv run python -m canvit_specialize.training.in1k train "$@"
