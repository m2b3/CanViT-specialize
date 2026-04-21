#!/bin/bash
#SBATCH --job-name=ade20k-canvit
#SBATCH --account=def-skrishna_gpu
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/ade20k-canvit-%j.out
#SBATCH --error=logs/ade20k-canvit-%j.err

set -eu  # NOT -x: would trace secret exports (HF_TOKEN, etc.) into logs
source slurm/setup.sh
mkdir -p logs

echo "=== ADE20K CanViT Probe Training ==="
echo "Job: $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

exec uv run python -m canvit_specialize.training.ade20k train "$@"
