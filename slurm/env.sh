# SLURM job environment setup. Sources .envrc, then sets up venv on $SLURM_TMPDIR.
# Source from sbatch scripts after `cd` to repo root.

source .envrc

echo "[env] Setting up SLURM environment..."
export PATH=$HOME/.local/bin:$PATH

# Venv on fast local SSD; cache stays at default (~/.cache/uv/) for persistence across jobs.
if [ -n "$SLURM_TMPDIR" ]; then
    export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv"
    echo "[env] Venv on SLURM_TMPDIR ($UV_PROJECT_ENVIRONMENT), uv cache at default"
else
    echo "[env] No SLURM_TMPDIR — running outside SLURM (interactive?)"
fi

uv sync
if [ -n "$UV_PROJECT_ENVIRONMENT" ]; then
    source "$UV_PROJECT_ENVIRONMENT/bin/activate"
fi

echo "[env] Done"
