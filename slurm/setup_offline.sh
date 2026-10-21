#!/bin/bash
# Install the training libraries from wheel files already copied to the HPC.
# The HPC has no internet on its compute nodes, so the wheels were downloaded
# on a Mac (see docs/HPC_SETUP.md). Run it as a scheduled job, not on the login node:
#
#     srun -p defq -n 1 -c 4 --mem=16G --time=00:30:00 bash slurm/setup_offline.sh
#
# Run from the repo folder, on coe-hpc3 (g17).

set -eo pipefail
cd "$(dirname "$0")/.."
command -v module >/dev/null 2>&1 || source /etc/profile.d/lmod.sh
module load python3/3.12.12

WHEELS="${WHEELS:-$HOME/wheels}"
echo "Using $(python3 --version), $(uname -m), wheels in $WHEELS ($(ls "$WHEELS" | wc -l) files)"

python3 -m venv .venv
source .venv/bin/activate
pip install --no-index --find-links "$WHEELS" -r requirements.txt

python - <<'PY'
import torch, transformers, peft, trl, bitsandbytes
print("torch", torch.__version__, "| CUDA build", torch.version.cuda)
print("transformers", transformers.__version__, "| peft", peft.__version__,
      "| trl", trl.__version__, "| bitsandbytes", bitsandbytes.__version__)
print("GPU visible here:", torch.cuda.is_available(), "(False is normal without a GPU)")
PY
echo "Setup done. Next: mkdir -p logs && sbatch slurm/gpu_smoke_test.sbatch"
