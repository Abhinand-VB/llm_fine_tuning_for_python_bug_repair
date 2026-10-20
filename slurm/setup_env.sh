#!/bin/bash
# One-time setup of the training environment on the HPC login node.
# Creates .venv in the repo, installs the training libraries and downloads
# the model so GPU jobs can run offline.
#
# Run from the repo folder on coe-hpc3:
#     module load python3/3.12.12
#     bash slurm/setup_env.sh
#
# Needs Python 3.10 or newer (check with `module avail python`).

set -eo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
echo "Using $($PY --version) at $(command -v $PY)"

$PY -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
python - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download("Qwen/Qwen2.5-Coder-0.5B-Instruct")
print("Model downloaded to", path)
PY

python - <<'PY'
import torch, transformers, peft, trl, bitsandbytes
print("torch", torch.__version__, "| CUDA build", torch.version.cuda)
print("transformers", transformers.__version__, "| peft", peft.__version__,
      "| trl", trl.__version__, "| bitsandbytes", bitsandbytes.__version__)
print("GPU visible here:", torch.cuda.is_available(), "(False is normal on a login node)")
PY
echo "Setup done. Next: mkdir -p logs && sbatch slurm/gpu_smoke_test.sbatch"
