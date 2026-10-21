#!/bin/bash
# GPU smoke test: load Qwen2.5-Coder-0.5B (bf16 and 4-bit), add LoRA,
# run one training step and record GPU memory. About 5 minutes.
#
# Submit from the repo folder on the coe-hpc3 login node:
#     mkdir -p logs
#     sbatch slurm/gpu_smoke_test.sbatch
#
# coe-hpc3 GPU partitions: gpuqs (max 2 days), gpuqm (7 days), gpuql (14 days).
# They mix P100, A100 and H100 nodes; we request an A100. If A100s are busy, use
# --gres=gpu:h100:1 instead. Avoid gpu:p100 (no bf16, not supported by recent PyTorch).
# Check what is free with:  sinfo -p gpuqs -o "%20N %10T %30G"

#SBATCH --job-name=qwen-gpu-smoke
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
nvidia-smi || echo "nvidia-smi not available"

# Same Python module that slurm/setup_env.sh used to create .venv
command -v module >/dev/null 2>&1 || source /etc/profile.d/lmod.sh
module load python3/3.12.12

source .venv/bin/activate

# Compute nodes have no internet: the model folder was copied from a Mac
# (see docs/HPC_SETUP.md, offline setup).
export MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen2.5-Coder-0.5B-Instruct}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

python src/gpu_smoke_test.py

echo "Finished at $(date)"
