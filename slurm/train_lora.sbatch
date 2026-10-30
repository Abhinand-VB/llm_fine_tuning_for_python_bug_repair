#!/bin/bash
# LoRA fine-tuning of Qwen2.5-Coder-0.5B on the bug-repair data (PROJECT_PLAN.md, Phase 4).
#
# Submit from the repo folder on the coe-hpc3 login node:
#     mkdir -p logs
#     sbatch slurm/train_lora.sbatch                 # short test run (MAX_STEPS below)
#     MAX_STEPS=-1 sbatch slurm/train_lora.sbatch    # full training (check-in 4)
#
# coe-hpc3 GPU partitions: gpuqs (max 2 days), gpuqm (7 days), gpuql (14 days). We request an
# A100; if A100s are busy use --gres=gpu:h100:1. Avoid gpu:p100 (no bf16). See docs/HPC_SETUP.md.

#SBATCH --job-name=qwen-train-lora
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
nvidia-smi || echo "nvidia-smi not available"

# Same Python module slurm/setup_env.sh used to create .venv
command -v module >/dev/null 2>&1 || source /etc/profile.d/lmod.sh
module load python3/3.12.12

source .venv/bin/activate

# Compute nodes have no internet: point at the model copied from a Mac (docs/HPC_SETUP.md).
export MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen2.5-Coder-0.5B-Instruct}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# Short test run by default (check-in 3). Override with MAX_STEPS=-1 for the full runs.
MAX_STEPS="${MAX_STEPS:-50}"

python src/train.py --config configs/lora.json --max-steps "$MAX_STEPS"

echo "Finished at $(date)"
