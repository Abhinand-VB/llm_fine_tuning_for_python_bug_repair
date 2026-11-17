#!/bin/bash
# Three-round repair evaluation of one system on MBPP and HumanEval (PROJECT_PLAN.md, Phase 5).
#
# Submit from the repo folder on the coe-hpc3 login node (after the adapter is trained):
#     mkdir -p logs
#     SYSTEM=baseline sbatch slurm/evaluate.sbatch
#     SYSTEM=lora     sbatch slurm/evaluate.sbatch
#     SYSTEM=qlora    sbatch slurm/evaluate.sbatch
#     SYSTEM=lora EVAL_SETS=mbpp LIMIT=100 sbatch slurm/evaluate.sbatch    # quick check
#
# Or run the whole check-in 4 pipeline (train both, then evaluate all three) with
#     bash slurm/run_all.sh
#
# Writes results/eval_<system>_<eval-set>.json for each evaluation set.

#SBATCH --job-name=qwen-eval
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
nvidia-smi || echo "nvidia-smi not available"

command -v module >/dev/null 2>&1 || source /etc/profile.d/lmod.sh
module load python3/3.12.12
source .venv/bin/activate

export MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen2.5-Coder-0.5B-Instruct}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

SYSTEM="${SYSTEM:-baseline}"
EVAL_SETS="${EVAL_SETS:-mbpp humaneval humaneval-gen}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LIMIT_ARG=""
[ -n "$LIMIT" ] && LIMIT_ARG="--limit $LIMIT"

# HumanEval option (b) mutants are built once from data/raw/HumanEval.jsonl.gz.
if [ ! -f data/processed/humaneval_mutants.jsonl ]; then
    python src/data.py
fi

for SET in $EVAL_SETS; do
    echo "=== $SYSTEM on $SET ($(date))"
    python src/evaluate.py --system "$SYSTEM" --eval-set "$SET" \
        --batch-size "$BATCH_SIZE" --workers "${SLURM_CPUS_PER_TASK:-8}" $LIMIT_ARG
done

echo "Finished at $(date)"
