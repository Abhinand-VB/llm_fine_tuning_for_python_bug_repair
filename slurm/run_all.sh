#!/bin/bash
# Check-in 4 pipeline on coe-hpc3: full LoRA and QLoRA training, then the three-round
# evaluation of the baseline, LoRA and QLoRA on MBPP and HumanEval, then the analysis.
# Each evaluation waits for its training job through a SLURM dependency, so this submits
# everything at once and returns.
#
# Run from the repo folder on the coe-hpc3 login node:
#     bash slurm/run_all.sh
#     MAX_STEPS=50 bash slurm/run_all.sh        # short end-to-end check first
#
# Follow with `squeue -u $USER` and `tail -f logs/*.out`. When all jobs finish:
#     python src/analyze.py                    # tables and charts in results/

set -eo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

MAX_STEPS="${MAX_STEPS:--1}"
export MAX_STEPS

lora=$(sbatch --parsable slurm/train_lora.sbatch)
qlora=$(sbatch --parsable slurm/train_qlora.sbatch)
echo "train: lora=$lora qlora=$qlora (MAX_STEPS=$MAX_STEPS)"

base_eval=$(SYSTEM=baseline sbatch --parsable slurm/evaluate.sbatch)
lora_eval=$(SYSTEM=lora sbatch --parsable --dependency=afterok:$lora slurm/evaluate.sbatch)
qlora_eval=$(SYSTEM=qlora sbatch --parsable --dependency=afterok:$qlora slurm/evaluate.sbatch)
echo "eval: baseline=$base_eval lora=$lora_eval qlora=$qlora_eval"
echo "When they finish: python src/analyze.py"
