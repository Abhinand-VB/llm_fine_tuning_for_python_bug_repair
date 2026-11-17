# HPC setup

How to get the training environment running on the SJSU College of Engineering HPC and check that a
GPU job works. Owner: Sai Bhavana.

## 1. Account and login

- **Account:** only faculty can request HPC access, so the course professor requests it for each student.
  Class-project access lasts up to 6 months.
- **Network:** the HPC is reachable only from the campus network (eduroam WiFi or a campus LAN port) or
  through the SJSU VPN.
- **Login:** your SJSUOne password; nothing shows while you type it.

  ```bash
  ssh SJSU_ID@coe-hpc1.sjsu.edu
  ```
- **The new GPU nodes are on the coe-hpc3 cluster,** reached from coe-hpc1. The first time only, set up
  an SSH key on coe-hpc1:

  ```bash
  ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -q -N ""
  cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys
  ```
  then in `~/.ssh/config` comment out `IdentityFile ~/.ssh/cluster` and add
  `IdentityFile ~/.ssh/id_ed25519`. After that, every time:

  ```bash
  ssh coe-hpc3
  ```
  All the steps below run on coe-hpc3.

## 2. Cluster layout (coe-hpc3)

```bash
sinfo -o "%15P %20N %10T %30G"     # partitions, nodes, state and GPU type
module avail python                # python3/3.11.7 and python3/3.12.12 (we use 3.12.12)
```

| Partition | Time limit | GPUs |
|---|---|---|
| `gpuqs` | 2 days | A100 (`cs001`–`cs004`, 4 per node), H100 (`g16`, `g18`, `g19`), P100 (rest) |
| `gpuqm` | 7 days | same nodes |
| `gpuql` | 14 days | same nodes |

Request a GPU type with `--gres`, for example `--gres=gpu:a100:1` or `--gres=gpu:h100:1`. Avoid
`gpu:p100`: 12 GB, no bf16 support, and not supported by the newest PyTorch builds. Our jobs use
`gpuqs` because each run is far shorter than 2 days. The `/home` drive is shared and almost full, so
keep large files (checkpoints) small and delete old ones.

The login node is for setup only. Never train on it; GPU work goes through `sbatch`.

## 3. Environment (offline setup)

Only coe-hpc1 can reach the internet. The GPU login node (g17) and all compute nodes cannot (`curl`
times out), and jobs must never run on coe-hpc1. So we download on a Mac, copy the files over with
`rsync`, and install from local files inside a scheduled job. The `/home` drive is shared by all
nodes, so files copied once are visible everywhere.

**On the Mac** (campus network or VPN), in the repo folder with its `.venv` active:

```bash
mkdir -p ~/hpc_files/wheels
pip download -r requirements.txt -d ~/hpc_files/wheels --only-binary=:all: \
  --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux_2_28_x86_64 \
  --python-version 3.12 --implementation cp
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-Coder-0.5B-Instruct', local_dir='$HOME/hpc_files/Qwen2.5-Coder-0.5B-Instruct')"
ssh SJSU_ID@coe-hpc1.sjsu.edu "mkdir -p ~/wheels ~/models"
rsync -av --progress ~/hpc_files/wheels/ SJSU_ID@coe-hpc1.sjsu.edu:~/wheels/
rsync -av --progress ~/hpc_files/Qwen2.5-Coder-0.5B-Instruct SJSU_ID@coe-hpc1.sjsu.edu:~/models/
```

(about 5 GB; the wheels are Linux x86_64 builds, so they differ from what a Mac would install).

**On the HPC**, get the repo (`git clone` on coe-hpc1, which is the only machine with internet, or
`rsync` the folder from the Mac), then on g17 install as a scheduled job:

```bash
ssh coe-hpc3
cd llm_fine_tuning_for_python_bug_repair
srun -p defq -n 1 -c 4 --mem=16G --time=00:30:00 bash slurm/setup_offline.sh
```

`slurm/setup_env.sh` is the online version, for a machine with internet access (for example a laptop
or Colab).

## 4. GPU smoke test

```bash
mkdir -p logs
sbatch slurm/gpu_smoke_test.sbatch     # gpuqs partition, one A100
squeue -u $USER                        # wait until the job disappears from the queue
cat logs/qwen-gpu-smoke-*.out
```

[`src/gpu_smoke_test.py`](../src/gpu_smoke_test.py) loads Qwen2.5-Coder-0.5B-Instruct in both planned
setups, adds LoRA (r=16, alpha=32, all projections), runs one training step on a 1,024-token example
(batch size 1) and writes [`results/gpu_smoke_test.json`](../results/gpu_smoke_test.json).

## 5. Results

| | LoRA (bf16) | QLoRA (4-bit NF4) |
|---|---:|---:|
| GPU | | |
| Memory after loading the model | | |
| Peak memory, one training step (1,024 tokens, batch 1) | | |
| Trainable parameters | | |

Cluster details found during setup: GPU type used, GRES name for requesting it, Python module loaded.

## 6. Check-in 4 pipeline (full training + evaluation)

Everything for check-in 4 is one command on coe-hpc3, from the repo folder:

```bash
mkdir -p logs
MAX_STEPS=20 bash slurm/run_all.sh     # optional: short end-to-end check first
bash slurm/run_all.sh                  # full LoRA + QLoRA training, then all evaluations
squeue -u $USER
```

`run_all.sh` submits the two training jobs and three `slurm/evaluate.sbatch` jobs (baseline,
LoRA, QLoRA); the LoRA/QLoRA evaluations wait for their training job (`--dependency=afterok`).
Each evaluation runs MBPP test mutants, HumanEval mutants and HumanEval generate + repair, and
builds `data/processed/humaneval_mutants.jsonl` on first use. When the jobs finish:

```bash
source .venv/bin/activate && python src/analyze.py    # results/summary.md, figures/, examples.md
```

Notes from the first full run (Nov 16, account `019107920`):
- Wheels downloaded on coe-hpc1 with `module load python3/3.11.5` (its default `python3` is 3.6,
  too old for pip to target Python 3.12) and `pip download ... --python-version 3.12`, adding
  `--platform manylinux_2_24_x86_64`; without it pip silently picks bitsandbytes 0.42, too old
  for transformers 5. Torch came from the CUDA 12.6 index (`torch 2.7.1+cu126`); the A100 nodes
  run driver 590.
- transformers 5 removed `warmup_ratio`; `src/train.py` passes the ratio as a float
  `warmup_steps` when the old name is missing.
- `srun -p defq ... --mem=16G` was refused; `-p cpuqs --mem=8G` works for the offline install.
- A 20-step LoRA test step takes ~4.2 s on an A100 (batch 1 × 16 accumulation), so a full
  3-epoch run (~514 optimizer steps) takes roughly 40 minutes.
- Files copied from a Windows checkout have CRLF line endings, which break the bash scripts;
  strip them with `sed -i 's/$//' slurm/*`.

## Troubleshooting

- **`torch.cuda.is_available()` is False in the job:** the job didn't get a GPU (check
  `--gres=gpu:1` and the partition) or the GPU driver is older than torch's CUDA build. Compare the
  CUDA version in `nvidia-smi` with "cuda (torch build)" in the log, and if needed reinstall torch
  for an older CUDA, for example
  `pip install torch --index-url https://download.pytorch.org/whl/cu118`.
- **"no kernel image is available" or a warning that the GPU is not supported:** the job landed on a
  P100. Request an A100 or H100 with `--gres` (see the job script), or install a torch build that still
  supports it: `pip install torch --index-url https://download.pytorch.org/whl/cu126`.
- **`bf16_supported: false`:** older GPUs (such as P100 or V100) have no bf16. The script falls back to
  fp16; training configs must do the same.
- **Disk quota exceeded:** set `HF_HOME` to a larger scratch folder, if the cluster has one, in both
  the setup script and the job script.
- **Job stays pending:** `squeue -u $USER` shows the reason. If the A100s are busy, switch the job
  script to `--gres=gpu:h100:1`.
