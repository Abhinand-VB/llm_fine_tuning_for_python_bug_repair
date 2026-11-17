# Check-in 4

Project Check-in 4

***LLM Fine-Tuning for Python Bug Repair***

Neha Rao (016403179) · Abhinand Vijayakumar Binsu (019107920) · Sai Bhavana Aluri (020092033)

●       **Full LoRA and QLoRA Training Runs (done):** Both adapters were trained on the full mutation dataset (2,741 training and 693 validation examples, 3 epochs, 516 optimizer steps) on one NVIDIA A100-PCIE-40GB on coe-hpc3, with the configs from Check-in 3 (r=16, alpha=32, dropout=0.05 on all seven projections; lr 2e-4 cosine, 3% warmup; batch 1 × 16 gradient accumulation; max length 1,024). The loss is computed on the fixed-code tokens only.

➔    LoRA (bf16 base): 8.80M trainable parameters (1.75% of the model), 30.8 minutes, 1.29 GB peak GPU memory, final training loss 0.0058.

➔    QLoRA (4-bit NF4 base, double quantization): the same 8.80M trainable parameters, 42.0 minutes, 1.15 GB peak GPU memory (11% less than LoRA), final training loss 0.0061. QLoRA is slower because of on-the-fly dequantization. (Its "2.7%" trainable share in the logs is only because the 4-bit weights are counted in packed form.)

➔    Validation loss is lowest after epoch 1 and rises afterward (LoRA 0.048 → 0.061 → 0.065; QLoRA 0.044 → 0.058 → 0.058), while training loss keeps falling toward zero. Both adapters overfit mildly after the first epoch, so a 1-epoch run is a cheap follow-up that may do as well or better.

➔    The Check-in 3 test runs were also completed (20-step LoRA and QLoRA jobs), which surfaced and fixed one incompatibility: transformers 5 removed `warmup_ratio`, so src/train.py now passes the ratio as a float `warmup_steps` when needed.

●       **Three-round Repair Evaluation on MBPP and HumanEval (done):** src/evaluate.py runs the same protocol for all three systems: greedy decoding, the model gets the problem, the current code and the test feedback, the extracted code runs in the sandbox, and a failure is fed back for up to three rounds. Generation is batched (32 prompts at a time, sorted by length). The baseline uses the 1-shot repair prompt; the fine-tuned systems use the zero-shot prompt they were trained on, built by the same src/prompts.py function. Evaluation sets: 1,483 held-out MBPP test mutants; 395 HumanEval mutants built by the same pipeline (option (b): one failing mutant per task and bug type, 156 tasks); and the 164 HumanEval tasks in generate-then-repair mode (option (a): round 0 writes a solution from the prompt = pass@1).

| System | MBPP mutants (pass after 3 rounds) | HumanEval mutants | HumanEval pass@1 | HumanEval after repair | Train time | Train peak GPU |
|---|---|---|---|---|---|---|
| Baseline (prompt-only) | 36.3% | 26.6% | 51.2% | 55.5% | 0 | n/a |
| LoRA | **68.1%** | **72.4%** | 7.9% | 7.9% | 30.8 min | 1.29 GB |
| QLoRA | 65.0% | 64.6% | 1.2% | 1.2% | 42.0 min | 1.15 GB |

➔    Fine-tuning nearly doubles the repair rate on held-out MBPP (36.3% → 68.1% LoRA, 65.0% QLoRA) and generalizes to HumanEval mutants, a benchmark never used in training (26.6% → 72.4% / 64.6%). LoRA fixes 531 MBPP mutants the baseline misses; the baseline fixes only 59 that LoRA misses.

➔    QLoRA keeps most of LoRA's gain (−3.1 points on MBPP, −7.8 on HumanEval mutants) with 11% less training memory.

➔    Almost all repairs happen in round 1 (LoRA 67.4% → 68.1% over rounds 2–3; baseline 33.5% → 36.3%). The untuned model returns the buggy code unchanged in 32% of its first answers on MBPP (48% on HumanEval mutants); with greedy decoding an unchanged answer produces the same prompt next round, so extra rounds rarely help. The fine-tuned models almost never echo the bug (2.5–2.8%).

➔    Gains hold for every bug type and grow with difficulty: boundary bugs 23.9% → 61.2% (LoRA), and the hardest level-2 mix (boundary+operator+variable) 9.5% → 61.9%.

➔    **Specialization cost:** on from-scratch generation (HumanEval option (a)) the fine-tuned models collapse: LoRA returns the prompt stub unchanged, with no function body, on 119 of 164 tasks. The repair data teaches "copy the code and change one or two tokens", so given an empty stub the adapter copies it. A mixed training set (repair plus some from-scratch MBPP solutions) is the natural fix and a planned follow-up.

➔    Per-task cost on an A100 is 0.21 s (baseline), 0.26 s (LoRA) and 0.39 s (QLoRA) for the whole three-round loop on MBPP, with ≤ 2.7 GB peak memory during evaluation.

➔    Full tables, charts (repair rate by round, accuracy vs compute, repair by bug type, loss curves) and qualitative examples: results/summary.md, results/figures/, results/examples.md.

●       **New code since Check-in 3:**

➔    src/evaluate.py (the full multi-round harness for all three systems and evaluation sets, with per-round, per-bug-type and per-error-type metrics and a per-task log); src/data.py (HumanEval loader: inlines `check(candidate)` into an MBPP-style assert list so the same sandbox and feedback formatter work, and builds the HumanEval mutants); src/analyze.py (tables, charts, qualitative examples); a generation prompt for HumanEval round 0 in src/prompts.py; src/train.py now imports its prompt from src/prompts.py so training and inference use identical prompts.

➔    slurm/evaluate.sbatch and slurm/run_all.sh: the whole Check-in 4 pipeline (both trainings, then the three evaluations, chained with SLURM dependencies) is one command on coe-hpc3. docs/HPC_SETUP.md documents the offline install and the issues met.

●       **Open decisions resolved:** HumanEval is reported both ways, options (a) and (b). The test feedback comes from the same tests used for scoring (MBPP has only three); absolute numbers are therefore optimistic, but all three systems see identical feedback, so the comparison is fair. This is listed as a limitation.

●       **Team Contributions:** *(to be reviewed and edited by the team)*

➔    **Abhinand Vijayakumar Binsu (019107920):** Built the full evaluation harness, the HumanEval evaluation sets and the analysis script; set up the training environment on his coe-hpc3 account; ran the LoRA/QLoRA test runs, the full training runs and all nine evaluations; produced the results tables and charts.

➔    **Neha Rao (016403179):** Built the shared prompt format, the code extractor and the first prompt-only baseline and evaluation loop (src/prompts.py, src/baseline.py, the first src/evaluate.py), which the final harness builds on.

➔    **Sai Bhavana Aluri (020092033):** Built the LoRA/QLoRA training pipeline (src/train.py), the two configs and the SLURM training jobs used for the full runs, and the HPC setup documentation.

●       **Remaining Work (report Nov 27, slides Nov 30):**

➔    Write the final report following the outline from Check-in 3, using results/summary.md and results/figures/, with qualitative examples from results/examples.md.

➔    Optional, if time allows: a 1-epoch run (validation loss suggests it may match or beat 3 epochs); 2 extra seeds per system for error bars (~30–40 minutes each); a mixed repair + generation training set to remove the specialization cost; visible/hidden test split for HumanEval feedback.

➔    Slides (6 max) and the software demo.

●       **Main Risks:**

➔    The synthetic bugs are simpler than real model mistakes; the option (a) results show the fine-tuned models do not transfer to repairing their own generations. We report this as a limitation and a direction for future work.

➔    Single seed: differences of a few points (LoRA vs QLoRA) need error bars before drawing firm conclusions.

➔    Benchmark contamination (Qwen may have seen MBPP/HumanEval in pretraining) affects all three systems equally; our mutated inputs are new.

●       **Google Drive:**

➔   https://drive.google.com/drive/folders/1KF98YepQ9mtO7_wYZsR63sDJW9Sd-a2e?usp=drive_link

●       **GitHub:**

➔   https://github.com/Abhinand-VB/llm_fine_tuning_for_python_bug_repair

*The instructor and TA have been invited to both.*
