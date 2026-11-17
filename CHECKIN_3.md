# Check-in 3

Project Check-in 3

***LLM Fine-Tuning for Python Bug Repair***

Neha Rao (016403179) · Abhinand Vijayakumar Binsu (019107920) · Sai Bhavana Aluri (020092033)

●       **Final Mutation Dataset (with survival statistics):** The bug-mutation pipeline is complete and the final training dataset is generated. Correct MBPP solutions are cleaned, mutated with an AST transformer (operator, boundary and variable changes), and every mutant is run against its unit tests in the sandbox. We keep only the mutants that actually **fail** a test, together with their execution feedback, as (problem, buggy code, feedback) → fixed code triples. Both difficulty levels are implemented: level 1 (a single mutation) and level 2 (2–3 mutations on the same solution).

➔    Size: 2,741 training examples (338 tasks), 693 validation examples (84 tasks) and 1,483 held-out MBPP test examples (479 tasks) — 4,917 in total. Training and validation come from the MBPP train/validation splits only; the test mutants come from the MBPP test tasks and are never used for training.

➔    Survival by bug type (train): variable 1,015, operator 616, boundary 516, and the level-2 mixes operator+variable 277, boundary+variable 139, boundary+operator 109, boundary+operator+variable 69. Boundary sites stay the scarcest, as predicted in Check-in 1.

➔    Survival by level (train): 1,876 single-mutation and 865 multi-mutation examples.

➔    The kept bugs break the code in realistic ways, not just assertion mismatches: 1,800 AssertionError, 497 TypeError, 260 IndexError, plus smaller counts of AttributeError, TimeoutError, RecursionError, ZeroDivisionError and others.

➔    Each example is stored as JSONL with task_id, split, problem, tests, buggy_code, feedback, fixed_code, bug_type, mutation, num_mutations, error_type, level and round. Pipeline code: src/mutate.py.

●       **Prompt Format and Prompt-only Baseline (in progress):** The shared prompt design is settled — one chat-template prompt carrying the problem statement, the current code and the test feedback, with an instruction to return only the corrected function. The prompt module (src/prompts.py), the code extractor for model output, and the measured baseline numbers for the untuned Qwen2.5-Coder-0.5B-Instruct are not finished yet. We will complete the module, run the baseline over a sample of the validation set, and record per-task latency to size the evaluation budget on the cluster.

●       **LoRA and QLoRA Test Runs on the HPC (infrastructure ready, runs pending):** The training pipeline is built and committed. src/train.py uses trl SFTTrainer with peft and computes the loss on the fixed-code response only; it logs trainable parameters, peak GPU memory, wall-clock time and final loss. The LoRA and QLoRA configs (r=16, alpha=32, dropout=0.05 on all projections; lr 2e-4 cosine with 3% warmup; 3 epochs; max sequence length 1,024; QLoRA adds 4-bit NF4 with double quantization) and one SLURM job per setup are in place, modeled on the A100 GPU smoke test with the offline model path. They default to short test runs (MAX_STEPS=50), with MAX_STEPS=-1 reserved for the full Check-in 4 runs. The remaining step is to submit the short LoRA and QLoRA jobs on coe-hpc3, confirm the loss decreases end to end, and record the first parameter / memory / time numbers.

●       **Tentative Abstract:** Code language models often generate programs with logic or syntax errors that fail unit tests. We investigate whether parameter-efficient fine-tuning can teach a small code model to repair Python programs using execution feedback. Starting from correct MBPP solutions, we inject controlled bugs (operator, boundary and variable mutations) and run unit tests in a sandbox to build triples of buggy code, test feedback and fixed code. We fine-tune Qwen2.5-Coder-0.5B-Instruct with LoRA and QLoRA and compare them with a prompt-only baseline in a three-round repair loop, where the model receives the failing test output after each attempt. We evaluate on held-out MBPP tasks and on HumanEval, reporting pass rate, per-round repair rate and pass@1 alongside trainable parameters, peak GPU memory and training time, to characterize the accuracy-versus-compute trade-off of each approach.

●       **Report Outline:** The planned sections for the final report are:

➔    1. Abstract. 2. Introduction (motivation, research question, contributions). 3. Related work (code LLMs, self-repair with execution feedback, LoRA and QLoRA). 4. Data (MBPP and HumanEval, exploration findings, bug-mutation pipeline, dataset statistics). 5. Method (model, prompt format, the three systems, training setup, three-round repair protocol, sandbox). 6. Experimental setup (SJSU HPC / SLURM, hyperparameters, metrics). 7. Results (main table, repair rate by round, accuracy versus compute, breakdown by bug type). 8. Discussion (qualitative examples, failure analysis, limitations). 9. Conclusion and future work. 10. References. 11. Appendix.

●       **Team Contributions:** What each member did for this check-in.

➔    **Abhinand Vijayakumar Binsu (019107920):** Finalized the bug-mutation pipeline (src/mutate.py) — level-2 multi-mutations, built-in-shadowing guards, deduplication and per-type caps — and generated the final train/validation/test dataset with the survival statistics reported above.

➔    **Neha Rao (016403179):** Led the prompt-format and baseline work: settled the shared chat-template prompt and the three-round protocol, and is finishing the prompt module, the output code extractor and the prompt-only baseline measurement.

➔    **Sai Bhavana Aluri (020092033):** Built the LoRA/QLoRA training pipeline (src/train.py), the two training configs and the SLURM jobs for coe-hpc3, and is preparing the short test runs on the HPC.

●       **Remaining Work:** Finish the prompt module and record the prompt-only baseline (Neha); submit the short LoRA and QLoRA test runs on the HPC and log the first efficiency numbers (Sai Bhavana). These feed into Check-in 4 (Nov 16): the full LoRA and QLoRA training runs and the three-round repair evaluation on held-out MBPP and HumanEval.

●       **Main Risks:**

➔    Synthetic bugs may be simpler than the mistakes real models make. The level-2 multi-mutations help, and HumanEval option (a) tests repair of the model's own mistakes; we note it as a limitation.

➔    HPC queue waits or GPU limits. We smoke-test small jobs first; the 0.5B model also fits on one consumer GPU or Colab as a fallback.

➔    Evaluation cost (3 rounds × hundreds of tasks × 3 systems). We batch generation and can subsample the MBPP test set if needed.

●       **Google Drive:**

➔   https://drive.google.com/drive/folders/1KF98YepQ9mtO7_wYZsR63sDJW9Sd-a2e?usp=drive_link

●       **GitHub:**

➔   https://github.com/Abhinand-VB/llm_fine_tuning_for_python_bug_repair

*The instructor and TA have been invited to both.*
