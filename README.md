# LLM Fine-Tuning for Python Bug Repair

Can parameter-efficient fine-tuning teach a small code model to repair its own Python bugs using execution feedback?

This project compares a **prompt-only baseline** against **LoRA** and **QLoRA** fine-tuning of
[Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct). Each model gets
up to three rounds to fix a buggy program. After every round, the failing unit-test output is fed back to it.

## Motivation

Code models often produce logic or syntax errors that fail unit tests. Prompt-only repair methods leave the
model's weights unchanged. Fine-tuning updates them, so the model can learn the relationship between a bug,
the execution feedback it produces, and the correct fix. We want to measure whether LoRA and QLoRA give
meaningful repair gains, and at what compute cost.

## Datasets

| Dataset | Role |
|---|---|
| [MBPP](https://github.com/google-research/google-research/tree/master/mbpp) (Mostly Basic Python Problems) | Training and validation |
| [HumanEval](https://github.com/openai/human-eval) | Held-out external test benchmark |

### Bug-mutation pipeline

Correct MBPP solutions go through controlled mutations to produce paired *buggy → fixed* examples:

- **Operator errors**: e.g. `+` ↔ `-`, `<` ↔ `<=`, `and` ↔ `or`
- **Boundary errors**: off-by-one changes in ranges, indices and slices
- **Variable errors**: swapped or misused identifiers

Each mutant runs against the task's unit tests in a sandbox. Mutants that fail are kept, along with their
error feedback (traceback or assertion failure), to form training triples:
`(buggy code, execution feedback) → fixed code`.

## Method

1. **Baseline**: prompt-only multi-round repair with the base model.
2. **LoRA**: low-rank adapters trained on the mutation dataset.
3. **QLoRA**: LoRA on a 4-bit quantized base model.

All three use the same **three-round correction protocol**: generate a fix, run the tests, return the
feedback, and repeat until the tests pass or three rounds have been used.

### Tooling

- Hugging Face `transformers`, `peft`, `trl` (SFT training)
- `bitsandbytes` (4-bit quantization for QLoRA)
- A custom MBPP bug-mutation pipeline
- A secure execution sandbox for running untrusted generated code (timeouts and process isolation)
- SLURM batch jobs on the SJSU College of Engineering HPC

## Evaluation

Evaluation runs on held-out MBPP tasks and on HumanEval.

**Task metrics**
- Pass rate (final, after up to 3 rounds)
- Repair rate (share of initially failing programs fixed), reported per round
- HumanEval pass@1

**Efficiency metrics**
- Trainable parameters
- Peak GPU memory
- Training time

**Deliverables**: comparison tables and charts of accuracy against compute for the baseline, LoRA and QLoRA.

## Team

| Member | Responsibilities |
|---|---|
| **Neha Rao** (016403179) | Leads the evaluation harness and HumanEval benchmarking; produces comparison charts and final documentation |
| **Abhinand Vijayakumar Binsu** (019107920) | Co-manages the MBPP data pipeline, assists with SLURM fine-tuning jobs, contributes to evaluation analysis |
| **Sai Bhavana Aluri** (020092033) | Leads model configuration and LoRA/QLoRA training via SLURM; supports dataset validation |

## References

1. San José State University, Computer Engineering. *College of Engineering HPC.*
2. Qwen. *Qwen2.5-Coder-0.5B-Instruct Model Card.* https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct
3. Google Research. *Mostly Basic Python Problems Dataset.* https://github.com/google-research/google-research/tree/master/mbpp
4. OpenAI. *HumanEval Evaluation Repository.* https://github.com/openai/human-eval
