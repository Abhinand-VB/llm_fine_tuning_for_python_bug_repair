"""Prompt-only baseline for the bug-repair task (PROJECT_PLAN.md, Phase 3).

Runs the untuned Qwen2.5-Coder-0.5B-Instruct through the three-round repair loop so we have a
number to compare the fine-tuned (LoRA / QLoRA) systems against. Each round:

    1. Build the prompt from the problem, the current code and the latest test feedback
       (src/prompts.py) and generate greedily, so pass@1 is deterministic.
    2. Extract the function and run it in the sandbox against the task's tests.
    3. If it passes, stop. Otherwise format the new failure as feedback and try again, up to
       three rounds.

The evaluation set is the held-out MBPP test mutants (data/processed/test.jsonl); these are
mutants of MBPP test tasks and were never used for training. We reuse the sandbox and the
feedback formatter from the mutation pipeline so the feedback the baseline sees matches the
feedback the models are trained on.

This is the baseline system only. The shared multi-round harness that runs all three systems is
Phase 5 (src/evaluate.py, check-in 4).

Usage (a GPU node, or any Linux/macOS machine with the model cached; the sandbox needs Unix):
    python src/baseline.py                       # full test set
    python src/baseline.py --limit 50            # quick budget check
    python src/baseline.py --few-shot 0          # zero-shot instead of 1-shot

Output:
    results/baseline.json   metrics (pass rate, repair rate by round, timing) and per-task log
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from prompts import default_few_shot, extract_code, render_prompt  # noqa: E402
from mutate import format_feedback, run  # noqa: E402  reuse the sandbox + feedback format

MODEL_ID_DEFAULT = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
TEST_FILE = ROOT / "data" / "processed" / "test.jsonl"
OUT = ROOT / "results" / "baseline.json"


def load_examples(path, limit=None):
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
            if limit and len(examples) >= limit:
                break
    return examples


def repair_loop(model, tok, example, few_shot, rounds, max_new_tokens):
    """Run one example through up to `rounds` repair attempts.

    Returns (solved_round or None, [per-round seconds], final_code).
    """
    import torch

    problem = example["problem"]
    tests = example["tests"]
    code = example["buggy_code"]
    feedback = example["feedback"]
    round_times = []

    for r in range(1, rounds + 1):
        prompt = render_prompt(tok, problem, code, feedback, few_shot=few_shot)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        round_times.append(round(time.time() - t0, 2))

        reply = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        code = extract_code(reply) or code

        passed, error = run(code, tests, "")
        if passed:
            return r, round_times, code
        # Feed the new failure back in for the next round.
        feedback, _ = format_feedback(code, tests, error or "")

    return None, round_times, code


def main():
    ap = argparse.ArgumentParser(description="Prompt-only bug-repair baseline (Phase 3).")
    ap.add_argument("--model", default=MODEL_ID_DEFAULT,
                    help="model id or local path (set MODEL_PATH on the offline HPC)")
    ap.add_argument("--data", default=str(TEST_FILE), help="held-out test mutants (JSONL)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N tasks")
    ap.add_argument("--rounds", type=int, default=3, help="repair rounds per task")
    ap.add_argument("--few-shot", type=int, default=1, choices=[0, 1],
                    help="number of few-shot examples (0 = zero-shot)")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    import os
    model_id = os.environ.get("MODEL_PATH", args.model)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    examples = load_examples(args.data, args.limit)
    few_shot = default_few_shot() if args.few_shot else []
    print(f"Loaded {len(examples)} tasks from {args.data}; {args.few_shot}-shot, "
          f"{args.rounds} rounds, model {model_id}")

    tok = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) \
        else (torch.float16 if torch.cuda.is_available() else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, device_map="auto" if torch.cuda.is_available() else None)
    model.eval()

    solved_at = []          # round each solved task was fixed in; None if never
    per_task = []
    t_start = time.time()
    for i, ex in enumerate(examples):
        r, times, final_code = repair_loop(model, tok, ex, few_shot, args.rounds,
                                            args.max_new_tokens)
        solved_at.append(r)
        per_task.append({"task_id": ex.get("task_id"), "bug_type": ex.get("bug_type"),
                         "level": ex.get("level"), "solved_round": r,
                         "round_seconds": times})
        status = f"solved@round {r}" if r else "unsolved"
        print(f"[{i + 1}/{len(examples)}] task {ex.get('task_id')} ({ex.get('bug_type')}): {status}")

    n = len(examples)
    solved = [r for r in solved_at if r is not None]
    all_times = [t for row in per_task for t in row["round_seconds"]]
    report = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "model": model_id,
        "data": args.data,
        "n_tasks": n,
        "few_shot": args.few_shot,
        "rounds": args.rounds,
        "pass_rate": round(len(solved) / n, 4) if n else 0.0,
        # Repair rate @ k: share of all tasks fixed by round k (cumulative).
        "repair_rate_by_round": {
            str(k): round(sum(1 for r in solved if r <= k) / n, 4) if n else 0.0
            for k in range(1, args.rounds + 1)
        },
        "avg_seconds_per_generation": round(sum(all_times) / len(all_times), 2) if all_times else 0.0,
        "total_minutes": round((time.time() - t_start) / 60, 1),
        "per_task": per_task,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\npass rate: {report['pass_rate']:.1%}  "
          f"(by round: {report['repair_rate_by_round']})")
    print(f"~{report['avg_seconds_per_generation']}s per generation; wrote {out_path}")


if __name__ == "__main__":
    main()
