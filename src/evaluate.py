"""Multi-round repair evaluation for all three systems (PROJECT_PLAN.md, Phase 5).

The protocol is the same for the prompt-only baseline, LoRA and QLoRA:
    1. Round k: give the model the problem, the current code and the latest test feedback
       (src/prompts.py) and decode greedily, so the result is deterministic.
    2. Extract the function and run it in the sandbox against the task's tests.
    3. If it passes, stop; otherwise format the new failure as feedback (the same formatter the
       training data used) and go to round k + 1, for up to --rounds rounds.

Generation is batched: in each round every still-failing task is generated together, sorted by
prompt length to reduce padding, then all candidates run in the sandbox in parallel.

Evaluation sets (src/data.py):
    mbpp           1,483 held-out MBPP test mutants (never used for training)
    humaneval      HumanEval canonical solutions mutated by the same pipeline (option (b))
    humaneval-gen  HumanEval option (a): round 0 writes a solution from the prompt (pass@1),
                   rounds 1..3 repair it with test feedback

Systems:
    baseline   untuned Qwen2.5-Coder-0.5B-Instruct, 1-shot repair prompt by default
    lora       base model (bf16/fp16) + adapters/lora, zero-shot (the format it was trained on)
    qlora      4-bit NF4 base + adapters/qlora, zero-shot

Usage (a GPU node, normally through slurm/evaluate.sbatch; the sandbox needs Linux/macOS/WSL):
    python src/evaluate.py --system baseline --eval-set mbpp
    python src/evaluate.py --system lora --eval-set humaneval-gen
    python src/evaluate.py --system qlora --eval-set mbpp --limit 100     # quick check

Output:
    results/eval_<system>_<eval-set>.json   metrics (pass rate, repair rate by round, breakdown
                                            by bug type / level / error type, efficiency) and a
                                            per-task log with the code produced in every round
"""

import argparse
import json
import multiprocessing
import os
import socket
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from data import EVAL_SETS, load_eval_set  # noqa: E402
from mutate import format_feedback, normalize, run  # noqa: E402  same sandbox + feedback as training
from prompts import (default_few_shot, extract_code, render_generation_prompt,  # noqa: E402
                     render_prompt)

MODEL_ID_DEFAULT = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
SYSTEMS = ("baseline", "lora", "qlora")
MAX_CODE_LOG = 4000    # characters of code kept per round in the per-task log


# ------------------------------------------------------------------ model

def load_system(system, model_id, adapter, dtype_name):
    """Load the tokenizer and model for one system. Returns (tokenizer, model, info)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    cuda = torch.cuda.is_available()
    if dtype_name == "auto":
        dtype = (torch.bfloat16 if cuda and torch.cuda.is_bf16_supported()
                 else torch.float16 if cuda else torch.float32)
    else:
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_name]

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"           # decoder-only batched generation pads on the left
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = {"device_map": {"": 0} if cuda else None}
    four_bit = False
    if system != "baseline":
        cfg = json.loads((ROOT / "configs" / f"{system}.json").read_text())
        four_bit = cfg["load_in_4bit"]
    if four_bit:
        # Evaluate QLoRA on the same 4-bit NF4 base it was trained on.
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=cfg.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=dtype)
    else:
        kwargs["dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    if system != "baseline":
        from peft import PeftModel
        adapter = Path(adapter or ROOT / "adapters" / system)
        if not (adapter / "adapter_config.json").exists():
            raise FileNotFoundError(f"no LoRA adapter at {adapter}; train it with src/train.py")
        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    info = {"model": model_id, "adapter": str(adapter) if system != "baseline" else None,
            "dtype": str(dtype).replace("torch.", ""), "load_in_4bit": four_bit,
            "device": torch.cuda.get_device_name(0) if cuda else "cpu"}
    return tok, model, info


def generate(model, tok, prompts, batch_size, max_new_tokens):
    """Greedy generation for a list of prompts, batched by length. Returns replies in order."""
    import torch

    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]), reverse=True)
    replies = [None] * len(prompts)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 temperature=None, top_p=None, top_k=None,
                                 pad_token_id=tok.pad_token_id)
        width = enc["input_ids"].shape[1]
        for j, i in enumerate(idx):
            replies[i] = tok.decode(out[j, width:], skip_special_tokens=True)
    return replies


# ------------------------------------------------------------------ sandbox

def program(example, code):
    """The program that runs in the sandbox: the HumanEval preamble (imports and helpers the
    prompt defines) followed by the model's code. MBPP examples have no preamble."""
    preamble = example.get("preamble") or ""
    return f"{preamble}\n{code}" if preamble else code


def check_all(examples, codes, pool):
    """Run every candidate in the sandbox. Returns [(passed, feedback, error_type)]."""
    def one(args):
        ex, code = args
        full = program(ex, code)
        passed, error = run(full, ex["tests"], "")
        if passed:
            return True, "", None
        feedback, error_type = format_feedback(full, ex["tests"], error or "")
        return False, feedback, error_type
    return list(pool.map(one, zip(examples, codes)))


def same_code(a, b):
    """Whether two programs are the same after normalization (the model returned the input)."""
    try:
        return normalize(a) == normalize(b)
    except SyntaxError:
        return a.strip() == b.strip()


# ------------------------------------------------------------------ evaluation loop

def evaluate(examples, model, tok, few_shot, rounds, batch_size, max_new_tokens, workers):
    """Run the multi-round repair protocol over all examples. Returns the per-task log."""
    log = [{"task_id": ex["task_id"], "bug_type": ex.get("bug_type"), "level": ex.get("level"),
            "initial_error_type": ex.get("error_type"), "mutation": ex.get("mutation"),
            "solved_round": None, "rounds": []} for ex in examples]
    code = [ex.get("buggy_code") for ex in examples]
    feedback = [ex.get("feedback") or "" for ex in examples]
    gen_seconds = 0.0

    with ThreadPoolExecutor(workers) as pool:
        # Round 0 (HumanEval option (a) only): write a first solution from the prompt.
        first = [i for i, ex in enumerate(examples) if ex.get("buggy_code") is None]
        if first:
            t0 = time.time()
            replies = generate(model, tok, [render_generation_prompt(tok, examples[i]["prompt"])
                                            for i in first], batch_size, max_new_tokens)
            gen_seconds += time.time() - t0
            cands = [extract_code(r) for r in replies]
            results = check_all([examples[i] for i in first], cands, pool)
            for i, c, (passed, fb, et) in zip(first, cands, results):
                code[i], feedback[i] = c, fb
                log[i]["initial_error_type"] = et
                log[i]["rounds"].append({"round": 0, "passed": passed, "error_type": et,
                                         "code": c[:MAX_CODE_LOG]})
                if passed:
                    log[i]["solved_round"] = 0
            print(f"round 0: {sum(r[0] for r in results)}/{len(first)} pass@1", flush=True)

        for r in range(1, rounds + 1):
            active = [i for i in range(len(examples)) if log[i]["solved_round"] is None]
            if not active:
                break
            t0 = time.time()
            prompts = [render_prompt(tok, examples[i]["problem"], code[i], feedback[i],
                                     few_shot=few_shot) for i in active]
            replies = generate(model, tok, prompts, batch_size, max_new_tokens)
            gen_seconds += time.time() - t0
            # An empty extraction keeps the previous code (the round is then a failed attempt).
            cands = [extract_code(rep) or code[i] for i, rep in zip(active, replies)]
            results = check_all([examples[i] for i in active], cands, pool)
            for i, c, (passed, fb, et) in zip(active, cands, results):
                log[i]["rounds"].append({"round": r, "passed": passed, "error_type": et,
                                         "unchanged": same_code(c, code[i]),
                                         "code": c[:MAX_CODE_LOG]})
                code[i], feedback[i] = c, fb
                if passed:
                    log[i]["solved_round"] = r
            solved = sum(1 for row in log if row["solved_round"] is not None)
            print(f"round {r}: fixed {sum(x[0] for x in results)}/{len(active)}; "
                  f"{solved}/{len(examples)} solved so far ({time.time() - t0:.0f}s)", flush=True)
    return log, gen_seconds


# ------------------------------------------------------------------ metrics

def rate(num, den):
    return round(num / den, 4) if den else 0.0


def summarize(log, rounds):
    """Aggregate the per-task log into the Phase 5 metrics.

    pass_rate               share of tasks whose final program passes all tests
    pass_at_1               (humaneval-gen) share passing on the first, unrepaired attempt
    repair_rate_by_round    share of the initially failing programs fixed by round k
                            (cumulative). For the mutant sets every program fails initially.
    """
    n = len(log)
    has_round0 = any(row["rounds"] and row["rounds"][0]["round"] == 0 for row in log)
    failing = [row for row in log if row["solved_round"] != 0]
    m = {
        "n_tasks": n,
        "pass_rate": rate(sum(row["solved_round"] is not None for row in log), n),
        "repair_rate_by_round": {
            str(k): rate(sum(1 for row in failing if row["solved_round"] is not None
                             and row["solved_round"] <= k), len(failing))
            for k in range(1, rounds + 1)},
        "n_initially_failing": len(failing),
    }
    if has_round0:
        m["pass_at_1"] = rate(sum(row["solved_round"] == 0 for row in log), n)
        m["pass_rate_by_round"] = {
            str(k): rate(sum(1 for row in log if row["solved_round"] is not None
                             and row["solved_round"] <= k), n) for k in range(0, rounds + 1)}

    def breakdown(key):
        groups = defaultdict(list)
        for row in failing:
            groups[str(row.get(key))].append(row)
        return {g: {"n": len(rows), "repair_rate": rate(sum(r["solved_round"] is not None
                                                             for r in rows), len(rows))}
                for g, rows in sorted(groups.items(), key=lambda kv: -len(kv[1]))}

    m["by_bug_type"] = breakdown("bug_type")
    m["by_level"] = breakdown("level")
    m["by_initial_error_type"] = breakdown("initial_error_type")

    first_repair = [row["rounds"][0] if row["rounds"][0]["round"] == 1 else
                    (row["rounds"][1] if len(row["rounds"]) > 1 else None) for row in failing]
    first_repair = [x for x in first_repair if x is not None]
    m["unchanged_rate_round1"] = rate(sum(x.get("unchanged", False) for x in first_repair),
                                      len(first_repair))
    m["avg_repair_rounds_used"] = round(
        sum(sum(1 for x in row["rounds"] if x["round"] > 0) for row in failing) / len(failing), 3) \
        if failing else 0.0
    return m


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description="Multi-round bug-repair evaluation (Phase 5).")
    ap.add_argument("--system", required=True, choices=SYSTEMS)
    ap.add_argument("--eval-set", default="mbpp", choices=EVAL_SETS)
    ap.add_argument("--adapter", default=None, help="adapter dir (default adapters/<system>)")
    ap.add_argument("--model", default=MODEL_ID_DEFAULT,
                    help="model id or local path (MODEL_PATH overrides it on the offline HPC)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N examples")
    ap.add_argument("--rounds", type=int, default=3, help="repair rounds per task")
    ap.add_argument("--few-shot", type=int, default=None, choices=[0, 1],
                    help="few-shot examples in the repair prompt (default: 1 for the baseline, "
                         "0 for the fine-tuned systems, which were trained zero-shot)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                    help="sandbox processes run in parallel")
    ap.add_argument("--out", default=None, help="default results/eval_<system>_<eval-set>.json")
    args = ap.parse_args()

    import torch

    model_id = os.environ.get("MODEL_PATH", args.model)
    few_shot_n = args.few_shot if args.few_shot is not None else (1 if args.system == "baseline" else 0)
    few_shot = default_few_shot()[:few_shot_n]
    examples = load_eval_set(args.eval_set, args.limit)
    print(f"[{args.system}/{args.eval_set}] {len(examples)} examples, {few_shot_n}-shot, "
          f"{args.rounds} rounds, batch {args.batch_size}", flush=True)

    tok, model, info = load_system(args.system, model_id, args.adapter, args.dtype)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    log, gen_seconds = evaluate(examples, model, tok, few_shot, args.rounds, args.batch_size,
                                args.max_new_tokens, args.workers)
    minutes = (time.time() - t0) / 60

    report = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "system": args.system,
        "eval_set": args.eval_set,
        **info,
        "few_shot": few_shot_n,
        "rounds": args.rounds,
        "decoding": "greedy",
        "max_new_tokens": args.max_new_tokens,
        "metrics": summarize(log, args.rounds),
        "efficiency": {
            "total_minutes": round(minutes, 2),
            "generation_minutes": round(gen_seconds / 60, 2),
            "seconds_per_task": round(minutes * 60 / len(examples), 3) if examples else 0.0,
            "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
            if torch.cuda.is_available() else None,
            "batch_size": args.batch_size,
        },
        "per_task": log,
    }
    out = Path(args.out) if args.out else ROOT / "results" / f"eval_{args.system}_{args.eval_set}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    m = report["metrics"]
    extra = f"pass@1 {m['pass_at_1']:.1%}, " if "pass_at_1" in m else ""
    print(f"[{args.system}/{args.eval_set}] {extra}pass rate {m['pass_rate']:.1%}; repair rate by "
          f"round {m['repair_rate_by_round']}; {minutes:.1f} min; wrote {out}")


if __name__ == "__main__":
    # The sandbox starts one process per run from worker threads; forking a threaded process
    # that holds CUDA state is unsafe, so use forkserver where it is available (as mutate.py does).
    if "forkserver" in multiprocessing.get_all_start_methods():
        multiprocessing.set_start_method("forkserver")
    main()
