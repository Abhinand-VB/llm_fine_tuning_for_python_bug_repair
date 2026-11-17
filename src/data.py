"""Loading the evaluation sets for the multi-round repair harness (PROJECT_PLAN.md, Phase 5).

Every evaluation example has the same shape as a row of data/processed/*.jsonl, so the harness
(src/evaluate.py) treats MBPP and HumanEval the same way:

    task_id, problem, tests, buggy_code, feedback, fixed_code, bug_type, level, error_type,
    preamble (HumanEval only: code the prompt defines before the target function)

Evaluation sets:
    mbpp           held-out MBPP test mutants (data/processed/test.jsonl, from src/mutate.py)
    humaneval      HumanEval option (b): canonical solutions mutated by the same pipeline
                   (data/processed/humaneval_mutants.jsonl, built by `python src/data.py`)
    humaneval-gen  HumanEval option (a): no buggy code; the model first writes a solution
                   from the prompt (round 0 = pass@1), then repairs it with test feedback

HumanEval tests are a `check(candidate)` function. We inline its body at module level with
`candidate` renamed to the entry point, then split it into lines. That makes the tests look like
MBPP's assert list, so the sandbox and the feedback formatter from src/mutate.py work unchanged
and the feedback names the failing assert.

Usage (Linux, macOS or WSL; the sandbox uses Unix-only modules):
    python src/data.py          # build data/processed/humaneval_mutants.jsonl
"""

import ast
import gzip
import json
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

PROCESSED = ROOT / "data" / "processed"
HUMANEVAL_RAW = ROOT / "data" / "raw" / "HumanEval.jsonl.gz"
HUMANEVAL_MUTANTS = PROCESSED / "humaneval_mutants.jsonl"
HUMANEVAL_SEED = 258
HUMANEVAL_CAP = 1          # failing mutants kept per task and bug type (like the MBPP test split)
HUMANEVAL_MAX_TRIES = 10   # candidates run per task and bug type

EVAL_SETS = ("mbpp", "humaneval", "humaneval-gen")


def read_jsonl(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


# ------------------------------------------------------------------ HumanEval

def load_humaneval_raw():
    with gzip.open(HUMANEVAL_RAW, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class _Rename(ast.NodeTransformer):
    def __init__(self, old, new):
        self.old, self.new = old, new

    def visit_Name(self, node):
        if node.id == self.old:
            node.id = self.new
        return node


def humaneval_tests(test_code, entry_point):
    """HumanEval's `check(candidate)` as a flat, MBPP-style list of test lines.

    Top-level statements other than `def check` (METADATA, imports) are kept; the body of
    check is inlined with `candidate` renamed to the entry point. No HumanEval check body
    contains a `return`, so inlining preserves the semantics.
    """
    tree = ast.parse(test_code)
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "check":
            arg = node.args.args[0].arg if node.args.args else "candidate"
            for stmt in node.body:
                body.append(_Rename(arg, entry_point).visit(stmt))
        else:
            body.append(node)
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    return [line for line in ast.unparse(module).split("\n") if line.strip()]


def split_prompt(prompt, entry_point):
    """Split a HumanEval prompt into (preamble, target function header + docstring).

    The preamble holds the imports and any helper functions defined before the target (e.g.
    `encode_cyclic` for HumanEval/38). It is prepended to every candidate before it runs, so a
    model that returns only the target function is not penalized for dropping the imports.
    """
    marker = f"def {entry_point}("
    idx = prompt.find(marker)
    if idx < 0:
        return "", prompt
    return prompt[:idx].rstrip() + "\n", prompt[idx:]


def humaneval_problem(row):
    """The problem statement shown to the model: the full HumanEval prompt (signature, docstring
    and examples)."""
    return "Complete the following Python function so that it satisfies its docstring.\n\n" + \
        row["prompt"].strip()


def build_humaneval_mutants(workers=8):
    """HumanEval option (b): mutate canonical solutions with the MBPP pipeline.

    Each solution (prompt + canonical body) is normalized with ast.unparse, checked against its
    tests, and every single-mutation site is tried; at most HUMANEVAL_CAP failing mutant is
    kept per task and bug type. The buggy code keeps the docstring, so the model sees exactly
    the function it has to fix.
    """
    import mutate

    rng = random.Random(HUMANEVAL_SEED)
    rows = load_humaneval_raw()
    examples, stats = [], Counter()
    dropped = []

    with ThreadPoolExecutor(workers) as pool:
        for row in rows:
            entry = row["entry_point"]
            preamble, _ = split_prompt(row["prompt"], entry)
            tests = humaneval_tests(row["test"], entry)
            fixed = mutate.normalize(row["prompt"] + row["canonical_solution"])
            ok, _ = mutate.run(fixed, tests, "")
            if not ok:
                dropped.append(row["task_id"])
                continue
            by_type, _ = mutate.candidates_for_task(fixed, rng)
            for bug_type in mutate.BUG_TYPES:
                cands = by_type.get(bug_type, [])[:HUMANEVAL_MAX_TRIES]
                stats[f"{bug_type}_generated"] += len(cands)
                results = list(pool.map(lambda c: mutate.run(c[0], tests, ""), cands))
                kept = 0
                for (buggy, desc, _), (passed, error) in zip(cands, results):
                    if passed or kept >= HUMANEVAL_CAP:
                        continue
                    feedback, error_type = mutate.format_feedback(buggy, tests, error or "")
                    examples.append({
                        "task_id": row["task_id"], "split": "humaneval",
                        "problem": humaneval_problem(row), "tests": tests,
                        "buggy_code": buggy, "feedback": feedback, "fixed_code": fixed,
                        "bug_type": bug_type, "mutation": desc, "num_mutations": 1,
                        "error_type": error_type, "level": 1, "round": 1,
                        # Prepended before the model's code runs; the model's own definitions
                        # come after it, so they win if it restates the helpers.
                        "entry_point": entry, "preamble": preamble,
                    })
                    kept += 1
                stats[f"{bug_type}_kept"] += kept

    PROCESSED.mkdir(parents=True, exist_ok=True)
    with open(HUMANEVAL_MUTANTS, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    tasks = len({ex["task_id"] for ex in examples})
    print(f"HumanEval: {len(rows)} tasks, {len(dropped)} canonical solutions failed {dropped}")
    print(f"Wrote {len(examples)} mutants over {tasks} tasks to {HUMANEVAL_MUTANTS.relative_to(ROOT)}")
    print(dict(stats))
    print(Counter(ex["error_type"] for ex in examples).most_common())
    return examples


def load_eval_set(name, limit=None):
    """Load one evaluation set as a list of examples (see the module docstring)."""
    if name == "mbpp":
        return read_jsonl(PROCESSED / "test.jsonl", limit)
    if name == "humaneval":
        if not HUMANEVAL_MUTANTS.exists():
            raise FileNotFoundError(f"{HUMANEVAL_MUTANTS} is missing; run `python src/data.py`")
        return read_jsonl(HUMANEVAL_MUTANTS, limit)
    if name == "humaneval-gen":
        out = []
        for row in load_humaneval_raw()[:limit]:
            entry = row["entry_point"]
            preamble, _ = split_prompt(row["prompt"], entry)
            out.append({
                "task_id": row["task_id"], "split": "humaneval",
                "problem": humaneval_problem(row), "prompt": row["prompt"],
                "tests": humaneval_tests(row["test"], entry),
                "buggy_code": None, "feedback": "", "fixed_code": None,
                "bug_type": "generation", "level": 0, "error_type": None,
                "entry_point": entry, "preamble": preamble,
            })
        return out
    raise ValueError(f"unknown evaluation set {name!r}; choose from {EVAL_SETS}")


if __name__ == "__main__":
    import multiprocessing
    if "forkserver" in multiprocessing.get_all_start_methods():
        multiprocessing.set_start_method("forkserver")
    build_humaneval_mutants()
