"""Bug-mutation pipeline (v1): build (buggy code, feedback) -> fixed code examples from MBPP.

For every task in the chosen MBPP splits:
  1. Normalize the reference solution (line endings, tabs) and re-serialize it with
     `ast.unparse`, so the fixed and buggy code share the same formatting.
  2. Run the normalized reference in the sandbox. Tasks whose reference fails are dropped.
  3. Enumerate every single-mutation site of the three bug types (operator, boundary,
     variable) and build one candidate mutant per site.
  4. Run up to MAX_TRIES sampled candidates per task and bug type in the sandbox. A mutant is
     usable only if it fails at least one test; mutants that pass all three tests are
     equivalent or undetected and are discarded.
  5. Keep at most CAP failing mutants per task and bug type, to balance the bug types.

Only the train (601-974) and validation (511-600) splits are mutated by default. Test
mutants for the held-out evaluation can be generated with `--splits test`; they are written to
their own file and must never be used for training.

Usage (Linux, macOS or WSL; the sandbox uses Unix-only modules):
    python src/mutate.py
    python src/mutate.py --splits train --limit 20     # quick smoke test

Outputs:
    data/processed/{split}.jsonl        one example per line (git-ignored)
    exploration/mutation_stats.json     yield statistics per split and bug type
"""

import argparse
import ast
import builtins
import copy
import json
import multiprocessing
import random
import re
import sys
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sandbox import evaluate_code  # noqa: E402  (sandbox.py lives at the repo root)

RAW_MBPP = ROOT / "data" / "raw" / "mbpp.jsonl"
MBPP_URL = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"
OUT_DIR = ROOT / "data" / "processed"
STATS_PATH = ROOT / "exploration" / "mutation_stats.json"

MBPP_SPLITS = {
    "test": range(11, 511),
    "validation": range(511, 601),
    "train": range(601, 975),
}
BUG_TYPES = ("operator", "boundary", "variable")

SEED = 258
MAX_TRIES = 10      # candidates run per task and bug type
CAP = 3             # failing mutants kept per task and bug type
TIMEOUT = 3.0       # seconds per sandbox run
MAX_FEEDBACK_CHARS = 500
MAX_FRAMES = 3      # innermost traceback frames kept in the feedback

BUILTIN_NAMES = set(dir(builtins))

# Operator swaps. Each operator maps to one plausible mistake.
BINOP_SWAP = {
    ast.Add: ast.Sub, ast.Sub: ast.Add,
    ast.Mult: ast.Add, ast.Div: ast.Mult,
    ast.FloorDiv: ast.Div, ast.Mod: ast.FloorDiv,
    ast.Pow: ast.Mult,
}
CMPOP_SWAP = {
    ast.Lt: ast.LtE, ast.LtE: ast.Lt,
    ast.Gt: ast.GtE, ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
    ast.In: ast.NotIn, ast.NotIn: ast.In,
    ast.Is: ast.IsNot, ast.IsNot: ast.Is,
}
BOOLOP_SWAP = {ast.And: ast.Or, ast.Or: ast.And}

OP_SYMBOL = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//",
    ast.Mod: "%", ast.Pow: "**", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
    ast.Eq: "==", ast.NotEq: "!=", ast.In: "in", ast.NotIn: "not in", ast.Is: "is",
    ast.IsNot: "is not", ast.And: "and", ast.Or: "or",
}


# ---------------------------------------------------------------- loading

def load_mbpp():
    if not RAW_MBPP.exists():
        import urllib.request
        RAW_MBPP.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {MBPP_URL}")
        urllib.request.urlretrieve(MBPP_URL, RAW_MBPP)
    with open(RAW_MBPP, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize(code):
    """Normalize line endings, expand tabs and re-serialize with ast.unparse."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(code.replace("\r\n", "\n").expandtabs(4))
    return ast.unparse(tree)


# ---------------------------------------------------------------- mutation sites

def shift(expr, delta):
    """Return `expr` moved by +/-1, folding into an existing integer constant when possible."""
    if isinstance(expr, ast.Constant) and type(expr.value) is int:
        return ast.Constant(expr.value + delta)
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.USub) \
            and isinstance(expr.operand, ast.Constant) and type(expr.operand.value) is int:
        return ast.Constant(-expr.operand.value + delta)
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, (ast.Add, ast.Sub)) \
            and isinstance(expr.right, ast.Constant) and type(expr.right.value) is int:
        value = expr.right.value + (delta if isinstance(expr.op, ast.Add) else -delta)
        if value == 0:
            return expr.left
        return ast.BinOp(expr.left, type(expr.op)(), ast.Constant(value))
    return ast.BinOp(expr, ast.Add() if delta > 0 else ast.Sub(), ast.Constant(abs(delta)))


def function_locals(func):
    """Map each parameter and assigned name of a function to the line where it is first defined.

    Names that shadow built-ins and comprehension variables (which only exist inside the
    comprehension) are left out.
    """
    args = func.args
    first = {a.arg: func.lineno for a in args.posonlyargs + args.args + args.kwonlyargs}
    first |= {a.arg: func.lineno for a in (args.vararg, args.kwarg) if a}
    comp_targets = {id(n) for c in ast.walk(func) if isinstance(c, ast.comprehension)
                    for n in ast.walk(c.target)}
    for n in ast.walk(func):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and id(n) not in comp_targets:
            first[n.id] = min(first.get(n.id, n.lineno), n.lineno)
    return {name: line for name, line in first.items() if name not in BUILTIN_NAMES}


def find_sites(tree):
    """List every single-mutation site as (bug_type, walk_index, action).

    `walk_index` is the node's position in `ast.walk(tree)`. A deep copy of the tree walks in
    the same order, so the site can be found again in the copy and mutated there.
    """
    nodes = list(ast.walk(tree))
    sites = []
    for i, node in enumerate(nodes):
        # Operator
        if isinstance(node, (ast.BinOp, ast.AugAssign)) and type(node.op) in BINOP_SWAP:
            sites.append(("operator", i, ("op",)))
        elif isinstance(node, ast.BoolOp):
            sites.append(("operator", i, ("op",)))
        elif isinstance(node, ast.Compare):
            for k, op in enumerate(node.ops):
                if type(op) in CMPOP_SWAP:
                    sites.append(("operator", i, ("cmp", k)))

        # Boundary (off-by-one)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range" \
                and 1 <= len(node.args) <= 3 and not node.keywords:
            stop = 0 if len(node.args) == 1 else 1
            for d in (1, -1):
                sites.append(("boundary", i, ("range", stop, d)))
        elif isinstance(node, ast.Slice):
            for attr in ("lower", "upper"):
                if getattr(node, attr) is not None:
                    for d in (1, -1):
                        sites.append(("boundary", i, ("slice", attr, d)))
        elif isinstance(node, ast.Subscript) and not isinstance(node.slice, (ast.Slice, ast.Tuple)) \
                and not (isinstance(node.slice, ast.Constant) and not isinstance(node.slice.value, int)):
            for d in (1, -1):
                sites.append(("boundary", i, ("index", d)))
        if isinstance(node, ast.Compare):
            for k, side in enumerate([node.left, *node.comparators]):
                if isinstance(side, ast.Constant) and type(side.value) is int:
                    for d in (1, -1):
                        sites.append(("boundary", i, ("cmpconst", k, d)))

    # Variable: replace a read of a local with another local of the same function that is
    # defined on an earlier line (otherwise many mutants are a trivial UnboundLocalError)
    for func in nodes:
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        local = function_locals(func)
        if len(local) < 2:
            continue
        for node in ast.walk(func):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in local:
                i = nodes.index(node)
                for other in sorted(local):
                    if other != node.id and local[other] < node.lineno:
                        sites.append(("variable", i, ("var", other)))
    return sites


def apply_site(tree, walk_index, action):
    """Apply one mutation to a copy of `tree`. Returns (mutated tree, description)."""
    tree = copy.deepcopy(tree)
    node = list(ast.walk(tree))[walk_index]
    kind = action[0]
    line = getattr(node, "lineno", None)

    if kind == "op":
        old = type(node.op)
        new = BOOLOP_SWAP[old] if isinstance(node, ast.BoolOp) else BINOP_SWAP[old]
        node.op = new()
        desc = f"{OP_SYMBOL[old]} -> {OP_SYMBOL[new]}"
    elif kind == "cmp":
        k = action[1]
        old = type(node.ops[k])
        node.ops[k] = CMPOP_SWAP[old]()
        desc = f"{OP_SYMBOL[old]} -> {OP_SYMBOL[CMPOP_SWAP[old]]}"
    elif kind == "range":
        _, k, d = action
        before = ast.unparse(node)
        node.args[k] = shift(node.args[k], d)
        desc = f"{before} -> {ast.unparse(node)}"
    elif kind == "slice":
        _, attr, d = action
        before = ast.unparse(node)
        setattr(node, attr, shift(getattr(node, attr), d))
        desc = f"[{before}] -> [{ast.unparse(node)}]"
    elif kind == "index":
        before = ast.unparse(node)
        node.slice = shift(node.slice, action[1])
        desc = f"{before} -> {ast.unparse(node)}"
    elif kind == "cmpconst":
        _, k, d = action
        before = ast.unparse(node)
        if k == 0:
            node.left = shift(node.left, d)
        else:
            node.comparators[k - 1] = shift(node.comparators[k - 1], d)
        desc = f"{before} -> {ast.unparse(node)}"
    elif kind == "var":
        desc = f"{node.id} -> {action[1]}"
        node.id = action[1]
    else:
        raise ValueError(kind)

    ast.fix_missing_locations(tree)
    return tree, f"line {line}: {desc}" if line else desc


# ---------------------------------------------------------------- execution feedback

FRAME = re.compile(r'^\s*File "(.*?)", line (\d+), in (.*)$')
TIMEOUT_MSG = "TimeoutError: Execution exceeded the time limit."


def run(code, tests, setup):
    """Run code against its tests. The tests are joined into one string so the traceback
    line number tells us which test failed."""
    return evaluate_code(code, ["\n".join(tests)], setup, timeout_seconds=TIMEOUT)


def format_feedback(code, tests, error):
    """Turn the sandbox traceback into short, readable test feedback.

    Frames inside sandbox.py are dropped. Frames in the solution point at the buggy code's
    own line numbers, and the test-module frame is replaced by the failing assert.
    Returns (feedback, error_type).
    """
    if error.startswith("TimeoutError"):
        return (f"TimeoutError: the program did not finish within {TIMEOUT:g} seconds "
                "(possible infinite loop).", "TimeoutError")
    if not error.startswith("Traceback"):
        return error.strip(), error.split(":", 1)[0]

    lines = error.rstrip().split("\n")
    code_lines = code.split("\n")
    failed_test = None
    frames = []         # [frame text, times called in a row]
    in_tests = False    # whether the sandbox was exec'ing the tests rather than the solution
    i = 1
    while i < len(lines):
        repeat = re.match(r"\s*\[Previous line repeated (\d+) more times?\]", lines[i])
        if repeat:
            if frames:
                frames[-1][1] += int(repeat.group(1))
            i += 1
            continue
        m = FRAME.match(lines[i])
        if not m:
            break
        path, lineno, func = m.group(1), int(m.group(2)), m.group(3)
        i += 1
        source = lines[i].strip() if i < len(lines) and lines[i].startswith("    ") else ""
        # Skip the source and caret lines Python prints under a frame
        while i < len(lines) and lines[i].startswith("    "):
            i += 1
        if path != "<string>":
            in_tests = source.startswith("exec(t,")
            continue
        if func == "<module>" and in_tests:
            if failed_test is None and 1 <= lineno <= len(tests):
                failed_test = tests[lineno - 1].strip()
        elif 1 <= lineno <= len(code_lines):
            frame = f"  line {lineno}, in {func}\n    {code_lines[lineno - 1].strip()}"
            if frames and frames[-1][0] == frame:
                frames[-1][1] += 1
            else:
                frames.append([frame, 1])
    message = "\n".join(lines[i:]).strip()
    error_type = message.split(":", 1)[0].strip() or "Error"

    # Collapse deep recursion and keep only the innermost frames
    frames = [f + (f"\n  [called {n} times in a row]" if n > 1 else "") for f, n in frames[-MAX_FRAMES:]]

    parts = []
    if failed_test:
        parts.append(f"Failed test: {truncate(failed_test, 200)}")
    if frames:
        parts.append("Traceback (most recent call last):\n" + "\n".join(frames))
    parts.append(truncate(message, 200))
    return truncate("\n".join(parts), MAX_FEEDBACK_CHARS), error_type


def truncate(text, limit):
    return text if len(text) <= limit else text[:limit - 3] + "..."


# ---------------------------------------------------------------- pipeline

def candidates_for_task(fixed_code, rng):
    """All unique single mutants of `fixed_code`, grouped by bug type and shuffled."""
    tree = ast.parse(fixed_code)
    by_type = defaultdict(list)
    seen = {fixed_code}
    for bug_type, walk_index, action in find_sites(tree):
        mutated, desc = apply_site(tree, walk_index, action)
        try:
            buggy = ast.unparse(mutated)
            compile(buggy, "<mutant>", "exec")
        except (SyntaxError, ValueError):
            continue
        if buggy in seen:
            continue
        seen.add(buggy)
        by_type[bug_type].append((buggy, desc))
    for items in by_type.values():
        rng.shuffle(items)
    return by_type


def process_split(name, tasks, pool):
    rng = random.Random(f"{SEED}-{name}")
    stats = {
        "tasks": len(tasks),
        "reference_failed": [],
        "tasks_with_example": 0,
        "by_type": {t: Counter() for t in BUG_TYPES},
        "error_types": {t: Counter() for t in BUG_TYPES},
    }

    # 1-2. Normalize and validate the references
    prepared = []
    for task in tasks:
        try:
            fixed = normalize(task["code"])
        except SyntaxError:
            stats["reference_failed"].append(task["task_id"])
            continue
        prepared.append((task, fixed))
    ref_results = pool.map(lambda p: run(p[1], p[0]["test_list"], p[0]["test_setup_code"]), prepared)
    valid = []
    for (task, fixed), (passed, _) in zip(prepared, ref_results):
        if passed:
            valid.append((task, fixed))
        else:
            stats["reference_failed"].append(task["task_id"])
    print(f"[{name}] {len(valid)}/{len(tasks)} references pass after normalization")

    # 3. Enumerate candidates, 4. pick up to MAX_TRIES per task and bug type
    jobs = []
    for task, fixed in valid:
        by_type = candidates_for_task(fixed, rng)
        for bug_type in BUG_TYPES:
            items = by_type.get(bug_type, [])
            stats["by_type"][bug_type]["candidates"] += len(items)
            stats["by_type"][bug_type]["tasks_with_site"] += bool(items)
            for buggy, desc in items[:MAX_TRIES]:
                jobs.append((task, fixed, bug_type, buggy, desc))
    print(f"[{name}] running {len(jobs)} mutants in the sandbox")
    results = pool.map(lambda j: run(j[3], j[0]["test_list"], j[0]["test_setup_code"]), jobs)

    # 5. Keep failing mutants, capped per task and bug type
    examples = []
    kept = Counter()
    for (task, fixed, bug_type, buggy, desc), (passed, error) in zip(jobs, results):
        counts = stats["by_type"][bug_type]
        counts["run"] += 1
        if passed:
            counts["passed_tests"] += 1
            continue
        counts["failed_tests"] += 1
        feedback, error_type = format_feedback(buggy, task["test_list"], error or "")
        stats["error_types"][bug_type][error_type] += 1
        if kept[task["task_id"], bug_type] >= CAP:
            continue
        kept[task["task_id"], bug_type] += 1
        counts["kept"] += 1
        examples.append({
            "task_id": task["task_id"],
            "split": name,
            "problem": task["text"],
            "tests": task["test_list"],
            "buggy_code": buggy,
            "feedback": feedback,
            "fixed_code": fixed,
            "bug_type": bug_type,
            "mutation": desc,
            "error_type": error_type,
            "level": 1,
            "round": 1,
        })
    stats["tasks_with_example"] = len({e["task_id"] for e in examples})
    stats["examples"] = len(examples)
    return examples, stats


def summarize(stats):
    out = {k: v for k, v in stats.items() if k not in ("by_type", "error_types")}
    out["reference_failed"] = sorted(stats["reference_failed"])
    out["by_type"] = {}
    for bug_type in BUG_TYPES:
        c = stats["by_type"][bug_type]
        run_count = c["run"] or 1
        out["by_type"][bug_type] = {
            "tasks_with_site": c["tasks_with_site"],
            "candidates": c["candidates"],
            "run": c["run"],
            "failed_tests": c["failed_tests"],
            "passed_tests": c["passed_tests"],
            "failure_rate": round(c["failed_tests"] / run_count, 3),
            "kept": c["kept"],
            "error_types": dict(stats["error_types"][bug_type].most_common()),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--splits", nargs="+", default=["train", "validation"], choices=list(MBPP_SPLITS))
    parser.add_argument("--limit", type=int, help="only use the first N tasks of each split (smoke test)")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count())
    args = parser.parse_args()

    entries = load_mbpp()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_stats = {"seed": SEED, "max_tries": MAX_TRIES, "cap": CAP, "timeout_s": TIMEOUT, "splits": {}}

    with ThreadPoolExecutor(args.workers) as pool:
        for name in args.splits:
            tasks = [e for e in entries if e["task_id"] in MBPP_SPLITS[name]][:args.limit]
            examples, stats = process_split(name, tasks, pool)
            path = OUT_DIR / f"{name}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for ex in examples:
                    f.write(json.dumps(ex) + "\n")
            all_stats["splits"][name] = summarize(stats)
            print(f"[{name}] wrote {len(examples)} examples to {path.relative_to(ROOT)}")

    if args.limit is None:
        STATS_PATH.write_text(json.dumps(all_stats, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {STATS_PATH.relative_to(ROOT)}")
    else:
        print(json.dumps(all_stats, indent=2))


if __name__ == "__main__":
    # The sandbox starts a process per run from worker threads; forking a threaded
    # process is unsafe, so use forkserver where it is available.
    if "forkserver" in multiprocessing.get_all_start_methods():
        multiprocessing.set_start_method("forkserver")
    main()
