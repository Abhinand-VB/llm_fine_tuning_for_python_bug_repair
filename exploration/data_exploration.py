"""Static exploration of the MBPP and HumanEval datasets.

Downloads both datasets, computes descriptive statistics and counts potential
mutation sites for the bug-injection pipeline. Code is only parsed with `ast`,
never executed.

Usage:
    python exploration/data_exploration.py

Outputs:
    data/raw/                      downloaded dataset files (git-ignored)
    exploration/stats.json         all computed statistics
    exploration/figures/*.png      charts used in DATA_EXPLORATION.md
"""

import ast
import builtins
import gzip
import json
import statistics
import urllib.request
import warnings
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "exploration"
FIG_DIR = OUT_DIR / "figures"

MBPP_URL = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"
MBPP_SANITIZED_URL = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/sanitized-mbpp.json"
HUMANEVAL_URL = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"

# Official MBPP split by task_id (Austin et al., 2021)
MBPP_SPLITS = {
    "prompt": range(1, 11),
    "test": range(11, 511),
    "validation": range(511, 601),
    "train": range(601, 975),
}

BUILTIN_NAMES = set(dir(builtins))


# ---------------------------------------------------------------- loading

def download(url, name):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / name
    if not path.exists():
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, path)
    return path


def load_mbpp():
    path = download(MBPP_URL, "mbpp.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_mbpp_sanitized():
    path = download(MBPP_SANITIZED_URL, "sanitized-mbpp.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_humaneval():
    path = download(HUMANEVAL_URL, "HumanEval.jsonl.gz")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_of(task_id):
    for name, ids in MBPP_SPLITS.items():
        if task_id in ids:
            return name
    return None


# ---------------------------------------------------------------- analysis

def normalize(code):
    """Normalize line endings and expand tabs so `ast` parses consistently."""
    return code.replace("\r\n", "\n").expandtabs(4)


def safe_parse(code):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            return ast.parse(normalize(code))
    except SyntaxError:
        return None


def has_syntax_warning(code):
    r"""True if the code triggers a SyntaxWarning, e.g. "\d" in a non-raw regex string."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", SyntaxWarning)
        try:
            compile(normalize(code), "<task>", "exec")
        except SyntaxError:
            return False
    return any(issubclass(w.category, SyntaxWarning) for w in caught)


def count_lines(code):
    return len([ln for ln in normalize(code).split("\n") if ln.strip()])


def imported_modules(tree):
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    return mods


def shadowed_builtins(tree):
    """Names that are assigned or used as parameters and also are Python built-ins."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    return sorted(names & BUILTIN_NAMES)


def mutation_sites(tree):
    """Count places where each planned mutation type could be applied."""
    operator = boundary = 0
    for node in ast.walk(tree):
        # Operator mutations: arithmetic, comparison and boolean operators
        if isinstance(node, (ast.BinOp, ast.AugAssign, ast.BoolOp)):
            operator += 1
        elif isinstance(node, ast.Compare):
            operator += len(node.ops)
        # Boundary mutations (off-by-one): range() calls, slice bounds,
        # non-string indices like arr[i], and integer literals in comparisons
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range":
            boundary += 1
        elif isinstance(node, ast.Slice):
            boundary += sum(x is not None for x in (node.lower, node.upper))
        elif isinstance(node, ast.Subscript) and not isinstance(node.slice, ast.Slice) \
                and not (isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)):
            boundary += 1
        if isinstance(node, ast.Compare):
            boundary += sum(isinstance(x, ast.Constant) and type(x.value) is int
                            for x in [node.left, *node.comparators])

    # Variable mutations: a read of a local variable that could be swapped
    # for another local variable in the same function
    variable = 0
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        local = {a.arg for a in func.args.args}
        local |= {n.id for n in ast.walk(func) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        if len(local) < 2:
            continue
        variable += sum(
            1 for n in ast.walk(func)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in local
        )
    return {"operator": operator, "boundary": boundary, "variable": variable}


def summarize(values):
    values = sorted(values)
    return {
        "mean": round(statistics.mean(values), 2),
        "median": statistics.median(values),
        "p90": values[int(0.9 * (len(values) - 1))],
        "max": values[-1],
    }


def analyze_mbpp(entries):
    by_split = {name: [] for name in MBPP_SPLITS}
    for e in entries:
        by_split[split_of(e["task_id"])].append(e)

    split_stats = {}
    for name, rows in by_split.items():
        trees = [safe_parse(r["code"]) for r in rows]
        sites = [mutation_sites(t) for t in trees if t]
        imports = Counter(m for t in trees if t for m in imported_modules(t))
        split_stats[name] = {
            "tasks": len(rows),
            "parse_failures": [r["task_id"] for r, t in zip(rows, trees) if t is None],
            "code_lines": summarize([count_lines(r["code"]) for r in rows]),
            "code_chars": summarize([len(normalize(r["code"])) for r in rows]),
            "problem_chars": summarize([len(r["text"]) for r in rows]),
            "tests_per_task": dict(Counter(len(r["test_list"]) for r in rows)),
            "with_setup_code": sum(bool(r["test_setup_code"].strip()) for r in rows),
            "with_challenge_tests": sum(bool(r["challenge_test_list"]) for r in rows),
            "with_class": sum(any(isinstance(n, ast.ClassDef) for n in ast.walk(t)) for t in trees if t),
            "multi_function": sum(
                sum(isinstance(n, ast.FunctionDef) for n in ast.walk(t)) > 1 for t in trees if t),
            "uses_crlf": sum("\r\n" in r["code"] for r in rows),
            "uses_tabs": sum("\t" in r["code"] for r in rows),
            "syntax_warnings": sum(has_syntax_warning(r["code"]) for r in rows),
            "shadows_builtin": sum(bool(shadowed_builtins(t)) for t in trees if t),
            "top_imports": imports.most_common(10),
            "tasks_with_import": sum(bool(imported_modules(t)) for t in trees if t),
            "mutation_sites_mean": {k: round(statistics.mean(s[k] for s in sites), 2) for k in sites[0]},
            "tasks_with_site": {k: sum(s[k] > 0 for s in sites) for k in sites[0]},
        }

    shadow_counts = Counter(
        name for e in entries if (t := safe_parse(e["code"])) for name in shadowed_builtins(t))
    return by_split, split_stats, shadow_counts.most_common(10)


def analyze_humaneval(entries):
    trees = [safe_parse(e["prompt"] + e["canonical_solution"]) for e in entries]
    sites = [mutation_sites(t) for t in trees if t]
    imports = Counter(m for t in trees if t for m in imported_modules(t))
    return {
        "tasks": len(entries),
        "parse_failures": [e["task_id"] for e, t in zip(entries, trees) if t is None],
        "prompt_lines": summarize([count_lines(e["prompt"]) for e in entries]),
        "solution_lines": summarize([count_lines(e["canonical_solution"]) for e in entries]),
        "asserts_per_task": summarize([e["test"].count("assert") for e in entries]),
        "top_imports": imports.most_common(10),
        "mutation_sites_mean": {k: round(statistics.mean(s[k] for s in sites), 2) for k in sites[0]},
        "tasks_with_site": {k: sum(s[k] > 0 for s in sites) for k in sites[0]},
    }


# ---------------------------------------------------------------- figures

# Reference palette (light mode) from the project's chart style
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES_1 = "#2a78d6"


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.yaxis.label.set_color(TEXT_SECONDARY)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def make_figures(by_split, split_stats, humaneval):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "figure.facecolor": SURFACE})

    # 1. Solution length distribution: small multiples, shared x-axis
    panels = [
        ("MBPP train (374)", [count_lines(r["code"]) for r in by_split["train"]]),
        ("MBPP test (500)", [count_lines(r["code"]) for r in by_split["test"]]),
        ("HumanEval (164)", [count_lines(e["canonical_solution"]) for e in humaneval]),
    ]
    bins = range(0, 32, 2)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharex=True)
    for ax, (title, values) in zip(axes, panels):
        clipped = [min(v, 30) for v in values]
        ax.hist(clipped, bins=bins, color=SERIES_1, edgecolor=SURFACE, linewidth=2)
        ax.set_title(title, fontsize=10, color=TEXT_PRIMARY, loc="left")
        ax.set_xlabel("Non-blank lines in reference solution")
        ax.axvline(statistics.median(values), color=TEXT_SECONDARY, linestyle="--", linewidth=1)
        ax.text(statistics.median(values) + 0.6, ax.get_ylim()[1] * 0.9,
                f"median {statistics.median(values):g}", fontsize=8, color=TEXT_SECONDARY)
        style_axes(ax)
    axes[0].set_ylabel("Tasks")
    fig.suptitle("Solution length (values above 30 lines grouped at 30)",
                 fontsize=11, color=TEXT_PRIMARY, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "solution_length.png", dpi=150)
    plt.close(fig)

    # 2. Mutation site availability on the training split
    train = split_stats["train"]
    labels = ["Operator", "Boundary", "Variable"]
    keys = ["operator", "boundary", "variable"]
    pct = [100 * train["tasks_with_site"][k] / train["tasks"] for k in keys]
    fig, ax = plt.subplots(figsize=(7, 2.6))
    bars = ax.barh(labels[::-1], pct[::-1], color=SERIES_1, height=0.55)
    for bar, k in zip(bars, keys[::-1]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.0f}%  (avg {train['mutation_sites_mean'][k]:g} sites/task)",
                va="center", fontsize=9, color=TEXT_PRIMARY)
    ax.set_xlim(0, 130)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("% of MBPP training tasks with at least one site")
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_title("Where bugs can be injected (MBPP train)", fontsize=11, color=TEXT_PRIMARY, loc="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mutation_sites.png", dpi=150)
    plt.close(fig)

    # 3. Most common imports across MBPP train
    top = train["top_imports"][:8]
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.barh([m for m, _ in top][::-1], [c for _, c in top][::-1], color=SERIES_1, height=0.55)
    for i, (_, c) in enumerate(top[::-1]):
        ax.text(c + 0.3, i, str(c), va="center", fontsize=9, color=TEXT_PRIMARY)
    ax.set_xlabel("MBPP training tasks importing the module")
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_title("Modules the sandbox must allow (MBPP train)", fontsize=11, color=TEXT_PRIMARY, loc="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "imports.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    mbpp = load_mbpp()
    sanitized = load_mbpp_sanitized()
    humaneval = load_humaneval()

    by_split, split_stats, shadow_top = analyze_mbpp(mbpp)
    he_stats = analyze_humaneval(humaneval)

    sanitized_by_split = Counter(split_of(e["task_id"]) for e in sanitized)

    stats = {
        "mbpp_total": len(mbpp),
        "mbpp_splits": split_stats,
        "mbpp_sanitized_by_split": dict(sanitized_by_split),
        "mbpp_shadowed_builtins_top": shadow_top,
        "humaneval": he_stats,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    make_figures(by_split, split_stats, humaneval)

    print(f"MBPP: {len(mbpp)} tasks | sanitized: {len(sanitized)} | HumanEval: {len(humaneval)}")
    for name, s in split_stats.items():
        print(f"  {name:<10} {s['tasks']:>4} tasks, median {s['code_lines']['median']} lines")
    print(f"Wrote {OUT_DIR / 'stats.json'} and figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
