"""Measure training-example lengths in tokens with the Qwen2.5-Coder tokenizer.

Builds one repair example per MBPP task in our training format (proposal, Section 6):

    Problem: <text>
    Buggy code: <code>
    Test feedback: <error message>
    Corrected code: <code>

Bugs have not been generated yet, so the reference solution stands in for both the
buggy and the corrected code (a mutation only changes a few tokens). The feedback is
"AssertionError: " followed by the task's first test line.

The problem, buggy code and feedback form the user message and the corrected code is
the assistant reply, wrapped with the model's chat template (this includes Qwen's
default system prompt, so it is the length the trainer will actually see).

Code is normalised the same way the mutation pipeline will do it (\r\n -> \n, tabs ->
4 spaces). Nothing is executed.

Usage:
    pip install transformers matplotlib jinja2
    python exploration/token_lengths.py
    python exploration/token_lengths.py --tokenizer /path/to/local/tokenizer   # offline

Outputs:
    exploration/token_lengths.json           all statistics
    exploration/figures/token_lengths.png    histogram with the 1,024 limit marked
"""

import argparse
import json
import statistics
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "exploration"
FIG_DIR = OUT_DIR / "figures"

MBPP_URL = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"
MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

# Official MBPP split by task_id (same as data_exploration.py)
SPLITS = {
    "train": range(601, 975),
    "validation": range(511, 601),
    "test": range(11, 511),
}
LIMITS = (512, 1024)


# ---------------------------------------------------------------- data

def load_mbpp():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / "mbpp.jsonl"
    if not path.exists():
        print(f"Downloading {MBPP_URL}")
        urllib.request.urlretrieve(MBPP_URL, path)
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalise(code):
    return code.replace("\r\n", "\n").expandtabs(4).strip()


# ---------------------------------------------------------------- examples

def build_messages(task, all_tests=False):
    code = normalise(task["code"])
    tests = task["test_list"] if all_tests else task["test_list"][:1]
    feedback = "\n".join(f"AssertionError: {t.strip()}" for t in tests)
    user = (
        f"Problem: {task['text'].strip()}\n\n"
        f"Buggy code:\n{code}\n\n"
        f"Test feedback:\n{feedback}\n\n"
        f"Corrected code:"
    )
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": code},
    ]


def count_tokens(tok, task, all_tests=False):
    messages = build_messages(task, all_tests)
    text = tok.apply_chat_template(messages, tokenize=False)
    return len(tok(text).input_ids)


def summarise(values):
    s = sorted(values)
    p90 = s[min(len(s) - 1, int(round(0.9 * (len(s) - 1))))]
    out = {
        "n": len(s),
        "median": statistics.median(s),
        "p90": p90,
        "max": s[-1],
        "mean": round(statistics.mean(s), 1),
    }
    for limit in LIMITS:
        out[f"over_{limit}"] = sum(v > limit for v in s)
    return out


# ---------------------------------------------------------------- figure

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES_1 = "#2a78d6"
LIMIT_COLOR = "#c2410c"


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


def make_figure(lengths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "figure.facecolor": SURFACE})

    x_max = 1100
    bins = range(0, x_max + 25, 25)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharex=True)
    for ax, (split, values) in zip(axes, lengths.items()):
        med = statistics.median(values)
        ax.hist([min(v, x_max) for v in values], bins=bins, color=SERIES_1,
                edgecolor=SURFACE, linewidth=0.8)
        ax.axvline(1024, color=LIMIT_COLOR, linewidth=1.5)
        ax.axvline(med, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
        top = ax.get_ylim()[1]
        ax.text(1010, top * 0.92, "1,024 limit", fontsize=8, color=LIMIT_COLOR, ha="right")
        ax.text(med + 15, top * 0.80, f"median {med:g}", fontsize=8, color=TEXT_SECONDARY)
        ax.set_title(f"MBPP {split} ({len(values)})  max {max(values)}",
                     fontsize=10, color=TEXT_PRIMARY, loc="left")
        ax.set_xlabel("Tokens per training example")
        style_axes(ax)
    axes[0].set_ylabel("Tasks")
    fig.suptitle("Training example length, Qwen2.5-Coder tokenizer with chat template "
                 f"(values above {x_max} grouped at {x_max})",
                 fontsize=11, color=TEXT_PRIMARY, x=0.01, ha="left")
    fig.tight_layout()
    path = FIG_DIR / "token_lengths.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default=MODEL_ID,
                        help="Hugging Face model id or local tokenizer folder")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    tasks = load_mbpp()
    lengths, lengths_all_tests, stats = {}, {}, {}
    for split, ids in SPLITS.items():
        rows = [t for t in tasks if t["task_id"] in ids]
        lengths[split] = [count_tokens(tok, t) for t in rows]
        lengths_all_tests[split] = [count_tokens(tok, t, all_tests=True) for t in rows]
        longest = max(rows, key=lambda t: count_tokens(tok, t))
        stats[split] = {
            "one_test_feedback": summarise(lengths[split]),
            "all_tests_feedback": summarise(lengths_all_tests[split]),
            "longest_task_id": longest["task_id"],
        }

    # Fixed overhead: chat template + system prompt + field labels with empty content
    empty = {"text": "", "code": "", "test_list": [""]}
    stats["template_overhead_tokens"] = count_tokens(tok, empty)

    print(f"{'split':<11}{'n':>5}{'median':>8}{'p90':>6}{'max':>6}{'>512':>6}{'>1024':>7}   (all-3-tests max)")
    for split in SPLITS:
        s = stats[split]["one_test_feedback"]
        a = stats[split]["all_tests_feedback"]
        print(f"{split:<11}{s['n']:>5}{s['median']:>8g}{s['p90']:>6}{s['max']:>6}"
              f"{s['over_512']:>6}{s['over_1024']:>7}   ({a['max']}, task {stats[split]['longest_task_id']})")
    print(f"Template overhead: {stats['template_overhead_tokens']} tokens")

    (OUT_DIR / "token_lengths.json").write_text(json.dumps(stats, indent=2))
    fig = make_figure(lengths)
    print(f"Wrote {OUT_DIR / 'token_lengths.json'} and {fig}")


if __name__ == "__main__":
    main()
