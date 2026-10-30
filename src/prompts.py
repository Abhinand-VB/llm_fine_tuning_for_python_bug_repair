"""Prompt format and code extraction for the bug-repair task (PROJECT_PLAN.md, Phase 3).

One chat-template prompt is used everywhere (baseline, training and evaluation) so the three
systems see the same inputs. A turn gives the model the problem statement, the current code and
the latest test feedback, and asks it to return only the corrected function in a single code
block. The model's reply is then run through `extract_code` to recover the function before it
goes back to the sandbox.

Keeping this in one place means the prompt and the extractor can be tuned once; `src/train.py`
and `src/evaluate.py` import from here rather than re-deriving the format.

Functions:
    build_messages(problem, code, feedback, few_shot=...) -> list[dict]   chat messages
    render_prompt(tokenizer, problem, code, feedback, ...) -> str         templated prompt string
    extract_code(text) -> str                                            function out of model output
    default_few_shot() -> list[dict]                                     1 illustrative repair example
"""

import re

SYSTEM_PROMPT = (
    "You are an expert Python programmer fixing buggy code. You are given a problem "
    "description, a candidate solution that fails its unit tests, and the feedback from "
    "running those tests. Return the corrected solution as a single Python function (plus any "
    "helper definitions it needs) inside one ```python code block, and nothing else: no "
    "explanation, no extra prose."
)

# MBPP tasks 1-10 are reserved for few-shot prompting and are never mutated into the train,
# validation or test data (see src/mutate.py), so showing one here cannot leak into evaluation.
# This illustrative repair makes the expected format explicit: a boundary (off-by-one) bug that
# an AssertionError exposes, and a corrected function in a single code block.
_FEW_SHOT = [
    {
        "problem": "Write a function to return the factorial of a non-negative integer n.",
        "code": (
            "def factorial(n):\n"
            "    result = 1\n"
            "    for i in range(1, n):\n"
            "        result *= i\n"
            "    return result"
        ),
        "feedback": (
            "Failed test: assert factorial(5) == 120\n"
            "AssertionError"
        ),
        "fixed": (
            "def factorial(n):\n"
            "    result = 1\n"
            "    for i in range(1, n + 1):\n"
            "        result *= i\n"
            "    return result"
        ),
    },
]


def default_few_shot():
    """One illustrative repair example drawn from the reserved MBPP 1-10 range.

    Return a copy so callers can slice it (e.g. `default_few_shot()[:0]` for zero-shot) without
    mutating the module-level list.
    """
    return [dict(ex) for ex in _FEW_SHOT]


def _user_turn(problem, code, feedback):
    """The user half of one repair turn: problem, current code and the test feedback."""
    parts = [
        "Problem:",
        problem.strip(),
        "",
        "Current code:",
        "```python",
        code.strip(),
        "```",
    ]
    feedback = (feedback or "").strip()
    if feedback:
        parts += ["", "Test feedback:", feedback]
    parts += ["", "Return the corrected function."]
    return "\n".join(parts)


def _assistant_turn(fixed):
    """The assistant half of a few-shot example: the fixed code in one block."""
    return f"```python\n{fixed.strip()}\n```"


def build_messages(problem, code, feedback, few_shot=None):
    """Build the chat `messages` list for one repair turn.

    `few_shot` is a list of {problem, code, feedback, fixed} dicts rendered as prior
    user/assistant turns; pass `default_few_shot()` for the standard 1-shot prompt or `[]` for
    zero-shot. The final user turn is the task to solve.
    """
    if few_shot is None:
        few_shot = default_few_shot()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in few_shot:
        messages.append({"role": "user",
                         "content": _user_turn(ex["problem"], ex["code"], ex.get("feedback", ""))})
        messages.append({"role": "assistant", "content": _assistant_turn(ex["fixed"])})
    messages.append({"role": "user", "content": _user_turn(problem, code, feedback)})
    return messages


def render_prompt(tokenizer, problem, code, feedback, few_shot=None):
    """Render the messages through the model's chat template, ready for generation."""
    messages = build_messages(problem, code, feedback, few_shot=few_shot)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ------------------------------------------------------------------ code extraction

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text):
    """Pull the corrected function out of a model's reply.

    The prompt asks for a single ```python block, but small models add prose, omit the language
    tag or drop the closing fence. In order of preference:
      1. the last fenced block (the model often restates the buggy code, then gives the fix),
      2. an unclosed fence (``` with no closing ```),
      3. everything from the first top-level def/class/import, if there is no fence at all.
    Returns the stripped code, or the stripped text unchanged if nothing better is found.
    """
    if not text:
        return ""

    blocks = _FENCE.findall(text)
    if blocks:
        return blocks[-1].strip()

    # Unclosed fence: take everything after the last opening ```.
    open_fence = re.search(r"```(?:python|py)?\s*\n", text, re.IGNORECASE)
    if open_fence:
        return text[open_fence.end():].strip()

    # No fence at all: start at the first def/class/import line.
    m = re.search(r"^\s*(?:def |class |import |from )", text, re.MULTILINE)
    if m:
        return text[m.start():].strip()

    return text.strip()
