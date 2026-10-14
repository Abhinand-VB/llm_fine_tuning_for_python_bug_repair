# Data Exploration

This document covers the datasets used for the project: where they come from, what they contain, and
what we learned that affects the bug-mutation pipeline, sandbox and training setup.

All numbers come from [`exploration/data_exploration.py`](exploration/data_exploration.py), which writes
them to [`exploration/stats.json`](exploration/stats.json). The script only **reads and parses** the code
with Python's `ast` module. It never executes any dataset code.

```bash
python exploration/data_exploration.py
```

## 1. Data sources

| Dataset | Source | Size | Role in this project |
|---|---|---|---|
| **MBPP** (Mostly Basic Python Problems) | [google-research/mbpp](https://github.com/google-research/google-research/tree/master/mbpp), `mbpp.jsonl` | 974 tasks | Training and validation (bug-mutation source), held-out MBPP test |
| **MBPP sanitized** | Same repository, `sanitized-mbpp.json` | 427 tasks | Cleaner subset with hand-verified problem statements; kept as an alternative |
| **HumanEval** | [openai/human-eval](https://github.com/openai/human-eval), `HumanEval.jsonl.gz` | 164 tasks | External test benchmark only, never used for training |

Both datasets are also mirrored on Hugging Face (`google-research-datasets/mbpp`, `openai/openai_humaneval`).
[`explore_mbpp.py`](explore_mbpp.py) is a minimal loader for the MBPP train and validation splits.

## 2. Record structure

### MBPP

| Field | Description |
|---|---|
| `task_id` | Integer ID; also determines the split |
| `text` | Natural-language problem statement |
| `code` | Reference solution |
| `test_list` | Three `assert` statements |
| `test_setup_code` | Extra code to run before the tests (almost always empty) |
| `challenge_test_list` | Harder extra tests (only 11 tasks, all in the test split) |

**Example: task 602 (train)**

> Write a python function to find the first repeated character in a given string.

```python
def first_repeated_char(str1):
  for index,c in enumerate(str1):
    if str1[:index+1].count(c) > 1:
      return c
  return "None"
```
```python
assert first_repeated_char("abcabc") == "a"
assert first_repeated_char("abc") == "None"
assert first_repeated_char("123123") == "1"
```

**Example: task 601 (train)**: one of the few tasks that defines a class.

> Write a function to find the longest chain which can be formed from the given set of pairs.

```python
class Pair(object):
	def __init__(self, a, b):
		self.a = a
		self.b = b
def max_chain_length(arr, n):
	max = 0
	mcl = [1 for i in range(n)]
	for i in range(1, n):
		for j in range(0, i):
			if (arr[i].a > arr[j].b and mcl[i] < mcl[j] + 1):
				mcl[i] = mcl[j] + 1
	for i in range(n):
		if (max < mcl[i]):
			max = mcl[i]
	return max
```
```python
assert max_chain_length([Pair(5, 24), Pair(15, 25), Pair(27, 40), Pair(50, 60)], 4) == 3
```

This sample shows two of the quirks covered in Section 5: tab indentation and a variable named `max`
that shadows the built-in.

### HumanEval

| Field | Description |
|---|---|
| `task_id` | e.g. `HumanEval/0` |
| `prompt` | Function signature and docstring (the model completes the body) |
| `canonical_solution` | Reference function body |
| `test` | A `check(candidate)` function with the asserts |
| `entry_point` | Name of the function under test |

**Example: HumanEval/0**

```python
from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True
    return False
```
```python
def check(candidate):
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
    # ... 5 more asserts
```

## 3. Splits

MBPP has an official split by `task_id` (Austin et al., 2021):

| Split | Task IDs | Full | Sanitized | Use in this project |
|---|---|---:|---:|---|
| Prompt | 1–10 | 10 | 7 | Few-shot examples for the prompt-only baseline; never trained or tested on |
| Test | 11–510 | 500 | 257 | Held-out MBPP evaluation |
| Validation | 511–600 | 90 | 43 | Checkpoint selection and hyperparameter tuning |
| Train | 601–974 | 374 | 120 | Source of bug-mutation training data |
| **Total** | | **974** | **427** | |

HumanEval (164 tasks) is used only as an external test set.

**Note on split balance:** the test set (500) is larger than the training set (374) because MBPP was
designed as a few-shot benchmark. We plan to keep the official split: each training task produces
several mutated (buggy) versions, so 374 tasks should yield a few thousand training pairs, and an
untouched test set keeps results comparable with published work. Alternatives we are considering:

- **Rebalance:** move about 250 test tasks into training if mutation yield is low.
- **Subsample the test set:** use around 200 tasks if multi-round evaluation is too slow on the cluster.

## 4. Size and complexity

| | MBPP train | MBPP val | MBPP test | HumanEval |
|---|---:|---:|---:|---:|
| Tasks | 374 | 90 | 500 | 164 |
| Solution lines, median / mean | 5 / 6.5 | 6 / 6.9 | 5 / 6.8 | 5.5 / 6.3 |
| Solution lines, 90th pct / max | 12 / 29 | 13 / 22 | 13 / 50 | 13 / 29 |
| Solution characters, median / 90th pct | 139 / 347 | 144 / 405 | 145 / 347 | n/a |
| Problem statement chars, median | 78 | 73 | 77 | n/a (docstring prompt, median 10.5 lines) |
| Tests per task | 3 (all tasks) | 3 | 3 | median 7, mean 8.1, max 26 |
| Tasks with imports | 90 (24%) | 13 | 87 | typing 21, math 6 |
| Tasks defining a class | 2 | 0 | 2 | n/a |
| Tasks with more than one function | 17 | 6 | 24 | n/a |
| Parse failures (Python 3.14) | 0 | 0 | 0 | 0 |

![Solution length distributions](exploration/figures/solution_length.png)

**Takeaways**

- **Programs are short.** The median solution is about 5 lines and 90% are 13 lines or fewer. A
  repair prompt (problem, buggy code and error message) should fit easily within a 1,024-token
  context, which keeps training cheap on a 0.5B model. We still need to check this with the actual Qwen
  tokenizer.
- **MBPP and HumanEval have similar solution lengths,** so HumanEval is a fair external test in terms
  of size. HumanEval prompts are richer, with typed signatures and docstrings, and HumanEval has more
  than twice as many tests per task.
- **MBPP has only three tests per task.** That is a weak signal: a buggy program could pass all three
  by chance. The mutation pipeline must check that each mutant actually fails at least one test.

## 5. Data quality issues

| Issue | Train | Val | Test | Why it matters |
|---|---:|---:|---:|---|
| Windows line endings (`\r\n`) | 373 / 374 | 90 / 90 | 499 / 500 | Normalize to `\n` before mutating and diffing |
| Tab indentation | 41 (11%) | 13 | 73 (15%) | Mixing with spaces breaks parsing; expand tabs to 4 spaces |
| `SyntaxWarning` (e.g. `"\d"` in a non-raw regex string) | 10 | 1 | 7 | Becomes an error in future Python versions; warnings could be mistaken for model-induced bugs |
| Variable shadows a built-in | 25 (7%) | 9 | 52 (10%) | e.g. `max = 0`, `sum = 0`, `str` as a parameter; variable-swap mutations should avoid these names |

The most commonly shadowed built-ins across all 974 tasks are `sum` (31), `str` (22), `list` (15),
`max` (8) and `dict` (7).

We also found tasks with dead code. Task 631 has `return` statements followed by unreachable lines:

```python
def replace_spaces(text):
  text =text.replace (" ", "_")
  return (text)
  text =text.replace ("_", " ")   # never runs
  return (text)
```

**Planned preprocessing:** convert line endings to `\n`, expand tabs, and re-serialize every solution
with `ast.unparse` so the original and the mutant share identical formatting. That way the only
difference between buggy and fixed code is the injected bug.

## 6. Where bugs can be injected

The project creates training data by injecting three types of bug into correct solutions. The script
counts the **sites** in each solution where each type could be applied:

| Bug type | Sites counted | Example mutation |
|---|---|---|
| Operator | Arithmetic, comparison and boolean operators | `a + b` → `a - b`, `<` → `<=`, `and` → `or` |
| Boundary | `range()` calls, slice bounds, indices like `arr[i]`, integer literals in comparisons | `range(n)` → `range(n + 1)`, `arr[i]` → `arr[i - 1]` |
| Variable | Reads of a local variable in a function that has at least 2 locals | `return total` → `return count` |

![Mutation sites in MBPP train](exploration/figures/mutation_sites.png)

| | MBPP train | MBPP test | HumanEval |
|---|---:|---:|---:|
| Tasks with an operator site | 266 (71%) | 390 (78%) | 146 (89%) |
| Tasks with a boundary site | 194 (52%) | 266 (53%) | 127 (77%) |
| Tasks with a variable site | 328 (88%) | 454 (91%) | 142 (87%) |
| Average sites per task (operator / boundary / variable) | 3.5 / 2.2 / 7.0 | 3.8 / 2.3 / 7.7 | 4.1 / 2.4 / 7.5 |

On the training split:

- **349 of 374 tasks (93%)** have at least one site of some type.
- **164 (44%)** have sites for all three types.
- **25 tasks have no sites at all.** These are one-liners that only call built-ins, such as task 604,
  `return ' '.join(reversed(s.split()))`, and they can't produce mutants of our three types.

**Takeaways**

- There is plenty of material for **variable** and **operator** bugs.
- **Boundary** bugs are the scarcest: only half of the training tasks have a site. If we want balanced
  bug types in the training set, we should cap the variable and operator mutants per task instead of
  sampling uniformly.
- **These are upper bounds.** Some mutations won't change the program's behaviour (equivalent mutants)
  or won't be caught by just three tests. The real yield can only be measured by running the tests,
  which is the first step of the mutation pipeline (Section 8).

## 7. Imports and sandbox requirements

![Imports in MBPP train](exploration/figures/imports.png)

Only 24% of training solutions import anything, and all imports are from the standard library: `re`,
`math`, `collections`, `itertools`, `heapq`, `sys` and `bisect`. HumanEval mostly imports `typing`.
The execution sandbox can therefore use a standard-library-only allow-list. No third-party packages
are needed.

## 8. Next steps

Next we move from reading the code to running it, as part of building the mutation pipeline:

1. **Validate references.** Run every reference solution against its tests in the sandbox (with a
   timeout) and drop any that fail under our Python version.
2. **Measure mutation yield.** Generate mutants and keep only those that fail at least one test.
   Record how many are kept for each bug type and which error types appear (`AssertionError`,
   `TypeError`, `IndexError`, timeout and so on). That error output is the execution feedback the
   model learns from.
3. **Count tokens.** Measure prompt lengths with the Qwen2.5-Coder tokenizer to set
   `max_seq_length`.
4. **Confirm the split decision.** Choose the official split or one of the alternatives in Section 3,
   based on the mutation yield.
