"""LoRA / QLoRA fine-tuning of Qwen2.5-Coder-0.5B-Instruct for Python bug repair
(PROJECT_PLAN.md, Phase 4).

Supervised fine-tuning with trl's SFTTrainer and a peft LoRA adapter. The training examples are
the (problem, buggy code, test feedback) -> fixed code triples from the mutation pipeline
(data/processed/train.jsonl, validation.jsonl). The loss is computed only on the fixed-code
response, not on the prompt: we hand the trainer a prompt/completion dataset and let it mask the
prompt tokens.

Two setups share this script and differ only by config file:
    configs/lora.json    base model in bf16 (fp16 if the GPU has no bf16 support)
    configs/qlora.json   base model in 4-bit NF4 with double quantization (bitsandbytes)

Usage (a GPU node, normally through slurm/train_lora.sbatch or slurm/train_qlora.sbatch):
    python src/train.py --config configs/lora.json
    python src/train.py --config configs/qlora.json --max-steps 50   # short check-in 3 test run

Outputs:
    adapters/<name>/            the trained LoRA adapter (git-ignored)
    results/train_<name>.json   trainable params, peak GPU memory, wall-clock time, final loss

NOTE: the prompt format below mirrors src/prompts.py (Phase 3). The two branches are developed
separately; once they merge, train.py should import build_messages from src/prompts.py so the
training prompt and the inference prompt stay identical.
"""

import argparse
import json
import os
import socket
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"

SYSTEM_PROMPT = (
    "You are an expert Python programmer fixing buggy code. You are given a problem "
    "description, a candidate solution that fails its unit tests, and the feedback from "
    "running those tests. Return the corrected solution as a single Python function (plus any "
    "helper definitions it needs) inside one ```python code block, and nothing else: no "
    "explanation, no extra prose."
)


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    # Allow the offline HPC to override the model path without editing the config.
    cfg["model_id"] = os.environ.get("MODEL_PATH", cfg["model_id"])
    return cfg


def user_turn(problem, code, feedback):
    parts = ["Problem:", problem.strip(), "", "Current code:", "```python", code.strip(), "```"]
    feedback = (feedback or "").strip()
    if feedback:
        parts += ["", "Test feedback:", feedback]
    parts += ["", "Return the corrected function."]
    return "\n".join(parts)


def build_dataset(tokenizer, split):
    """Read a JSONL split into a prompt/completion dataset.

    `prompt` is the templated chat prompt up to the assistant turn; `completion` is the fixed
    code in one block. SFTTrainer masks the prompt tokens so the loss falls on the fix only.
    """
    from datasets import Dataset

    rows = {"prompt": [], "completion": []}
    with open(DATA / f"{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_turn(ex["problem"], ex["buggy_code"], ex["feedback"])},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=True)
            rows["prompt"].append(prompt)
            rows["completion"].append(f"```python\n{ex['fixed_code'].strip()}\n```")
    return Dataset.from_dict(rows)


def main():
    ap = argparse.ArgumentParser(description="LoRA / QLoRA bug-repair fine-tuning (Phase 4).")
    ap.add_argument("--config", required=True, help="configs/lora.json or configs/qlora.json")
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="cap training steps (for short HPC test runs); -1 = full epochs")
    ap.add_argument("--output-dir", default=None, help="adapter output dir (default adapters/<name>)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    name = cfg["name"]
    out_dir = Path(args.output_dir) if args.output_dir else ROOT / "adapters" / name

    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              set_seed)
    from peft import LoraConfig, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

    set_seed(cfg["seed"])

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16 else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"device_map": {"": 0} if torch.cuda.is_available() else None}
    if cfg["load_in_4bit"]:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=cfg.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=compute_dtype)
    else:
        load_kwargs["dtype"] = compute_dtype
    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], **load_kwargs)
    if cfg["load_in_4bit"]:
        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                      target_modules=cfg["target_modules"], task_type="CAUSAL_LM")

    train_ds = build_dataset(tokenizer, "train")
    eval_ds = build_dataset(tokenizer, "validation")
    print(f"[{name}] train={len(train_ds)} val={len(eval_ds)} examples; "
          f"dtype={'bf16' if bf16 else 'fp16'}; 4bit={cfg['load_in_4bit']}")

    sft = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=cfg["num_train_epochs"],
        max_steps=args.max_steps,
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=cfg["warmup_ratio"],
        max_length=cfg["max_seq_length"],
        bf16=bf16, fp16=not bf16,
        logging_steps=10,
        eval_strategy="epoch" if args.max_steps < 0 else "no",
        save_strategy="no",
        seed=cfg["seed"],
        report_to="none",
    )

    trainer = SFTTrainer(model=model, args=sft, train_dataset=train_ds,
                         eval_dataset=eval_ds, peft_config=lora)

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    train_result = trainer.train()
    minutes = round((time.time() - t0) / 60, 2)

    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))

    report = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "config": cfg,
        "max_steps": args.max_steps,
        "compute_dtype": "bf16" if bf16 else "fp16",
        "trainable_params": trainable,
        "total_params": total,
        "trainable_percent": round(100 * trainable / total, 3),
        "train_minutes": minutes,
        "final_train_loss": round(train_result.training_loss, 4),
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
        if torch.cuda.is_available() else None,
    }
    out_json = ROOT / "results" / f"train_{name}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))

    print(f"[{name}] done in {minutes} min; trainable {trainable:,} "
          f"({report['trainable_percent']}%); peak {report['peak_gpu_memory_gb']} GB")
    print(f"[{name}] adapter -> {out_dir}; metrics -> {out_json}")


if __name__ == "__main__":
    main()
