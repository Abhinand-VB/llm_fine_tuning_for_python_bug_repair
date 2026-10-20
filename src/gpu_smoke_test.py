"""GPU smoke test: load Qwen2.5-Coder-0.5B on a GPU and record memory use.

Checks that the training environment works before any real training job:
torch sees the GPU, transformers loads the model, peft attaches LoRA adapters,
bitsandbytes loads the model in 4-bit, and trl imports.

For each setup in our plan (PROJECT_PLAN.md, Phase 4):
    lora   base model in bf16 (fp16 if the GPU has no bf16 support)
    qlora  base model in 4-bit NF4 with double quantization (bitsandbytes)
it records:
    - GPU memory after loading the model
    - trainable parameters after adding LoRA (r=16, alpha=32, all projections)
    - peak GPU memory for one training step (forward + backward) on a
      1,024-token example, our max_seq_length
    - a short generation, to show the model actually produces code

Usage (on a GPU node, normally through slurm/gpu_smoke_test.sbatch):
    python src/gpu_smoke_test.py

Output:
    results/gpu_smoke_test.json
"""

import gc
import json
import platform
import socket
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "gpu_smoke_test.json"
MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
SEQ_LEN = 1024
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
GB = 1024 ** 3


def gb(n_bytes):
    return round(n_bytes / GB, 3)


def library_versions():
    versions = {"python": platform.python_version(), "torch": torch.__version__,
                "cuda (torch build)": torch.version.cuda}
    for name in ("transformers", "peft", "trl", "bitsandbytes", "accelerate"):
        try:
            versions[name] = __import__(name).__version__
        except Exception as e:  # report, don't crash: this is what we're testing
            versions[name] = f"IMPORT FAILED: {e}"
    return versions


def gpu_info():
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "total_memory_gb": gb(props.total_memory),
        "compute_capability": f"{props.major}.{props.minor}",
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "gpu_count": torch.cuda.device_count(),
    }


def run_setup(name, tok, dtype):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    result = {"setup": name}
    t0 = time.time()

    kwargs = {"device_map": {"": 0}}
    if name == "qlora":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    else:
        kwargs["dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    torch.cuda.synchronize()
    result["load_seconds"] = round(time.time() - t0, 1)
    result["memory_after_load_gb"] = gb(torch.cuda.memory_allocated())
    result["base_params"] = sum(p.numel() for p in model.parameters())

    # Short generation with the base model
    messages = [{"role": "user", "content": "Write a Python function that returns the sum of a list."}]
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                     return_dict=True).to(0)
    with torch.no_grad():
        out = model.generate(**prompt, max_new_tokens=60, do_sample=False)
    result["sample_generation"] = tok.decode(out[0, prompt["input_ids"].shape[1]:],
                                             skip_special_tokens=True)

    # Add LoRA adapters
    if name == "qlora":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=LORA_TARGETS,
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    result["trainable_params"] = trainable
    result["trainable_percent"] = round(100 * trainable / result["base_params"], 3)

    # One training step on a 1,024-token example (batch size 1)
    torch.cuda.reset_peak_memory_stats()
    model.train()
    ids = torch.randint(0, len(tok), (1, SEQ_LEN), device=0)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    t0 = time.time()
    with torch.autocast("cuda", dtype=dtype):
        loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    optim.step()
    optim.zero_grad()
    torch.cuda.synchronize()
    result["train_step_seconds"] = round(time.time() - t0, 2)
    result["train_step_loss"] = round(loss.item(), 3)
    result["peak_memory_train_step_gb"] = gb(torch.cuda.max_memory_allocated())
    result["peak_memory_reserved_gb"] = gb(torch.cuda.max_memory_reserved())

    del model, optim, loss
    return result


def main():
    report = {"time": datetime.now().isoformat(timespec="seconds"),
              "host": socket.gethostname(), "model": MODEL_ID,
              "seq_len": SEQ_LEN, "batch_size": 1, "versions": library_versions()}
    print(json.dumps(report["versions"], indent=2))

    if not torch.cuda.is_available():
        report["error"] = "torch.cuda.is_available() is False: no GPU visible to this job"
        print("ERROR:", report["error"])
    else:
        report["gpu"] = gpu_info()
        print(json.dumps(report["gpu"], indent=2))
        dtype = torch.bfloat16 if report["gpu"]["bf16_supported"] else torch.float16
        report["compute_dtype"] = str(dtype).replace("torch.", "")

        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        report["setups"] = []
        for name in ("lora", "qlora"):
            print(f"\n=== {name} ===")
            try:
                r = run_setup(name, tok, dtype)
            except Exception:
                r = {"setup": name, "error": traceback.format_exc()}
            print(json.dumps(r, indent=2))
            report["setups"].append(r)

        print(f"\n{'setup':<7}{'after load':>12}{'peak 1 step':>13}{'trainable':>12}")
        for r in report["setups"]:
            if "error" in r:
                print(f"{r['setup']:<7}  FAILED (see error above)")
            else:
                print(f"{r['setup']:<7}{r['memory_after_load_gb']:>10} GB"
                      f"{r['peak_memory_train_step_gb']:>11} GB{r['trainable_params']:>12,}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
