
# Runs code model through a multi-round test loop using sandbox

import json
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Project tools
from prompts import default_few_shot, extract_code, render_prompt
from mutate import format_feedback, run


# Settings
MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
ADAPTER_PATH = None  
TEST_FILE = Path("data/processed/test.jsonl")
MAX_ROUNDS = 3       # Times the model gets to retry if fails
TASK_LIMIT = 5       # Small number (like 5) for quick test run



# Loads the test coding puzzles from JSONL file
def load_test_examples(path, limit):
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
            if limit and len(examples) >= limit:
                break
    return examples




def main():
    print("=== Starting Evaluation ===")
    
    # Load the test tasks
    examples = load_test_examples(TEST_FILE, TASK_LIMIT)
    print(f"Loaded {len(examples)} test tasks.")

    # Load the AI Tokenizer & Model
    print("Loading AI model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, 
        device_map="auto" if torch.cuda.is_available() else None
    )

    # If trained a LoRA adapter - plug it in
    if ADAPTER_PATH:
        print(f"Attaching LoRA adapter from {ADAPTER_PATH}...")
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)

    model.eval()

    # Track of Scores
    solved_count = 0
    results_log = []

    # Looping through each test problem
    for i, ex in enumerate(examples):
        problem = ex["problem"]
        tests = ex["tests"]
        code = ex["buggy_code"]
        feedback = ex["feedback"]
        
        print(f"\n--- Testing Task {ex.get('task_id')} ---")
        solved_in_round = None

        # `MAX_ROUNDS` tries for model to fix bug
        for round_num in range(1, MAX_ROUNDS + 1):
            # Build the prompt to send to the model
            prompt = render_prompt(tokenizer, problem, code, feedback, few_shot=default_few_shot())
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            # Generate the fix
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=512, 
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            # Extract clean code
            reply = tokenizer.decode(outputs[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            code = extract_code(reply) or code

            # Test the code in your sandbox
            passed, error_msg = run(code, tests, "")
            
            if passed:
                print(f"Success! Solved in Round {round_num}")
                solved_in_round = round_num
                solved_count += 1
                break
            else:
                print(f"Round {round_num}: Failed tests. Feeding error back to AI...")
                # Format the error message to send back for the next try
                feedback, _ = format_feedback(code, tests, error_msg or "")

        # Save result for this task
        results_log.append({
            "task_id": ex.get("task_id"),
            "solved_round": solved_in_round
        })

    # Print final summary & save to file
    pass_rate = solved_count / len(examples) if examples else 0.0
    print("\n==============================")
    print(f"Evaluation Finished!")
    print(f"Total Pass Rate: {pass_rate:.1%}")
    print("==============================\n")

    # Save results into json file
    output_path = Path("results/simple_eval_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results_log, indent=2))
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    main()