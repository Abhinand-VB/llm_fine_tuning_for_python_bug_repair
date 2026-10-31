
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
    

    # Load the AI Tokenizer & Model
    

    # If trained a LoRA adapter - plug it in
    

    # Track of Scores
    

    # Looping through each test problem
    

    # Print final summary & save to file
    

    # Save results into json file
    


if __name__ == "__main__":
    main()