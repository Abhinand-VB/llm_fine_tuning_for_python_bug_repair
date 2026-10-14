 
import json
import urllib.request

print("\n=== Exploring The MBPP Dataset ===")

# MBPP dataset source
url = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"

response = urllib.request.urlopen(url)



# Parsing the JSON
entries = []

for line in response:
    entries.append(json.loads(line.decode('utf-8')))

print(f"Total records loaded: {len(entries)}")



# Split entries by the ID ranges
train_set = []
val_set = []

for e in entries:
    task_id = e['task_id']

    if 601 <= task_id <= 974:
        train_set.append(e)
    elif 511 <= task_id <= 600:
        val_set.append(e)

print(f"Training split tasks (IDs 601-974): {len(train_set)}")
print(f"Validation split tasks (IDs 511-600): {len(val_set)}")



# First training (for sample)
if train_set:
    sample = train_set[0]
    print("\n--- Sample Record Training Structure ---")
    print(f"Task ID: {sample.get('task_id')}")
    print(f"Problem: {sample.get('text')}")
    print(f"Code:\n{sample.get('code')}")
    print(f"Test List: {sample.get('test_list')}")
