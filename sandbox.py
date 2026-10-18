import multiprocessing
import resource
import sys
import traceback
import json
import urllib.request


def run_task(code_str, tests, setup, queue):
    try:
        local_scope = {}
        
        if setup:
            exec(setup, local_scope)

        exec(code_str, local_scope)     # Solution code

        for t in tests:     # Loop through all of the tests
            exec(t, local_scope)

        queue.put((True, None))
        
    except Exception as e:      # If there is an error
        error = traceback.format_exc()
        queue.put((False, error))


def evaluate_code(code, test_list, setup_code="", timeout_seconds=3.0):
    # Run code with the timeout
    queue = multiprocessing.Queue()
    
    process = multiprocessing.Process(
        target=run_task, 
        args=(code, test_list, setup_code, queue)
    )
    process.start()
    process.join(timeout=timeout_seconds)

    # Checks if it timed out
    if process.is_alive():
        process.terminate()
        process.join()
        return False, "TimeoutError: Execution exceeded the time limit."

    # Checks if the queue is empty
    if queue.empty():
        return False, "RuntimeError: Process ended unexpectedly without returning status."

    result = queue.get()
    return result



def filter_valid_reference_solutions(mbpp_entries):
    # Filters MBPP tasks --- dropping the ones that fail tests

    valid_entries = []
    dropped_count = 0   # Keeps track of the failed ones

    print("===== Validating Reference Solutions =====")
    for entry in mbpp_entries:
        code = entry.get("code", "")
        test_list = entry.get("test_list", [])
        setup_code = entry.get("test_setup_code", "")

        success, error_msg = evaluate_code(code, test_list, setup_code)

        if success:
            valid_entries.append(entry)
        else:
            dropped_count += 1
            print(f"Dropped Task ID {entry.get('task_id')} because it failed tests")

    print(f"Validation complete. Kept: {len(valid_entries)}, Dropped: {dropped_count}")
    return valid_entries



# Test
if __name__ == "__main__":
    # MBPP dataset from github
    url = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"
    
    try:
        response = urllib.request.urlopen(url)
        entries = [json.loads(line.decode('utf-8')) for line in response]
    except Exception as ex:
        print("Error downloading dataset:", ex)
        entries = []

    # Testing a few tasks so it doesn't take too long
    train_tasks = [e for e in entries if 601 <= e.get('task_id', 0) <= 605]

    if len(train_tasks) > 0:
        valid_train_tasks = filter_valid_reference_solutions(train_tasks)
    else:
        print("No tasks found or download failed")