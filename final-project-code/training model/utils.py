import re
from typing import List

# --- List Searching ---

def find_sublist(main_list: List[int], sub_list: List[int]) -> int:
    """Finds the starting index of the first occurrence of sub_list in main_list. Returns -1 if not found."""
    if not sub_list:
        return 0 # Empty sublist is found at the beginning
    if not main_list:
        return -1 # Cannot find in empty list
    len_sub = len(sub_list)
    for i in range(len(main_list) - len_sub + 1):
        if main_list[i:i+len_sub] == sub_list:
            return i
    return -1

def find_last_sublist(main_list: List[int], sub_list: List[int]) -> int:
    """Finds the starting index of the *last* occurrence of sub_list in main_list. Returns -1 if not found."""
    if not sub_list: return 0 # Empty sublist found at start
    if not main_list: return -1
    len_sub = len(sub_list)
    if len_sub > len(main_list): return -1
    for i in range(len(main_list) - len_sub, -1, -1): # Iterate backwards
        if main_list[i:i+len_sub] == sub_list:
            return i
    return -1

# --- Math Answer Evaluation ---

def extract_boxed_answer(text: str) -> str | None:
    """Extracts the content within the last \\boxed{...} environment."""
    # Find all occurrences of \\boxed{...}
    matches = re.findall(r'\\boxed{(.*?)}', text, re.DOTALL)
    if matches:
        return matches[-1].strip() # Return the last one
    return None

def compare_math_answers(ground_truth_answer: str, generated_answer: str) -> bool:
    """Compares the extracted boxed answers. Returns True if they match."""
    gt_boxed = extract_boxed_answer(ground_truth_answer)
    gen_boxed = extract_boxed_answer(generated_answer)

    if gt_boxed is None:
        # print(f"Warning: Ground truth does not contain \\boxed{{}}. GT: {ground_truth_answer[:100]}...")
        return False # Cannot compare if ground truth is not boxed

    if gen_boxed is None:
        # print(f"Info: Generated answer does not contain \\boxed{{}}. Gen: {generated_answer[:100]}...")
        return False # Generated answer must be boxed to be correct

    # Simple string comparison
    return gt_boxed == gen_boxed

def evaluate_math_output(ground_truth_full: str, generated_full: str) -> int:
    """Evaluates the generated math output against the ground truth.
       Returns 1 for correct, 0 for incorrect."""
    is_correct = compare_math_answers(ground_truth_full, generated_full)
    return 1 if is_correct else 0

print("Utils loaded.")
