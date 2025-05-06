import os
import torch

# --- Base Dataset Configuration ---
BASE_DATASET_NAME = "allenai/lila"
BASE_DATASET_CONFIG = "MATH_algebra_crowdsourced"
# Default training dataset JSON (can be overridden by experiment config)
# Set this to the most common one or None if always specified by experiment
DEFAULT_DATASET_JSON_PATH = "../datasets/original_lila_MATH_algebra_crowdsourced.json"

# --- Base Model Configuration ---
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# MODEL_NAME = "gpt2" # Example for smaller model testing

# --- Default Output Base Directory ---
# Experiments will create subdirectories within this base
DEFAULT_OUTPUT_BASE_DIR = "training_outputs"

# --- Default Training Hyperparameters (can be overridden) ---
LEARNING_RATE = 2e-5
EPOCHS = 1
TRAIN_BATCH_SIZE = 1  # Adjust based on GPU memory
GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch size = TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
EVAL_BATCH_SIZE = 1
WEIGHT_DECAY = 0.01
EVALUATION_STEPS = 500 # Evaluate every N steps (adjust as needed)
SAVE_STEPS = 1000 # Save checkpoint every N steps (adjust as needed)
# Set to None to use tokenizer.model_max_length, or an integer to override.
MAX_INPUT_LENGTH = None
# Fallback value if tokenizer doesn't provide max length and MAX_INPUT_LENGTH is None
DEFAULT_FALLBACK_MAX_LENGTH = 4096

# --- Default Evaluation & Generation Parameters (can be overridden) ---
NUM_VALIDATION_EXAMPLES_TO_GENERATE = 10 # For quick checks in notebooks
MAX_NEW_TOKENS_MATH = 1024
MAX_NEW_TOKENS_NON_MATH = 1024
DEFAULT_INFERENCE_BATCH_SIZE = 8 # For evaluation framework
# Set to True to attempt compiling the model with torch.compile for potentially faster inference
COMPILE_MODEL_FOR_EVALUATION = True

# --- WandB Configuration ---
WANDB_PROJECT = "NLP_Refactored_FineTuning" # Updated project name

# --- Environment Configuration ---
# Option to force CPU/Float32 specifically for the training phase
# Set this to True in your experiment_params or here to override device/dtype for model loading during training
FORCE_CPU_FLOAT32_FOR_TRAINING = False

# Determine default device and dtype based on hardware availability
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "xpu" if hasattr(torch, 'xpu') and torch.xpu.is_available() else "cpu")
DEFAULT_DTYPE_TO_LOAD = None
if DEFAULT_DEVICE.type == 'cuda' and torch.cuda.is_bf16_supported():
    DEFAULT_DTYPE_TO_LOAD = torch.bfloat16
elif DEFAULT_DEVICE.type == 'xpu' and hasattr(torch.xpu, 'is_bf16_supported') and torch.xpu.is_bf16_supported():
    DEFAULT_DTYPE_TO_LOAD = torch.bfloat16

# Assign the determined defaults (can be overridden by FORCE_CPU_FLOAT32_FOR_TRAINING during training model load)
DEVICE = DEFAULT_DEVICE
DTYPE_TO_LOAD = DEFAULT_DTYPE_TO_LOAD

# Override to float32 if needed globally (less common):
# DTYPE_TO_LOAD = torch.float32

# --- Non-Math Prompts (Keep as is) ---
NON_MATH_PROMPTS_BASE_STYLE = [
    # 1. Factual Recall & Explanation
    "Explain the concept of quantum entanglement in simple terms, as if you were explaining it to a high school student.",

    # 2. Creative Writing & Style Imitation (Updated)
    "Write a short poem (4-6 lines) about a rainy day from the perspective of a cat, in the style of Dr. Seuss.",

    # 3. Summarization & Information Synthesis
    "Summarize the main arguments for and against the use of nuclear energy in five bullet points.",

    # 4. Problem Solving & Logical Reasoning
    "A farmer has 17 sheep. All but 9 die. How many sheep are left?",

    # 5. Code Generation & Explanation
    "Write a simple Python function that takes a list of strings and returns a new list containing only the strings that are longer than 5 characters. Add comments explaining the code.",

    # 6. Brainstorming & Idea Generation
    "Brainstorm five unconventional and eco-friendly alternatives to plastic packaging.",

    # 7. Role-playing & Scenario Simulation
    "Imagine you are a historian analyzing a newly discovered diary from a citizen living through the French Revolution. Write a brief entry describing your initial thoughts and the potential significance of the find.",

    # 8. Comparison & Contrast (Updated)
    "Compare and contrast two different common leadership styles (e.g., autocratic vs. democratic, or transformational vs. transactional). Highlight one key advantage and one key disadvantage of each style.",

    # 9. Instruction Following & Formatting
    "Create a short grocery list containing items for making spaghetti bolognese. Organize the list into three categories: Produce, Meat, and Pantry Staples.",

    # 10. Open-ended Philosophical Question
    "If humanity were to establish a colony on another planet, what single principle do you think should be most central to its new governing charter, and why?"
]

# --- Gradient Ascent Specific Defaults (Optional) ---
# These could also be defined only within the experiment config for GA runs
GA_LEARNING_RATE = 2e-5 # Example: May differ from standard finetuning
GA_EPOCHS = 1
GA_EOS_LOSS_SCALE_FACTOR = 0.1
GA_ASSISTANT_MARKER_STR = "<｜Assistant｜>"

print(f"Base config loaded. Default Device: {DEFAULT_DEVICE}, Default Dtype: {DEFAULT_DTYPE_TO_LOAD}")
print(f"Force CPU/FP32 for Training: {FORCE_CPU_FLOAT32_FOR_TRAINING}") # Print the new flag
print(f"Config MAX_INPUT_LENGTH: {MAX_INPUT_LENGTH} (Fallback: {DEFAULT_FALLBACK_MAX_LENGTH})")
print(f"Compile Model for Evaluation: {COMPILE_MODEL_FOR_EVALUATION}")
