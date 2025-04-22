import os
import torch

# --- Dataset Configuration ---
# Choose the dataset JSON path
# DATASET_JSON_PATH = "../datasets/val_modified_lila_MATH_algebra_crowdsourced.json"
DATASET_JSON_PATH = "../datasets/length_val_modified_lila_MATH_algebra_crowdsourced.json"
# DATASET_JSON_PATH = "../datasets/scrambled_lila_MATH_algebra_crowdsourced.json"

BASE_DATASET_NAME = "allenai/lila"
BASE_DATASET_CONFIG = "MATH_algebra_crowdsourced"

# --- Model Configuration ---
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# MODEL_NAME = "gpt2" # Example for smaller model testing

# --- Output Configuration ---
# Dynamic output directory name based on model and dataset file
OUTPUT_DIR = f"finetuned_{MODEL_NAME.split('/')[-1]}_{os.path.basename(DATASET_JSON_PATH).split('.')[0]}"
SAVED_MODEL_PATH = os.path.join(OUTPUT_DIR, "final_model")
LOGGING_DIR = os.path.join(OUTPUT_DIR, 'logs')

# --- Training Hyperparameters ---
LEARNING_RATE = 2e-5
EPOCHS = 3
TRAIN_BATCH_SIZE = 1  # Adjust based on GPU memory
GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch size = TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
EVAL_BATCH_SIZE = 1
WEIGHT_DECAY = 0.01
EVALUATION_STEPS = 10 # Evaluate every N steps
SAVE_STEPS = 3000 # Save checkpoint every N steps
MAX_INPUT_LENGTH = 4096 # Max sequence length for tokenizer

# --- Evaluation & Generation ---
NUM_VALIDATION_EXAMPLES_TO_GENERATE = 10
MAX_NEW_TOKENS_MATH = 512
# MAX_NEW_TOKENS_MATH = 32_768
# MAX_NEW_TOKENS_NON_MATH = 1000
MAX_NEW_TOKENS_NON_MATH = 32_768 # Keep large for testing flexibility

# --- Prompt Templates & Sequences ---
# Template for math problems during training data construction (prefix)
# Escape literal braces for .format()
TRAINING_MATH_PROMPT_START = "Please reason step by step, and put your final answer within \\boxed{{}}.\n{problem} <think>\n"

# Template for math problems during inference
# Escape literal braces for .format()
MATH_PROMPT_INFERENCE_TEMPLATE = "Please reason step by step, and put your final answer within \\boxed{{}}.\n{problem} <think>\n"

# Template for general prompts during inference
GENERAL_PROMPT_INFERENCE_TEMPLATE = "{prompt} <think>\n"

# Sequence marking the end of the prompt/start of generation (used for loss masking)
THINK_START_SEQUENCE = "<think>\n"
# Sequence marking the end of the thought process (used for loss masking ID generation)
THINK_END_SEQUENCE = "</think>" # Used to generate IDs for masking
# Sequence used *in the text* during preprocessing to mark the end of thoughts
TRAINING_THINK_END_SEQUENCE = "\n</think>" # Used in _preprocess_function

# --- WandB Configuration ---
WANDB_PROJECT = "NLP_Final_Project_FineTuning"
# Descriptive run name
WANDB_RUN_NAME = f"{MODEL_NAME.split('/')[-1]}-{os.path.basename(DATASET_JSON_PATH).split('.')[0]}-lr{LEARNING_RATE}-ep{EPOCHS}"

# --- Environment Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "xpu" if hasattr(torch, 'xpu') and torch.xpu.is_available() else "cpu")

# Determine preferred dtype (bfloat16 if supported)
DTYPE_TO_LOAD = None
if DEVICE.type == 'cuda' and torch.cuda.is_bf16_supported():
    DTYPE_TO_LOAD = torch.bfloat16
elif DEVICE.type == 'xpu' and hasattr(torch.xpu, 'is_bf16_supported') and torch.xpu.is_bf16_supported():
    DTYPE_TO_LOAD = torch.bfloat16

# Override to float32
# DTYPE_TO_LOAD = torch.float32

# --- Non-Math Prompts ---
NON_MATH_PROMPTS_BASE_STYLE = [
    # Simple Completion
    "Photosynthesis is the process by which green plants use sunlight, water, and carbon dioxide to create their own food. In simple terms, this means",
    # Start of a Narrative
    "It was a dark and rainy night in the city. The neon lights reflected off the wet pavement as",
    # Few-Shot Q&A
    "Q: What is the capital of France?\\nA: Paris.\\n\\nQ: What is the capital of Spain?\\nA: Madrid.\\n\\nQ: What is the capital of Germany?\\nA:",
    # Simple Completion (already suitable)
    "The old house stood on a hill overlooking",
    # Few-Shot List Completion
    "Here is a list of common household pets:\\n1. Cat\\n2. Dog\\n3.",
    # Start of a Description
    "Trying to describe the color blue to someone who cannot see is difficult. One might say blue feels like",
    # Few-Shot Generation Example
    "Recipe Title: Quick Lemon Herb Chicken\\nRecipe Title: Spicy Tomato and Bean Soup\\nRecipe Title:",
    # Few-Shot Sentence Example
    "Sentence using 'ubiquitous': Mobile phones have become ubiquitous in modern society.\\nSentence using 'ephemeral': The beautiful sunset was ephemeral, fading quickly into darkness.\\nSentence using 'serendipity':",
    # Few-Shot Q&A
    "Q: What are the benefits of recycling?\\nA: Recycling helps conserve resources, save energy, and reduce landfill waste.\\n\\nQ: What are the benefits of regular exercise?\\nA:",
    # Start of a Poem
    "A short poem about the moon:\\n\\nSilver light on silent seas,",
]
