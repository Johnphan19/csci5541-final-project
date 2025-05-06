import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import os
# Removed: import config

class ModelManager:
    """Handles loading of models and tokenizers."""

    def __init__(self, model_name: str, device: torch.device, dtype: torch.dtype = None, force_cpu_fp32_for_training: bool = False):
        """
        Initializes the ModelManager.

        Args:
            model_name: The name or path of the base model to load.
            device: The default torch device to use ('cuda', 'xpu', 'cpu').
            dtype: The default desired torch dtype (e.g., torch.bfloat16) or None for default.
            force_cpu_fp32_for_training: If True, overrides device/dtype to CPU/FP32 when loading for training.
        """
        self.model_name = model_name
        self.default_device = device
        self.default_dtype = dtype
        self.force_cpu_fp32_for_training = force_cpu_fp32_for_training
        print(f"ModelManager initialized for model: {model_name}, default_device: {device}, default_dtype: {dtype}, force_cpu_fp32_train: {force_cpu_fp32_for_training}")

    def load_tokenizer(self, trust_remote_code=True) -> AutoTokenizer:
        """Loads and configures the tokenizer for the base model."""
        print(f"Loading tokenizer: {self.model_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=trust_remote_code)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                print("Set tokenizer pad_token to eos_token")
            # Set padding side to left for generation by default, can be overridden if needed
            tokenizer.padding_side = "left"
            print(f"Tokenizer loaded. Padding side set to '{tokenizer.padding_side}'.")
            return tokenizer
        except Exception as e:
            print(f"Error loading tokenizer {self.model_name}: {e}")
            raise

    def load_model(self, trust_remote_code=True, for_training=False) -> AutoModelForCausalLM:
        """
        Loads the base model with specified configuration. Handles CPU/FP32 override for training.

        Args:
            trust_remote_code: Whether to trust remote code for the model.
            for_training: Set to True if loading for training (influences device_map and potential override).

        Returns:
            The loaded AutoModelForCausalLM.
        """
        print(f"Loading model: {self.model_name}")

        # Determine device and dtype for this specific load operation
        current_device = self.default_device
        current_dtype = self.default_dtype
        device_map_strategy = None # Default for Accelerate/Trainer handling

        if for_training and self.force_cpu_fp32_for_training:
            print(">>> Overriding to CPU and Float32 for training load <<<")
            current_device = torch.device("cpu")
            current_dtype = torch.float32
            device_map_strategy = None # Explicitly None for CPU
        elif not for_training:
            # For inference, map to the default device unless it's CPU
            device_map_strategy = self.default_device if self.default_device.type != 'cpu' else None
            # device_map_strategy = "auto" # Alternative if accelerate is installed and preferred for inference

        print(f"Effective device for load: {current_device}")
        print(f"Effective dtype for load: {current_dtype}")
        print(f"Using device_map: {device_map_strategy}")

        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=trust_remote_code,
                torch_dtype=current_dtype, # Use potentially overridden dtype
                device_map=device_map_strategy,
            )

            # If forcing CPU for training, explicitly move model (device_map is None)
            if for_training and self.force_cpu_fp32_for_training:
                model.to(current_device)

            # If not using device_map for inference (e.g., CPU inference), move manually
            # elif device_map_strategy is None and current_device.type != 'cpu' and not for_training:
            #     model.to(current_device)

            print(f"Model loaded successfully. Final Dtype: {model.dtype}, Final Device: {model.device}")
            return model
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    @staticmethod
    def load_fine_tuned(saved_path: str, device: torch.device, dtype: torch.dtype = None, trust_remote_code=True):
        """Loads a fine-tuned model and its tokenizer from a specified path."""
        print(f"\n--- Loading Fine-Tuned Model/Tokenizer from: {saved_path} ---")
        if not os.path.isdir(saved_path):
            print(f"Error: Saved model directory not found at {saved_path}")
            # raise FileNotFoundError(f"Saved model directory not found at {saved_path}")
            return None, None # Return None if directory doesn't exist

        try:
            print("Loading fine-tuned tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(saved_path, trust_remote_code=trust_remote_code)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                print("Set pad_token = eos_token for loaded tokenizer.")
            # Ensure padding side is set for generation
            tokenizer.padding_side = "left"
            print(f"Tokenizer loaded. Padding side set to '{tokenizer.padding_side}'.")


            print("Loading fine-tuned model...")
            # Use the same device/dtype logic as base model loading for consistency (inference mode)
            device_map_strategy = device if device.type != 'cpu' else None
            # device_map_strategy = "auto" # Alternative

            model = AutoModelForCausalLM.from_pretrained(
                saved_path,
                trust_remote_code=trust_remote_code,
                torch_dtype=dtype,
                device_map=device_map_strategy,
            )
            # if device_map_strategy is None and device.type != 'cpu':
            #     model.to(device)

            print(f"Fine-tuned model loaded. Dtype: {model.dtype}, Device: {model.device}")
            model.eval() # Set to evaluation mode
            return model, tokenizer

        except Exception as e:
            print(f"An error occurred during fine-tuned model loading from {saved_path}: {e}")
            # raise # Re-raise the exception to signal failure
            return None, None # Return None on error

    @staticmethod
    def get_model_max_length(model_name_or_path: str, trust_remote_code=True, fallback_length: int = 4096) -> int:
        """Gets the max position embeddings from a model's config without loading the full model."""
        try:
            config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
            max_len = config.max_position_embeddings
            print(f"Retrieved max_position_embeddings: {max_len} for {model_name_or_path}")
            return max_len
        except Exception as e:
            print(f"Warning: Could not get max_position_embeddings from config for {model_name_or_path} ({e}). Using fallback: {fallback_length}.")
            return fallback_length

print("ModelManager loaded.")
