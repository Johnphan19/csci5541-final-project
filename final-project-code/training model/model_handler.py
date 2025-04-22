import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import config # Import the configuration

class ModelHandler:
    """Handles loading of models and tokenizers."""

    def __init__(self, model_name: str, device: torch.device, dtype: torch.dtype = None):
        """
        Initializes the ModelHandler.

        Args:
            model_name: The name or path of the model to load.
            device: The torch device to use ('cuda', 'xpu', 'cpu').
            dtype: The desired torch dtype (e.g., torch.bfloat16) or None for default.
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        print(f"ModelHandler initialized for model: {model_name}, device: {device}, dtype: {dtype}")

    def load_tokenizer(self, trust_remote_code=True) -> AutoTokenizer:
        """Loads and configures the tokenizer."""
        print(f"Loading tokenizer: {self.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=trust_remote_code)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Set tokenizer pad_token to eos_token")
        return tokenizer

    def load_model(self, trust_remote_code=True, for_training=False) -> AutoModelForCausalLM:
        """
        Loads the model with specified configuration.

        Args:
            trust_remote_code: Whether to trust remote code for the model.
            for_training: Set to True if loading for training (influences device_map).

        Returns:
            The loaded AutoModelForCausalLM.
        """
        print(f"Loading model: {self.model_name}")
        # For training with Trainer/Accelerate, device_map is often handled automatically or set to "auto".
        # For inference, explicitly mapping to the target device is common if not using "auto".
        device_map_strategy = None if for_training else (self.device if self.device.type != 'cpu' else None)
        # device_map_strategy = "auto" # Alternative if accelerate is installed and preferred

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=self.dtype,
            device_map=device_map_strategy,
        )

        # If not using device_map or device_map="auto", and device is not CPU, move the model manually.
        # This is less common now with device_map but can be a fallback.
        # if device_map_strategy is None and self.device.type != 'cpu':
        #     model.to(self.device)

        print(f"Model loaded successfully. Dtype: {model.dtype}, Device: {model.device}")
        return model

    @staticmethod
    def load_fine_tuned(saved_path: str, device: torch.device, dtype: torch.dtype = None):
        """Loads a fine-tuned model and its tokenizer from a specified path."""
        print(f"\n--- Loading Fine-Tuned Model/Tokenizer from: {saved_path} ---")
        if not os.path.isdir(saved_path):
            print(f"Error: Saved model directory not found at {saved_path}")
            return None, None

        try:
            print("Loading fine-tuned tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(saved_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                print("Set pad_token = eos_token for loaded tokenizer.")

            print("Loading fine-tuned model...")
            # Use the same device/dtype logic as base model loading for consistency
            device_map_strategy = device if device.type != 'cpu' else None
            # device_map_strategy = "auto" # Alternative

            model = AutoModelForCausalLM.from_pretrained(
                saved_path,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map=device_map_strategy,
            )
            # if device_map_strategy is None and device.type != 'cpu':
            #     model.to(device)

            print(f"Fine-tuned model loaded. Dtype: {model.dtype}, Device: {model.device}")
            model.eval() # Set to evaluation mode
            return model, tokenizer

        except Exception as e:
            print(f"An error occurred during fine-tuned model loading: {e}")
            return None, None
