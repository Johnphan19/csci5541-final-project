import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from typing import List, Dict, Optional
import gc
import time # Import time for compile timing

class InferenceEngine:
    """Handles text generation using a loaded model and tokenizer."""

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, device: torch.device,
                 inference_style: str = 'think', config_max_length: Optional[int] = None, fallback_max_length: int = 4096,
                 compile_model: bool = False):
        """
        Initializes the InferenceEngine and determines effective max length.

        Args:
            model: The loaded model (base or fine-tuned) for generation.
            tokenizer: The corresponding tokenizer (must have chat_template).
            device: The torch device the model is on.
            inference_style: Determines the prompt format for generation ('think' or 'no_think').
            config_max_length: The max_length value from the configuration (can be None).
            fallback_max_length: The fallback value to use if needed.
            compile_model: If True, attempts to compile the model using torch.compile.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.inference_style = inference_style
        self.config_max_length = config_max_length
        self.fallback_max_length = fallback_max_length
        self.model.eval() # Ensure model is in eval mode

        if not self.tokenizer.chat_template:
             print("Warning: Tokenizer does not have a chat_template defined. Inference formatting might fail or produce unexpected results.")

        self.marker_str_no_think = "<｜Assistant｜>"

        self.effective_max_length = self._determine_effective_max_length()

        print(f"InferenceEngine initialized with inference_style='{self.inference_style}'. Effective max length: {self.effective_max_length}")

        # --- Attempt Model Compilation ---
        if compile_model:
            print(f"Attempting to compile model with torch.compile (Device: {self.device})...")
            # Check PyTorch version compatibility (torch.compile introduced in 2.0)
            if hasattr(torch, 'compile') and int(torch.__version__.split('.')[0]) >= 2:
                try:
                    start_time = time.time()
                    # Use default mode for broad compatibility initially
                    self.model = torch.compile(self.model)
                    end_time = time.time()
                    print(f"Model compiled successfully in {end_time - start_time:.2f} seconds.")
                    # Optional: Add a dummy forward pass to trigger compilation if needed,
                    # but generate methods will trigger it anyway.
                except Exception as e:
                    print(f"Warning: Model compilation failed: {e}. Proceeding without compilation.")
            else:
                print("Warning: torch.compile not available or PyTorch version < 2.0. Skipping compilation.")
        else:
            print("Model compilation not requested.")

    def _determine_effective_max_length(self) -> int:
        """Determines the max length to use based on config, model, and fallback."""
        if isinstance(self.config_max_length, int):
            print(f"Using max_length from config: {self.config_max_length}")
            return self.config_max_length
        else:
            try:
                model_max_len = self.model.config.max_position_embeddings
                if isinstance(model_max_len, int) and model_max_len < 1e6:
                    print(f"Using max_length from model config: {model_max_len}")
                    return model_max_len
                else:
                     raise AttributeError # Treat large/invalid values as missing
            except AttributeError:
                print(f"Using fallback max_length: {self.fallback_max_length} (Config: {self.config_max_length}, Model config missing/invalid)")
                return self.fallback_max_length

    def _format_prompt(self, messages: list) -> str:
        """Applies chat template based on the inference_style."""
        try:
            if self.inference_style == 'think':
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True # Adds role marker, e.g., "<|Assistant|>"
                )
            elif self.inference_style == 'no_think':
                prompt_base = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False # Get only user part formatted
                )
                return prompt_base + self.marker_str_no_think # e.g., "<|Assistant|>"
            else:
                print(f"Warning: Unknown inference_style '{self.inference_style}'. Defaulting to 'think' style formatting (using add_generation_prompt=True).")
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
        except Exception as e:
            print(f"Error applying chat template: {e}")
            return "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages]) + "\nassistant:"

    def generate_math_batch(self, problems: list[str], max_new_tokens: int, batch_size: int = 8, temperature: float = 0.6, do_sample: bool = True) -> list[str]:
        """Generates solutions for a batch of math problems using the configured inference style."""
        all_responses = []
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        try:
            for i in range(0, len(problems), batch_size):
                batch_problems = problems[i:i+batch_size]

                batch_messages = [
                    [{"role": "user", "content": f"Please reason step by step, and put your final answer within \\boxed{{}}.\n{p}"}]
                    for p in batch_problems
                ]

                batch_prompt_strings = [self._format_prompt(msgs) for msgs in batch_messages]

                max_input_len = max(10, self.effective_max_length - max_new_tokens - 10) # Ensure positive, leave buffer
                inputs = self.tokenizer(
                    batch_prompt_strings,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_input_len # Use calculated max input len for this batch
                ).to(self.device)

                if torch.any(inputs['attention_mask'].sum(dim=1) == 0):
                     print(f"Warning: Some inputs in batch {i//batch_size} became empty after tokenization/truncation.")

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id, # Stop generation at EOS
                        do_sample=do_sample,
                        temperature=temperature,
                    )

                input_lengths = inputs['input_ids'].shape[1]
                generated_ids = outputs[:, input_lengths:]
                batch_response_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                all_responses.extend([resp.strip() for resp in batch_response_texts])

        except Exception as e:
            print(f"\nError during batch math generation: {e}")
            error_placeholder = "[Error during generation]"
            num_missing = len(problems) - len(all_responses)
            all_responses.extend([error_placeholder] * num_missing)
            if "out of memory" in str(e).lower():
                 print("OOM Error during batch generation. Try reducing batch_size or max_new_tokens.")
                 all_responses = [resp if resp != error_placeholder else "[OOM Error]" for resp in all_responses]
        finally:
            self.tokenizer.padding_side = original_padding_side

        return all_responses

    def generate_general_response(self, prompt: str, max_new_tokens: int, temperature: float = 0.6, do_sample: bool = True) -> str:
        """Generates a response for a general prompt using the configured inference style."""
        response_text = "[Error during generation]"
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        try:
            messages = [{"role": "user", "content": prompt}]

            formatted_prompt = self._format_prompt(messages)

            max_input_len = max(10, self.effective_max_length - max_new_tokens - 10) # Ensure positive, leave buffer
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_len # Use calculated max input len
            ).to(self.device)

            if inputs['input_ids'].shape[1] == 0:
                 print(f"  Warning: Input prompt resulted in zero tokens after tokenization/truncation. Prompt: '{formatted_prompt[:100]}...'")
                 return "[Input prompt too long or empty after processing]"

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id, # Stop generation at EOS
                    do_sample=do_sample,
                    temperature=temperature,
                )

            generated_ids = outputs[0, inputs['input_ids'].shape[1]:]
            response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        except Exception as e:
            print(f"\n   Error generating general response for prompt '{prompt[:50]}...': {e}")
            if "out of memory" in str(e).lower():
                 print("   OOM Error during generation. Try reducing max_new_tokens or using lower precision/quantization.")
                 return "[OOM Error during generation]"
        finally:
            self.tokenizer.padding_side = original_padding_side

        return response_text.strip()

    @staticmethod
    def cleanup_memory():
        """Attempts to clear GPU memory cache."""
        print("Attempting to clear GPU cache...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Cleared CUDA cache.")
        elif hasattr(torch.xpu, 'empty_cache') and torch.xpu.is_available():
            torch.xpu.empty_cache()
            print("Cleared XPU cache.")
        else:
            print("No CUDA or XPU cache to clear.")

print("InferenceEngine loaded.")
