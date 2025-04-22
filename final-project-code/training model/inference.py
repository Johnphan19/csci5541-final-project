import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
import config # Import the configuration

class Generator:
    """Handles text generation using a loaded model and tokenizer."""

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, device: torch.device):
        """
        Initializes the Generator.

        Args:
            model: The loaded model (base or fine-tuned) for generation.
            tokenizer: The corresponding tokenizer.
            device: The torch device the model is on.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval() # Ensure model is in eval mode
        try:
            self.model_max_length = self.model.config.max_position_embeddings
        except AttributeError:
            print(f"Warning: Could not get max_position_embeddings for model {model.config._name_or_path}. Using default max_length={config.MAX_INPUT_LENGTH}.")
            self.model_max_length = config.MAX_INPUT_LENGTH # Fallback
        print(f"Generator initialized. Model max length: {self.model_max_length}")


    def generate_math_response(self, problem: str, max_new_tokens: int = config.MAX_NEW_TOKENS_MATH) -> str:
        """Generates a solution for a given math problem prompt."""
        prompt = f"Problem:\n{problem}\n\nSolution:\n"
        response_text = "[Error during generation]"
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.model_max_length - max_new_tokens - 10 # Leave buffer
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=False, # Use greedy decoding for math problems
                    num_beams=1,
                )

            # Decode only the newly generated tokens
            generated_ids = outputs[0, inputs['input_ids'].shape[1]:]
            response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        except Exception as e:
            print(f"\nError during math generation: {e}")
            if "out of memory" in str(e).lower():
                response_text = "[OOM Error during generation]"
            # Handle other specific errors like device loss if necessary

        return response_text.strip()


    def generate_general_response(self, prompt: str, max_new_tokens: int = config.MAX_NEW_TOKENS_NON_MATH) -> str:
        """Generates a response for a general prompt using sampling."""
        response_text = "[Error during generation]"
        try:
            # Calculate max length for input tokens, leaving space for generation
            input_max_len = max(0, self.model_max_length - max_new_tokens - 20) # Buffer

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=input_max_len
            ).to(self.device)

            if inputs['input_ids'].shape[1] == 0:
                 print(f"  Warning: Input prompt resulted in zero tokens after tokenization/truncation. Prompt: '{prompt[:100]}...'")
                 return "[Input prompt too long or empty after processing]"

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    # Sampling parameters for general text
                    do_sample=True,
                    top_k=40,
                    top_p=0.9,
                    temperature=0.6,
                )

            generated_ids = outputs[0, inputs['input_ids'].shape[1]:]
            response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        except Exception as e:
            print(f"\n   Error generating general response for prompt '{prompt[:50]}...': {e}")
            if "out of memory" in str(e).lower():
                 print("   OOM Error during generation. Try reducing max_new_tokens or using lower precision/quantization.")
                 return "[OOM Error during generation]"
            # Handle other specific errors

        return response_text.strip()

    @staticmethod
    def compare_outputs(dataset: Dataset, generator_finetuned: 'Generator', generator_base: 'Generator', num_examples: int):
        """Generates and compares outputs from fine-tuned and base models for math problems."""
        print(f"\n--- Comparing Outputs for First {num_examples} Validation Examples ---")

        if 'validation' not in dataset:
            print("Error: 'validation' split not found in the dataset.")
            return

        validation_subset = dataset['validation'].select(range(min(num_examples, len(dataset['validation']))))
        input_column = 'input'
        output_column = 'output_answer'

        if input_column not in validation_subset.features or output_column not in validation_subset.features:
             print(f"Error: Required columns ('{input_column}', '{output_column}') not found in validation subset.")
             return

        for i, example in enumerate(validation_subset):
            print(f"\n--- Example {i+1} ---")
            problem = example[input_column]
            actual_solution = example[output_column]

            print(f"Problem:\n{problem[:500]}...\n") # Truncate long problems
            print(f"Actual Solution:\n{actual_solution}\n")

            if generator_finetuned:
                print("Generating with Fine-Tuned Model...")
                ft_solution = generator_finetuned.generate_math_response(problem)
                print(f"Fine-Tuned Model Solution:\n{ft_solution}\n")
            else:
                print("Skipping Fine-Tuned Model (not provided).\n")

            if generator_base:
                print("Generating with Base Model...")
                base_solution = generator_base.generate_math_response(problem)
                print(f"Base Model Solution:\n{base_solution}")
            else:
                print("Skipping Base Model (not provided).")

            print("-" * 30)

    @staticmethod
    def test_non_math_generation(prompts: list, generator_finetuned: 'Generator', generator_base: 'Generator'):
        """Generates responses for non-math prompts using both models."""
        print("\n\n--- Testing Non-Math Generation ---")

        if generator_finetuned:
            print("\n--- Generating Non-Math with FINE-TUNED Model ---")
            for i, prompt in enumerate(prompts):
                print(f"\nPrompt {i+1}: {prompt}")
                response = generator_finetuned.generate_general_response(prompt)
                print(f"Fine-Tuned Model Response:\n{response}")
                print("-" * 20)
        else:
             print("\nSkipping Non-Math Generation with Fine-Tuned Model (not provided).")


        if generator_base:
            print("\n--- Generating Non-Math with BASE Model ---")
            for i, prompt in enumerate(prompts):
                print(f"\nPrompt {i+1}: {prompt}")
                response = generator_base.generate_general_response(prompt)
                print(f"Base Model Response:\n{response}")
                print("-" * 20)
        else:
            print("\nSkipping Non-Math Generation with Base Model (not provided).")

        print("\n--- Non-Math Generation Test Complete ---")

    @staticmethod
    def cleanup_memory():
        """Attempts to clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Cleared CUDA cache.")
        elif hasattr(torch.xpu, 'empty_cache') and torch.xpu.is_available():
            torch.xpu.empty_cache()
            print("Cleared XPU cache.")

