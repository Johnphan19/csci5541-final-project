import os
import gc
import json
import math
import traceback
from tqdm.notebook import tqdm
from datasets import Dataset, DatasetDict, load_dataset
from typing import List, Dict, Optional, Tuple

# Import refactored components
from config_manager import ConfigManager
from model_manager import ModelManager
from inference_engine import InferenceEngine
from utils import evaluate_math_output # Use evaluation util

class EvaluationFramework:
    """Handles loading models, running inference, and evaluating math problems."""

    def __init__(self, config_manager: ConfigManager):
        """
        Initializes the EvaluationFramework.

        Args:
            config_manager: An instance of ConfigManager to access base config.
        """
        self.config_manager = config_manager
        self.base_config = config_manager.get_base_config()
        self.device = self.base_config['DEVICE']
        self.dtype = self.base_config['DTYPE_TO_LOAD']
        print("EvaluationFramework initialized.")

    def _load_evaluation_dataset(self, num_examples: Optional[int] = None) -> Tuple[Optional[Dataset], Optional[Dataset]]:
        """Loads and optionally subsets the validation and test datasets."""
        try:
            dataset_name = self.base_config['BASE_DATASET_NAME']
            dataset_config = self.base_config['BASE_DATASET_CONFIG']
            print(f"Loading dataset: {dataset_name} ({dataset_config})")
            dataset = load_dataset(dataset_name, dataset_config)

            validation_data = dataset.get('validation')
            test_data = dataset.get('test')

            if validation_data is None or test_data is None:
                 print("Error: 'validation' or 'test' split not found in the dataset.")
                 return None, None

            if num_examples is not None:
                print(f"Selecting first {num_examples} examples from validation and test splits.")
                val_count = min(num_examples, len(validation_data))
                test_count = min(num_examples, len(test_data))
                validation_data = validation_data.select(range(val_count))
                test_data = test_data.select(range(test_count))

            print(f"Using {len(validation_data)} validation examples.")
            print(f"Using {len(test_data)} test examples.")
            return validation_data, test_data

        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None, None

    def _generate_and_evaluate_split(self, generator: InferenceEngine, dataset_split: Dataset, split_name: str, batch_size: int, max_new_tokens: int) -> Tuple[float, List[Dict]]:
        """Generates responses for a dataset split and evaluates them."""
        scores = []
        detailed_results = []
        num_batches = math.ceil(len(dataset_split) / batch_size)
        model_name_short = generator.model.config._name_or_path.split('/')[-1] # Get short name for progress bar

        print(f"  Evaluating {split_name} Set (Batch Size: {batch_size})...")
        for i in tqdm(range(num_batches), desc=f"{split_name} Batches ({model_name_short})", leave=False):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(dataset_split))
            batch_examples = dataset_split.select(range(start_idx, end_idx))

            batch_problems = batch_examples['input']
            batch_ground_truths = batch_examples['output_answer']

            # Generate responses
            batch_generated_answers = generator.generate_math_batch(
                batch_problems,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size # Pass batch size to generator method
            )

            # Evaluate
            for idx, (gt, gen) in enumerate(zip(batch_ground_truths, batch_generated_answers)):
                score = evaluate_math_output(gt, gen) # Use util function
                scores.append(score)
                detailed_results.append({
                    'question_id': batch_examples[idx].get('id', f'{split_name}_{start_idx + idx}'), # Use index as fallback ID
                    'problem': batch_problems[idx],
                    'ground_truth': gt,
                    'generated': gen,
                    'score': score
                })

        accuracy = sum(scores) / len(scores) if scores else 0.0
        print(f"  {split_name} Accuracy: {accuracy*100:.2f}%")
        return accuracy * 100, detailed_results


    def _evaluate_single_model(self, model_identifier: str, validation_data: Dataset, test_data: Dataset,
                              batch_size: int, max_new_tokens: int, inference_style: str) -> Tuple[str, Dict, Dict]:
        """Loads, evaluates (using batching), and unloads a single model."""

        # --- Generate a unique short name for the model ---
        if model_identifier == self.base_config['MODEL_NAME']:
            model_name_short = self.base_config['MODEL_NAME'].split('/')[-1] # Use only last part for base
        elif os.path.sep in model_identifier or '/' in model_identifier:
            normalized_path = model_identifier.replace('\\', '/')
            parts = normalized_path.strip('/').split('/')
            # Try to get the parent directory name (experiment name) and 'final_model'
            if len(parts) >= 2 and parts[-1] == 'final_model':
                model_name_short = parts[-2] # Use the parent directory name
            elif len(parts) >= 1:
                 model_name_short = parts[-1] # Fallback to last part if not 'final_model'
            else:
                 model_name_short = model_identifier.replace(os.path.sep, '_').replace('/', '_') # Fallback
        else:
            model_name_short = model_identifier # Assume it's a HF identifier if not a path

        print(f"\n--- Starting evaluation for: {model_name_short} (from: {model_identifier}) ---")

        model_results = {}
        detailed_model_results = {'validation': [], 'test': []}
        model, tokenizer, generator = None, None, None
        config_max_len = self.base_config.get('MAX_INPUT_LENGTH') # Get from base config
        fallback_max_len = self.base_config.get('DEFAULT_FALLBACK_MAX_LENGTH', 4096) # Get from base config
        compile_model_flag = self.base_config.get('COMPILE_MODEL_FOR_EVALUATION', False) # Get compile flag

        try:
            # --- Load model and tokenizer ---
            print(f"  Loading model/tokenizer...")
            if model_identifier == self.base_config['MODEL_NAME']:
                # Use ModelManager instance for base model (doesn't need force_cpu flag here)
                mm = ModelManager(model_identifier, self.device, self.dtype)
                tokenizer = mm.load_tokenizer()
                model = mm.load_model(for_training=False) # Load for inference
            else:
                # Use static method for fine-tuned models
                if not os.path.isdir(model_identifier):
                     raise FileNotFoundError(f"Model directory not found: {model_identifier}")
                model, tokenizer = ModelManager.load_fine_tuned(model_identifier, self.device, self.dtype)

            if model is None or tokenizer is None:
                raise RuntimeError("Failed to load model or tokenizer.")
            print(f"  Model and tokenizer loaded.")

            # --- Setup Generator - Pass max length and compile config values ---
            generator = InferenceEngine(
                model,
                tokenizer,
                self.device,
                inference_style,
                config_max_length=config_max_len,
                fallback_max_length=fallback_max_len,
                compile_model=compile_model_flag # Pass compile flag
            )

            # --- Evaluate Splits ---
            val_acc, val_details = self._generate_and_evaluate_split(generator, validation_data, "Validation", batch_size, max_new_tokens)
            test_acc, test_details = self._generate_and_evaluate_split(generator, test_data, "Test", batch_size, max_new_tokens)

            model_results['validation_score'] = val_acc
            model_results['test_score'] = test_acc
            detailed_model_results['validation'] = val_details
            detailed_model_results['test'] = test_details

        except Exception as e:
            print(f"  Error evaluating {model_identifier} (named: {model_name_short}): {e}")
            traceback.print_exc()
            model_results['validation_score'] = -1.0 # Indicate error
            model_results['test_score'] = -1.0

        finally:
            # --- Clean up resources ---
            print(f"  Cleaning up resources for {model_name_short}...")
            del model
            del tokenizer
            del generator
            gc.collect()
            InferenceEngine.cleanup_memory()
            print(f"  Resource cleanup complete.")

        return model_name_short, model_results, detailed_model_results


    def run_evaluation(self, model_identifiers: List[str], num_examples: Optional[int] = None,
                       inference_batch_size: int = 4, max_new_tokens: int = 1024,
                       inference_style: str = 'think') -> Tuple[Dict, Dict]:
        """
        Runs the evaluation process for a list of model identifiers.

        Args:
            model_identifiers: List of model paths or HF names (including the base model).
            num_examples: Number of examples per split to evaluate (None for all).
            inference_batch_size: Batch size for inference.
            max_new_tokens: Max tokens to generate for each response.
            inference_style: Prompting style ('think' or 'no_think').

        Returns:
            A tuple containing:
                - aggregate_results: Dictionary mapping model short names to scores.
                - detailed_results: Dictionary mapping model short names to detailed outputs per split.
        """
        aggregate_results = {}
        detailed_results = {}

        # 1. Load Dataset
        validation_data, test_data = self._load_evaluation_dataset(num_examples)
        if validation_data is None or test_data is None:
            print("Evaluation aborted due to dataset loading failure.")
            return aggregate_results, detailed_results

        # 2. Evaluate each model sequentially
        for identifier in model_identifiers:
            model_name, model_res, detailed_res = self._evaluate_single_model(
                identifier, validation_data, test_data,
                inference_batch_size, max_new_tokens, inference_style
            )
            # Handle potential name collisions (less likely with new naming)
            if model_name in aggregate_results:
                 print(f"Warning: Duplicate model name '{model_name}' detected for identifier '{identifier}'. Overwriting previous results.")
            aggregate_results[model_name] = model_res
            detailed_results[model_name] = detailed_res
            print(f"--- Completed evaluation for: {model_name} ---")

        print("\n--- Overall Evaluation Complete ---")
        print("Aggregated Results:")
        print(json.dumps(aggregate_results, indent=2))
        return aggregate_results, detailed_results

    @staticmethod
    def save_results(results: Dict, filename: str):
        """Saves the results dictionary to a JSON file."""
        print(f"Saving results to {filename}...")
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            print("Results saved successfully.")
        except Exception as e:
            print(f"Error saving results to {filename}: {e}")

    @staticmethod
    def load_results(filename: str) -> Optional[Dict]:
        """Loads results from a JSON file."""
        print(f"Loading results from {filename}...")
        try:
            with open(filename, 'r') as f:
                loaded_results = json.load(f)
            print("Results loaded successfully.")
            return loaded_results
        except FileNotFoundError:
            print(f"Error: Results file {filename} not found.")
            return None
        except Exception as e:
            print(f"Error loading results from {filename}: {e}")
            return None

print("EvaluationFramework loaded.")
