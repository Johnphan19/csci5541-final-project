from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
import torch # Import torch
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union # For typing

# Import helpers from utils
from utils import find_sublist, find_last_sublist

# Custom Data Collator for Masking Prompt
class PromptMaskingDataCollator(DataCollatorForLanguageModeling):
    """
    Data collator that masks the prompt portion (up to and including a specified marker
    indicating the start of the assistant's turn) for loss calculation.
    Loss is only calculated on the tokens belonging to the assistant's response.
    Assumes the input has been formatted using a chat template.
    """
    def __init__(self, tokenizer: AutoTokenizer, assistant_marker_style: str = 'no_think', **kwargs):
        super().__init__(tokenizer=tokenizer, **kwargs)
        self.assistant_prompt_marker_ids = []
        self.assistant_marker_style = assistant_marker_style

        # Define marker strings based on the chosen style
        marker_str_no_think = "<｜Assistant｜>" # Marker *without* the <think> tag
        marker_str_think = "</think>" # Marker *including* the <think> tag

        if self.assistant_marker_style == 'think':
            assistant_prompt_marker_str = marker_str_think
            print(f"Using 'think' style marker: '{assistant_prompt_marker_str}'")
        elif self.assistant_marker_style == 'no_think':
            assistant_prompt_marker_str = marker_str_no_think
            print(f"Using 'no_think' style marker: '{assistant_prompt_marker_str}'")
        else:
            print(f"Warning: Unknown assistant_marker_style '{self.assistant_marker_style}'. Defaulting to 'no_think' style marker: '{marker_str_no_think}'.")
            assistant_prompt_marker_str = marker_str_no_think # Default to standard marker

        # Encode the chosen marker string
        if assistant_prompt_marker_str:
            self.assistant_prompt_marker_ids = self.tokenizer.encode(assistant_prompt_marker_str, add_special_tokens=False)
            print(f"PromptMaskingDataCollator: Initialized with style '{self.assistant_marker_style}'. Masking up to marker: '{assistant_prompt_marker_str}', IDs: {self.assistant_prompt_marker_ids}")
        else:
            print("Warning: Assistant prompt marker string is empty. Prompt masking will likely fail.")

        if not self.assistant_prompt_marker_ids:
             print("Warning: Assistant prompt marker IDs are empty after encoding. Prompt masking will likely fail.")

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        labels = batch["labels"].clone()
        input_ids = batch["input_ids"]

        for i in range(labels.size(0)):
            input_ids_list = input_ids[i].tolist()
            mask_until_idx = -1

            if self.assistant_prompt_marker_ids:
                start_idx = find_last_sublist(input_ids_list, self.assistant_prompt_marker_ids)

                if start_idx != -1:
                    mask_until_idx = start_idx + len(self.assistant_prompt_marker_ids)
                else:
                    print(f"Warning: Assistant prompt marker {self.assistant_prompt_marker_ids} ('{self.tokenizer.decode(self.assistant_prompt_marker_ids)}') not found in example {i}. Loss calculation might be incorrect. Example start: {self.tokenizer.decode(input_ids_list[:50])}")
                    continue
            else:
                 print(f"Warning: assistant_prompt_marker_ids is empty for example {i}. Skipping masking.")
                 continue

            if mask_until_idx != -1 and mask_until_idx < labels.size(1):
                labels[i, :mask_until_idx] = -100
            elif mask_until_idx >= labels.size(1):
                 print(f"Warning: Calculated mask index {mask_until_idx} is out of bounds for labels length {labels.size(1)} in example {i}. Masking entire sequence.")
                 labels[i, :] = -100

        batch["labels"] = labels
        return batch

class DatasetManager:
    """Handles dataset loading, preprocessing using chat templates, and collation."""

    def __init__(self, tokenizer: AutoTokenizer, config_max_length: Optional[int], fallback_max_length: int):
        """
        Initializes the DatasetManager and determines the effective max length.

        Args:
            tokenizer: The tokenizer to use for preprocessing (must have chat_template).
            config_max_length: The max_length value from the configuration (can be None).
            fallback_max_length: The fallback value to use if needed.
        """
        self.tokenizer = tokenizer
        self.config_max_length = config_max_length
        self.fallback_max_length = fallback_max_length
        self.effective_max_length = self._determine_effective_max_length()

        if not self.tokenizer.chat_template:
            print("Warning: Tokenizer does not have a chat_template defined. Preprocessing might fail or produce unexpected results.")

        print(f"DatasetManager initialized. Effective max_length={self.effective_max_length}. Using tokenizer chat template for formatting.")

    def _determine_effective_max_length(self) -> int:
        """Determines the max length to use based on config and tokenizer."""
        if isinstance(self.config_max_length, int):
            print(f"Using max_length from config: {self.config_max_length}")
            return self.config_max_length
        else:
            tokenizer_max_len = self.tokenizer.model_max_length
            # Handle cases where tokenizer_max_len is None or excessively large
            if isinstance(tokenizer_max_len, int) and tokenizer_max_len < 1e6:
                print(f"Using max_length from tokenizer: {tokenizer_max_len}")
                return tokenizer_max_len
            else:
                print(f"Using fallback max_length: {self.fallback_max_length} (Config: {self.config_max_length}, Tokenizer: {tokenizer_max_len})")
                return self.fallback_max_length

    def load_base_dataset(self, dataset_name: str, dataset_config: Optional[str] = None) -> DatasetDict:
        print(f"Loading base dataset: {dataset_name} ({dataset_config or 'default'})")
        try:
            dataset = load_dataset(dataset_name, dataset_config)
            print(f"Base dataset loaded successfully: {dataset}")
            return dataset
        except Exception as e:
            print(f"Error loading base dataset {dataset_name} ({dataset_config}): {e}")
            raise

    def load_train_split_from_json(self, json_path: str) -> Dataset:
        print(f"Loading training data split from: {json_path}")
        try:
            train_dataset = load_dataset('json', data_files={'train': json_path})['train']
            print(f"Training split loaded successfully from {json_path}. Num examples: {len(train_dataset)}")
            return train_dataset
        except Exception as e:
            print(f"Error loading training data from {json_path}: {e}")
            raise

    def _preprocess_function_finetune(self, examples):
        batch_messages = []
        required_cols = ['input', 'output_answer']
        if not all(col in examples for col in required_cols):
             print(f"Error: Preprocessing function (finetune) missing required columns: {required_cols}. Available: {list(examples.keys())}")
             raise ValueError(f"Missing required columns {required_cols} in dataset split.")

        for problem, assistant_response in zip(examples['input'], examples['output_answer']):
            messages = [
                {"role": "user", "content": f"Please reason step by step, and put your final answer within \\boxed{{}}.\n{problem}"},
                {"role": "assistant", "content": assistant_response }
            ]
            batch_messages.append(messages)

        formatted_texts = [
            self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in batch_messages
        ]
        formatted_texts_with_eos = [text + self.tokenizer.eos_token for text in formatted_texts]

        model_inputs = self.tokenizer(
            formatted_texts_with_eos,
            max_length=self.effective_max_length, # Use determined max length
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs

    def _preprocess_function_gradient_ascent(self, examples):
        batch_messages = []
        required_cols = ['input', 'output_answer']
        if not all(col in examples for col in required_cols):
             print(f"Error: Preprocessing function (gradient_ascent) missing required columns: {required_cols}. Available: {list(examples.keys())}")
             raise ValueError(f"Missing required columns {required_cols} in dataset split.")

        for problem, assistant_response in zip(examples['input'], examples['output_answer']):
            user_content = f"Please reason step by step, and put your final answer within \\boxed{{}}.\n{problem}"
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_response}
            ]
            batch_messages.append(messages)

        formatted_texts = [
            self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in batch_messages
        ]
        formatted_texts_with_eos = [text + self.tokenizer.eos_token for text in formatted_texts]

        model_inputs = self.tokenizer(
            formatted_texts_with_eos,
            truncation=True,
            max_length=self.effective_max_length, # Use determined max length
            padding=False
        )
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs

    def prepare_and_tokenize(self, base_dataset: DatasetDict, training_type: str, train_json_path: Optional[str] = None) -> DatasetDict:
        dataset_to_process = base_dataset.copy()

        if training_type == 'finetune':
            if not train_json_path:
                raise ValueError("train_json_path must be provided for 'finetune' training type.")
            try:
                modified_train_split = self.load_train_split_from_json(train_json_path)
                dataset_to_process['train'] = modified_train_split
                print("Replaced 'train' split with data from JSON for fine-tuning.")
                preprocess_func = self._preprocess_function_finetune
            except Exception as e:
                print(f"Failed to load or replace train split for fine-tuning: {e}")
                raise
        elif training_type == 'gradient_ascent':
            print("Using original 'train' split from base dataset for gradient ascent.")
            preprocess_func = self._preprocess_function_gradient_ascent
        else:
            raise ValueError(f"Unknown training_type: {training_type}. Must be 'finetune' or 'gradient_ascent'.")

        print(f"Tokenizing dataset for training type: {training_type}...")
        original_columns_per_split = {split: list(dataset_to_process[split].features.keys()) for split in dataset_to_process.keys()}

        try:
            tokenized_dataset = dataset_to_process.map(
                preprocess_func,
                batched=True,
            )
            print("Preprocessing function applied.")
        except Exception as e:
             print(f"Error during dataset mapping for {training_type}: {e}")
             raise

        final_tokenized_dataset = DatasetDict()
        kept_columns = ['input_ids', 'attention_mask', 'labels']

        for split, original_cols in original_columns_per_split.items():
            if split in tokenized_dataset:
                current_cols_in_split = tokenized_dataset[split].column_names
                cols_to_remove = [
                    col for col in original_cols
                    if col in current_cols_in_split and col not in kept_columns
                ]

                try:
                    if cols_to_remove:
                         final_tokenized_dataset[split] = tokenized_dataset[split].remove_columns(cols_to_remove)
                         print(f"Removed columns {cols_to_remove} from split '{split}'.")
                    else:
                         final_tokenized_dataset[split] = tokenized_dataset[split]
                         print(f"No original columns needed removal from split '{split}'.")

                    final_cols = final_tokenized_dataset[split].column_names
                    if not all(k_col in final_cols for k_col in kept_columns if k_col in tokenized_dataset[split].column_names):
                         print(f"Warning: Split '{split}' after column removal is missing some expected columns. Final columns: {final_cols}, Expected subset: {kept_columns}")

                except ValueError as e:
                     print(f"Warning: Could not remove columns from split '{split}'. Error: {e}")
                     print(f"Columns in '{split}' before removal attempt: {current_cols_in_split}")
                     print(f"Columns intended for removal: {cols_to_remove}")
                     final_tokenized_dataset[split] = tokenized_dataset[split]
            else:
                 print(f"Warning: Split '{split}' not found in tokenized_dataset after mapping.")

        print("Tokenization and column cleanup complete.")
        if "train" in final_tokenized_dataset and len(final_tokenized_dataset["train"]) > 0:
             print(f"Final tokenized training dataset features: {final_tokenized_dataset['train'].features}")
        if "validation" in final_tokenized_dataset and len(final_tokenized_dataset["validation"]) > 0:
             print(f"Final tokenized validation dataset features: {final_tokenized_dataset['validation'].features}")

        return final_tokenized_dataset

    def detokenize_dataset(self, tokenized_dataset: DatasetDict, num_examples: int = 3) -> Dict[str, List[str]]:
        print("\nDetokenizing examples for inspection...")
        detokenized_examples = {}
        for split_name, split_data in tokenized_dataset.items():
            print(f"  Processing split: {split_name}")
            if 'input_ids' not in split_data.features:
                print(f"    Warning: 'input_ids' not found in split '{split_name}'. Skipping detokenization.")
                continue

            count = min(num_examples, len(split_data))
            if count == 0:
                print(f"    Split '{split_name}' is empty. Skipping.")
                continue

            input_ids_batch = split_data.select(range(count))['input_ids']

            try:
                detokenized_texts = self.tokenizer.batch_decode(input_ids_batch, skip_special_tokens=False)
                detokenized_examples[split_name] = detokenized_texts
                print(f"    Detokenized {count} examples from '{split_name}'.")
            except Exception as e:
                print(f"    Error detokenizing split '{split_name}': {e}")
                detokenized_examples[split_name] = [f"[Error detokenizing: {e}]"]

        print("Detokenization inspection complete.")
        return detokenized_examples

    def get_data_collator(self, training_type: str, assistant_marker_style: str = 'no_think') -> Union[PromptMaskingDataCollator, DataCollatorForLanguageModeling]:
        if training_type == 'finetune':
            print(f"Initializing custom PromptMaskingDataCollator with style: '{assistant_marker_style}'")
            data_collator = PromptMaskingDataCollator(
                tokenizer=self.tokenizer,
                mlm=False,
                assistant_marker_style=assistant_marker_style
            )
            return data_collator
        elif training_type == 'gradient_ascent':
            print("Initializing standard DataCollatorForLanguageModeling for gradient ascent.")
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
            return data_collator
        else:
            raise ValueError(f"Unknown training_type: {training_type}. Cannot determine data collator.")

print("DatasetManager loaded.")
