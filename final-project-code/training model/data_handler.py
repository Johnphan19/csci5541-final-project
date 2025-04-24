from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
import config # Import the configuration
import torch # Import torch
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union # For typing

# Helper function to find the first occurrence of a sublist
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

# Helper function to find the *last* occurrence of a sublist
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

# Custom Data Collator for Masking Prompt
class PromptMaskingDataCollator(DataCollatorForLanguageModeling):
    """
    Data collator that masks the prompt portion (up to and including a specified marker
    indicating the start of the assistant's turn) for loss calculation.
    Loss is only calculated on the tokens belonging to the assistant's response.
    Assumes the input has been formatted using a chat template.
    """
    def __init__(self, tokenizer: AutoTokenizer, assistant_marker_style: str = 'no_think', **kwargs):
        """
        Initializes the collator.

        Args:
            tokenizer: The tokenizer instance.
            assistant_marker_style: Determines the marker sequence to mask up to.
                                    'no_think': Masks up to '<｜Assistant｜><think>\n\n</think>\n' (or similar standard marker).
                                    'think': Masks up to '<｜Assistant｜><think>\n'.
            **kwargs: Additional arguments for DataCollatorForLanguageModeling.
        """
        super().__init__(tokenizer=tokenizer, **kwargs)
        self.assistant_prompt_marker_ids = []
        self.assistant_marker_style = assistant_marker_style

        # Define marker strings based on the chosen style
        # Note: Adjust these strings precisely if the actual tokens differ slightly
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
        # Get standard batch processing from parent (padding, initial labels)
        batch = super().torch_call(examples)

        # Clone labels to modify them
        labels = batch["labels"].clone()
        input_ids = batch["input_ids"]

        # Iterate through each example in the batch
        for i in range(labels.size(0)):
            input_ids_list = input_ids[i].tolist()
            mask_until_idx = -1 # Index *after* the marker sequence to start calculating loss

            # --- Find the last occurrence of the assistant prompt marker ---
            if self.assistant_prompt_marker_ids:
                # We look for the *last* occurrence in case the marker appears elsewhere unexpectedly
                start_idx = find_last_sublist(input_ids_list, self.assistant_prompt_marker_ids)

                if start_idx != -1:
                    # Mask everything up to and including the *last* token of the marker sequence
                    mask_until_idx = start_idx + len(self.assistant_prompt_marker_ids)
                else:
                    # Marker not found - this indicates an issue with data or template application
                    print(f"Warning: Assistant prompt marker {self.assistant_prompt_marker_ids} ('{self.tokenizer.decode(self.assistant_prompt_marker_ids)}') not found in example {i}. Loss calculation might be incorrect. Example start: {self.tokenizer.decode(input_ids_list[:50])}")
                    continue # Skip specific masking for this example if marker not found
            else:
                 print(f"Warning: assistant_prompt_marker_ids is empty for example {i}. Skipping masking.")
                 continue

            # --- Apply Masking ---
            if mask_until_idx != -1 and mask_until_idx < labels.size(1): # Ensure index is valid
                labels[i, :mask_until_idx] = -100
            elif mask_until_idx >= labels.size(1):
                 print(f"Warning: Calculated mask index {mask_until_idx} is out of bounds for labels length {labels.size(1)} in example {i}. Masking entire sequence.")
                 labels[i, :] = -100

        # Replace original labels with modified ones
        batch["labels"] = labels
        return batch

class DataHandler:
    """Handles dataset loading, preprocessing using chat templates, and collation."""

    def __init__(self, tokenizer: AutoTokenizer, max_length: int = config.MAX_INPUT_LENGTH):
        """
        Initializes the DataHandler.

        Args:
            tokenizer: The tokenizer to use for preprocessing (must have chat_template).
            max_length: The maximum sequence length for tokenization.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        if not self.tokenizer.chat_template:
            print("Warning: Tokenizer does not have a chat_template defined. Preprocessing might fail or produce unexpected results.")

        print(f"DataHandler initialized. Using tokenizer chat template for formatting.")

    def load_and_prepare_datasets(self, base_dataset_name: str, base_dataset_config: str, train_json_path: str) -> DatasetDict:
        """Loads the base dataset, replaces the train split, and returns it."""
        print(f"Loading base dataset: {base_dataset_name} ({base_dataset_config})")
        dataset = load_dataset(base_dataset_name, base_dataset_config)
        print(f"Original dataset structure:\n{dataset}")

        print(f"Loading modified training data from: {train_json_path}")
        try:
            raw_train_dataset = load_dataset('json', data_files={'train': train_json_path})['train']
            dataset['train'] = raw_train_dataset
            print("Training dataset replaced successfully.")
            print(f"New dataset structure:\n{dataset}")

            required_cols = ['input', 'output_answer']
            for split in ['train', 'validation', 'test']:
                if split in dataset:
                    if not all(col in dataset[split].features for col in required_cols):
                         print(f"Warning: Split '{split}' is missing one or more required columns: {required_cols}. Features: {list(dataset[split].features.keys())}")
                else:
                     print(f"Warning: Split '{split}' not found in the dataset.")


        except Exception as e:
            print(f"Error loading or replacing training data from {train_json_path}: {e}")
            print("Proceeding with original dataset structure due to error.")

        return dataset

    def _preprocess_function(self, examples):
        """
        Internal preprocessing function to format and tokenize examples using the chat template.
        Assumes 'input' is the problem and 'output_answer' contains the full assistant response
        including the <think>...</think> block and the final answer.
        """
        batch_messages = []
        required_cols = ['input', 'output_answer']
        if not all(col in examples for col in required_cols):
             print(f"Error: Preprocessing function missing required columns: {required_cols}. Available: {list(examples.keys())}")
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
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )
        return model_inputs

    def tokenize_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """Applies the preprocessing function (using chat template) to the dataset."""
        print("Tokenizing dataset...")
        original_columns_per_split = {split: list(dataset[split].features.keys()) for split in dataset.keys()}
        tokenized_dataset = dataset.map(
            self._preprocess_function,
            batched=True,
        )
        print("Preprocessing function applied.")

        final_tokenized_dataset = DatasetDict()
        kept_columns = ['input_ids', 'attention_mask']

        for split, original_cols in original_columns_per_split.items():
            if split in tokenized_dataset:
                cols_to_remove = [col for col in original_cols if col not in kept_columns]
                try:
                    current_cols_in_split = tokenized_dataset[split].column_names
                    actual_cols_to_remove = [col for col in cols_to_remove if col in current_cols_in_split]
                    if actual_cols_to_remove:
                         final_tokenized_dataset[split] = tokenized_dataset[split].remove_columns(actual_cols_to_remove)
                         print(f"Removed columns {actual_cols_to_remove} from split '{split}'.")
                    else:
                         final_tokenized_dataset[split] = tokenized_dataset[split]
                         print(f"No original columns needed removal from split '{split}'.")

                except ValueError as e:
                     print(f"Warning: Could not remove columns from split '{split}'. Error: {e}")
                     print(f"Columns in '{split}' before removal attempt: {tokenized_dataset[split].column_names}")
                     print(f"Columns intended for removal: {cols_to_remove}")
                     final_tokenized_dataset[split] = tokenized_dataset[split]
            else:
                 print(f"Warning: Split '{split}' not found in tokenized_dataset after mapping.")

        print("Original columns removed after tokenization.")
        if "train" in final_tokenized_dataset and len(final_tokenized_dataset["train"]) > 0:
             print(f"Final tokenized training dataset example (first item keys): {list(final_tokenized_dataset['train'].features.keys())}")
        if "validation" in final_tokenized_dataset and len(final_tokenized_dataset["validation"]) > 0:
             print(f"Final tokenized validation dataset example (first item keys): {list(final_tokenized_dataset['validation'].features.keys())}")

        return final_tokenized_dataset

    def detokenize_dataset(self, tokenized_dataset: DatasetDict, num_examples: int = 3) -> Dict[str, List[str]]:
        """
        Detokenizes a few examples from each split of the tokenized dataset for inspection.

        Args:
            tokenized_dataset: The dataset containing 'input_ids'.
            num_examples: The number of examples to detokenize from each split.

        Returns:
            A dictionary where keys are split names and values are lists of detokenized strings.
        """
        print("\nDetokenizing examples for inspection...")
        detokenized_examples = {}
        for split_name, split_data in tokenized_dataset.items():
            print(f"  Processing split: {split_name}")
            if 'input_ids' not in split_data.features:
                print(f"    Warning: 'input_ids' not found in split '{split_name}'. Skipping detokenization.")
                continue

            # Select a small number of examples
            count = min(num_examples, len(split_data))
            if count == 0:
                print(f"    Split '{split_name}' is empty. Skipping.")
                continue

            # Get the input_ids for the selected examples
            input_ids_batch = split_data.select(range(count))['input_ids']

            # Detokenize
            try:
                detokenized_texts = self.tokenizer.batch_decode(input_ids_batch, skip_special_tokens=False) # Keep special tokens to see full structure
                detokenized_examples[split_name] = detokenized_texts
                print(f"    Detokenized {count} examples from '{split_name}'.")
            except Exception as e:
                print(f"    Error detokenizing split '{split_name}': {e}")
                detokenized_examples[split_name] = [f"[Error detokenizing: {e}]"]

        print("Detokenization inspection complete.")
        return detokenized_examples

    def get_data_collator(self, assistant_marker_style: str = 'no_think') -> DataCollatorForLanguageModeling:
        """
        Returns the custom data collator for language modeling with prompt masking.

        Args:
            assistant_marker_style: The style of marker to use for masking ('think' or 'no_think').

        Returns:
            The configured PromptMaskingDataCollator instance.
        """
        print(f"Initializing custom PromptMaskingDataCollator with style: '{assistant_marker_style}'")

        data_collator = PromptMaskingDataCollator(
            tokenizer=self.tokenizer,
            mlm=False,
            assistant_marker_style=assistant_marker_style
        )
        return data_collator
