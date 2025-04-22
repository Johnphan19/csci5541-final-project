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

# Custom Data Collator for Masking Prompt and Thoughts
class PromptMaskingDataCollator(DataCollatorForLanguageModeling):
    """
    Data collator that masks the prompt and thought portions (up to and including
    the *last token* of the first occurrence of THINK_END_SEQUENCE) for loss calculation.
    Loss is only calculated on the tokens that come *after* this sequence.
    """
    def __init__(self, tokenizer: AutoTokenizer, think_end_sequence_ids: List[int], **kwargs):
        super().__init__(tokenizer=tokenizer, **kwargs)
        self.think_end_sequence_ids = think_end_sequence_ids
        print(f"PromptMaskingDataCollator: Initialized. Masking up to the first occurrence of sequence IDs: {self.think_end_sequence_ids}")
        if not self.think_end_sequence_ids:
             print("Warning: Think end sequence IDs are empty or invalid. Prompt masking will likely fail.")

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Get standard batch processing from parent (padding, initial labels)
        batch = super().torch_call(examples)

        # Clone labels to modify them
        labels = batch["labels"].clone()
        input_ids = batch["input_ids"]

        # Iterate through each example in the batch
        for i in range(labels.size(0)):
            input_ids_list = input_ids[i].tolist()
            mask_until_idx = -1 # Index *after* the sequence to start calculating loss

            # --- Find the end of the think sequence (using the full sequence) ---
            if self.think_end_sequence_ids:
                start_idx = find_sublist(input_ids_list, self.think_end_sequence_ids)

                if start_idx != -1:
                    # Mask everything up to and including the *last* token of the sequence
                    mask_until_idx = start_idx + len(self.think_end_sequence_ids)
                    # Mask until the end of the sequence, but not including the last token of the sequence
                    # mask_until_idx = start_idx
                    # print(f"Debug: Found sequence at index {start_idx} in example {i}. Masking until index {mask_until_idx}.") # Optional debug print
                else:
                    # Sequence not found - this indicates an issue
                    print(f"Warning: Think end sequence {self.think_end_sequence_ids} not found in example {i}. Loss calculation might be incorrect. Example start: {self.tokenizer.decode(input_ids_list[:50])}")
                    # Fallback: Mask only the BOS token if present, otherwise mask nothing.
                    if input_ids_list and input_ids_list[0] == self.tokenizer.bos_token_id:
                        labels[i, 0] = -100
                    # Consider masking the entire label sequence if the end tag is crucial
                    # labels[i, :] = -100
                    continue # Skip masking for this example if sequence not found
            else:
                 # Should not happen if initialized correctly, but handle defensively
                 print(f"Warning: think_end_sequence_ids is empty for example {i}. Skipping masking.")
                 continue

            # --- Apply Masking ---
            if mask_until_idx != -1 and mask_until_idx < labels.size(1): # Ensure index is valid
                # Mask everything up to and including the found sequence end index
                labels[i, :mask_until_idx] = -100
            elif mask_until_idx >= labels.size(1):
                 print(f"Warning: Calculated mask index {mask_until_idx} is out of bounds for labels length {labels.size(1)} in example {i}. Masking entire sequence.")
                 labels[i, :] = -100
            # else: handled by the 'continue' above

        # Replace original labels with modified ones
        batch["labels"] = labels
        return batch

class DataHandler:
    """Handles dataset loading, preprocessing, and collation."""

    def __init__(self, tokenizer: AutoTokenizer, max_length: int = config.MAX_INPUT_LENGTH):
        """
        Initializes the DataHandler.

        Args:
            tokenizer: The tokenizer to use for preprocessing.
            max_length: The maximum sequence length for tokenization.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Get token IDs for the sequence indicating the end of the prompt/thoughts
        # This uses THINK_END_SEQUENCE for masking purposes.
        # Ensure add_special_tokens=False is appropriate here. If the sequence itself
        # relies on special tokens being added, adjust accordingly.
        self.think_end_sequence_ids = self.tokenizer.encode(config.THINK_END_SEQUENCE, add_special_tokens=False)

        print(f"DataHandler initialized. Masking End Sequence: '{config.THINK_END_SEQUENCE}', Encoded IDs for Masking: {self.think_end_sequence_ids}") # Clarified purpose
        print(f"DataHandler using Preprocessing End Sequence in text: '{config.TRAINING_THINK_END_SEQUENCE}'") # Added print for the text sequence
        if not self.think_end_sequence_ids:
            print("Warning: Could not encode THINK_END_SEQUENCE for masking. Masking will not work.")


    def load_and_prepare_datasets(self, base_dataset_name: str, base_dataset_config: str, train_json_path: str) -> DatasetDict:
        """Loads the base dataset, replaces the train split, and returns it."""
        print(f"Loading base dataset: {base_dataset_name} ({base_dataset_config})")
        dataset = load_dataset(base_dataset_name, base_dataset_config)
        print(f"Original dataset structure:\n{dataset}")

        print(f"Loading modified training data from: {train_json_path}")
        try:
            raw_train_dataset = load_dataset('json', data_files={'train': train_json_path})['train']
            # Replace the training dataset
            dataset['train'] = raw_train_dataset
            print("Training dataset replaced successfully.")
            print(f"New dataset structure:\n{dataset}")

            # Verify necessary columns exist in all splits used later
            required_cols = ['input', 'output_answer'] # 'input' is the problem, 'output_answer' contains thoughts+answer
            for split in ['train', 'validation', 'test']:
                if split in dataset:
                    if not all(col in dataset[split].features for col in required_cols):
                         print(f"Warning: Split '{split}' is missing one or more required columns: {required_cols}. Features: {list(dataset[split].features.keys())}")
                else:
                     print(f"Warning: Split '{split}' not found in the dataset.")


        except Exception as e:
            print(f"Error loading or replacing training data from {train_json_path}: {e}")
            # Decide how to handle error: raise exception, return original dataset, etc.
            # For now, just print the error and continue with the original dataset if loading failed.
            print("Proceeding with original dataset structure due to error.")

        return dataset


    def _preprocess_function(self, examples):
        """
        Internal preprocessing function to format and tokenize examples.
        Creates the full string: PROMPT + <think>\n + ... + \n</think> + ANSWER + EOS
        Assumes 'output_answer' contains only the final answer string.
        Example 'output_answer': "The final answer is \\boxed{42}"
        """
        texts = [
            config.TRAINING_MATH_PROMPT_START.format(problem=prob) # Includes <think>\n
            + config.TRAINING_THINK_END_SEQUENCE # Use the sequence for text construction
            # + " " # Add a space before the answer for potential tokenization benefits
            + "\n" # Newline before the answer
            + ans
            + self.tokenizer.eos_token
            for prob, ans in zip(examples['input'], examples['output_answer'])
        ]
        model_inputs = self.tokenizer(texts, max_length=self.max_length, truncation=True)
        return model_inputs

    def tokenize_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """Applies the preprocessing function to the dataset."""
        print("Tokenizing dataset...")
        # Keep track of original columns to remove them *after* mapping
        original_columns_per_split = {split: list(dataset[split].features.keys()) for split in dataset.keys()}

        tokenized_dataset = dataset.map(
            self._preprocess_function,
            batched=True,
            # No remove_columns here - keep original columns temporarily
        )
        print("Preprocessing function applied.")

        # Now, remove the original columns from each split, keeping only the tokenizer output
        final_tokenized_dataset = DatasetDict()
        kept_columns = ['input_ids', 'attention_mask'] # Columns generated by tokenizer

        for split, original_cols in original_columns_per_split.items():
            if split in tokenized_dataset:
                cols_to_remove = [col for col in original_cols if col not in kept_columns]
                try:
                    # Check if columns to remove actually exist in the tokenized split before removing
                    current_cols_in_split = tokenized_dataset[split].column_names
                    actual_cols_to_remove = [col for col in cols_to_remove if col in current_cols_in_split]
                    if actual_cols_to_remove:
                         final_tokenized_dataset[split] = tokenized_dataset[split].remove_columns(actual_cols_to_remove)
                         print(f"Removed columns {actual_cols_to_remove} from split '{split}'.")
                    else:
                         final_tokenized_dataset[split] = tokenized_dataset[split] # No columns to remove
                         print(f"No original columns needed removal from split '{split}'.")

                except ValueError as e:
                     print(f"Warning: Could not remove columns from split '{split}'. Error: {e}")
                     print(f"Columns in '{split}' before removal attempt: {tokenized_dataset[split].column_names}")
                     print(f"Columns intended for removal: {cols_to_remove}")
                     # Keep the split as is if removal fails, though this might cause issues later
                     final_tokenized_dataset[split] = tokenized_dataset[split]
            else:
                 print(f"Warning: Split '{split}' not found in tokenized_dataset after mapping.")


        print("Original columns removed after tokenization.")
        if "train" in final_tokenized_dataset and len(final_tokenized_dataset["train"]) > 0:
             print(f"Final tokenized training dataset example (first item keys): {list(final_tokenized_dataset['train'].features.keys())}")
        if "validation" in final_tokenized_dataset and len(final_tokenized_dataset["validation"]) > 0:
             print(f"Final tokenized validation dataset example (first item keys): {list(final_tokenized_dataset['validation'].features.keys())}")

        return final_tokenized_dataset

    def get_data_collator(self) -> DataCollatorForLanguageModeling:
        """Returns the custom data collator for language modeling with prompt/thought masking."""
        print(f"Initializing custom PromptMaskingDataCollator with masking sequence IDs: {self.think_end_sequence_ids} (from '{config.THINK_END_SEQUENCE}')") # Clarified which sequence is used for IDs
        if not self.think_end_sequence_ids:
            print("Error: Cannot initialize PromptMaskingDataCollator because think_end_sequence_ids are missing.")
            # Fallback to default collator to avoid crashing, but training will be incorrect.
            return DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        data_collator = PromptMaskingDataCollator(
            tokenizer=self.tokenizer,
            mlm=False, # Ensure MLM is False for causal LM
            think_end_sequence_ids=self.think_end_sequence_ids # Pass the END sequence IDs for masking
        )
        return data_collator
