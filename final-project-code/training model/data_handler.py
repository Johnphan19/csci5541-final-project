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
    Assumes the input has been formatted using a chat template, and the sequence
    to mask up to (THINK_END_SEQUENCE) is present within the assistant's part.
    """
    def __init__(self, tokenizer: AutoTokenizer, think_end_sequence_ids: List[int], **kwargs):
        super().__init__(tokenizer=tokenizer, **kwargs)
        self.think_end_sequence_ids = think_end_sequence_ids
        print(f"PromptMaskingDataCollator: Initialized. Masking up to the first occurrence of sequence IDs: {self.think_end_sequence_ids} (corresponding to '{config.THINK_END_SEQUENCE}')")
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
                    # print(f"Debug: Found sequence at index {start_idx} in example {i}. Masking until index {mask_until_idx}.") # Optional debug print
                else:
                    # Sequence not found - this indicates an issue with data or sequence
                    print(f"Warning: Think end sequence {self.think_end_sequence_ids} ('{config.THINK_END_SEQUENCE}') not found in example {i}. Loss calculation might be incorrect. Example start: {self.tokenizer.decode(input_ids_list[:50])}")
                    # Fallback: Mask only the padding tokens? Or mask nothing? Masking nothing might be safer.
                    # Let's mask nothing specific here, relying on padding mask (-100)
                    # labels[i, :] = -100 # Mask entire sequence if end tag is crucial and missing
                    continue # Skip specific masking for this example if sequence not found
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
            # raise ValueError("Tokenizer must have a chat_template for this DataHandler.")

        # Get token IDs for the sequence indicating the end of the prompt/thoughts
        # This uses THINK_END_SEQUENCE for masking purposes.
        self.think_end_sequence_ids = self.tokenizer.encode(config.THINK_END_SEQUENCE, add_special_tokens=False)

        print(f"DataHandler initialized. Using tokenizer chat template for formatting.")
        print(f"Masking End Sequence: '{config.THINK_END_SEQUENCE}', Encoded IDs for Masking: {self.think_end_sequence_ids}")
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
        Internal preprocessing function to format and tokenize examples using the chat template.
        Assumes 'input' is the problem and 'output_answer' contains the full assistant response
        including the <think>...</think> block and the final answer.
        """
        batch_messages = []
        # Assuming 'input' and 'output_answer' columns exist in the examples
        required_cols = ['input', 'output_answer']
        if not all(col in examples for col in required_cols):
             print(f"Error: Preprocessing function missing required columns: {required_cols}. Available: {list(examples.keys())}")
             # Return empty dict or raise error? Let's return something that map can handle.
             # Need to ensure the output structure matches tokenizer output for consistency.
             # Returning empty lists might cause issues later. Best to ensure data is correct upstream.
             raise ValueError(f"Missing required columns {required_cols} in dataset split.")

        for problem, assistant_response in zip(examples['input'], examples['output_answer']):
            # Construct the conversation structure for the chat template
            # Ensure the assistant_response from the dataset includes the <think>...</think> block
            # and the final answer for the masking logic to work.
            messages = [
                {"role": "user", "content": f"Please reason step by step, and put your final answer within \\boxed{{}}.\n{problem}"},
                {"role": "assistant", "content": assistant_response } # EOS is typically added by tokenizer/template or later
            ]
            batch_messages.append(messages)

        # Format the messages using the chat template (returns strings)
        # We add EOS manually before tokenizing if the template doesn't handle it for training examples.
        # Check tokenizer docs/behavior for specific model. Assuming manual EOS addition is needed here.
        formatted_texts = [
            self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in batch_messages
        ]
        formatted_texts_with_eos = [text + self.tokenizer.eos_token for text in formatted_texts]

        # Tokenize the formatted strings
        model_inputs = self.tokenizer(
            formatted_texts_with_eos,
            max_length=self.max_length,
            truncation=True,
            padding=False, # Collator will handle padding
        )
        return model_inputs

    def tokenize_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """Applies the preprocessing function (using chat template) to the dataset."""
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
        print(f"Initializing custom PromptMaskingDataCollator with masking sequence IDs: {self.think_end_sequence_ids} (from '{config.THINK_END_SEQUENCE}')")
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
