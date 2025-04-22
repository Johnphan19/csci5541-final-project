from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
import config # Import the configuration

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
        print("DataHandler initialized.")

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
            required_cols = ['input', 'output_answer']
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
        """Internal preprocessing function to format and tokenize examples."""
        texts = [
            f"Problem:\n{prob}\n\nSolution:\n{ans}{self.tokenizer.eos_token}"
            for prob, ans in zip(examples['input'], examples['output_answer'])
        ]
        model_inputs = self.tokenizer(texts, max_length=self.max_length, truncation=True)
        return model_inputs

    def tokenize_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """Applies the preprocessing function to the dataset."""
        print("Tokenizing dataset...")
        # Determine columns to remove - typically all original columns from one of the splits
        remove_cols = dataset["test"].column_names if "test" in dataset else (dataset["train"].column_names if "train" in dataset else [])
        if not remove_cols:
             print("Warning: Could not determine columns to remove during tokenization.")

        tokenized_dataset = dataset.map(
            self._preprocess_function,
            batched=True,
            remove_columns=remove_cols
        )
        print("Tokenization complete.")
        if "train" in tokenized_dataset and len(tokenized_dataset["train"]) > 0:
             print(f"Tokenized training dataset example (first item keys): {tokenized_dataset['train'][0].keys()}")
        if "validation" in tokenized_dataset and len(tokenized_dataset["validation"]) > 0:
             print(f"Tokenized validation dataset example (first item keys): {tokenized_dataset['validation'][0].keys()}")

        return tokenized_dataset

    def get_data_collator(self) -> DataCollatorForLanguageModeling:
        """Returns a data collator for language modeling."""
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        print("Data collator initialized.")
        return data_collator
