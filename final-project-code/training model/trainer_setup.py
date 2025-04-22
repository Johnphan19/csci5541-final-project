import wandb
import torch
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import Dataset
import config # Import the configuration

class TrainerSetup:
    """Handles WandB initialization, TrainingArguments configuration, and Trainer initialization."""

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, data_collator: DataCollatorForLanguageModeling,
                 train_dataset: Dataset, eval_dataset: Dataset):
        """
        Initializes the TrainerSetup.

        Args:
            model: The model to be trained.
            tokenizer: The tokenizer used.
            data_collator: The data collator.
            train_dataset: The tokenized training dataset.
            eval_dataset: The tokenized evaluation dataset.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.training_args = None
        self.trainer = None
        print("TrainerSetup initialized.")

    def setup_wandb(self):
        """Initializes and logs in to WandB."""
        print("Initializing WandB...")
        try:
            wandb.login() # Ensure you are logged in
            run = wandb.init(
                project=config.WANDB_PROJECT,
                config={
                    "learning_rate": config.LEARNING_RATE,
                    "epochs": config.EPOCHS,
                    "train_batch_size": config.TRAIN_BATCH_SIZE,
                    "eval_batch_size": config.EVAL_BATCH_SIZE,
                    "gradient_accumulation_steps": config.GRADIENT_ACCUMULATION_STEPS,
                    "effective_batch_size": config.TRAIN_BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS,
                    "model_name": config.MODEL_NAME,
                    "dataset_path": config.DATASET_JSON_PATH,
                    "weight_decay": config.WEIGHT_DECAY,
                    "optimizer": "AdamW", # Default for Trainer
                    "output_dir": config.OUTPUT_DIR,
                    "evaluation_steps": config.EVALUATION_STEPS,
                    "save_steps": config.SAVE_STEPS,
                },
                name=config.WANDB_RUN_NAME,
            )
            print("WandB initialized successfully.")
            return run
        except Exception as e:
            print(f"Failed to initialize WandB: {e}")
            print("Proceeding without WandB logging.")
            return None


    def configure_training_args(self) -> TrainingArguments:
        """Configures and returns the TrainingArguments."""
        self.training_args = TrainingArguments(
            output_dir=config.OUTPUT_DIR,
            learning_rate=config.LEARNING_RATE,
            per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
            gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
            num_train_epochs=config.EPOCHS,
            weight_decay=config.WEIGHT_DECAY,
            # evaluation_strategy="steps",
            eval_steps=config.EVALUATION_STEPS,
            save_strategy="steps",
            save_steps=config.SAVE_STEPS,
            load_best_model_at_end=False, # Keep False unless you need the absolute best based on metric
            # metric_for_best_model="eval_loss", # Only relevant if load_best_model_at_end=True
            # greater_is_better=False, # Lower loss is better
            logging_dir=config.LOGGING_DIR,
            logging_steps=10,
            # FP16/BF16 handling: Enable based on device capability
            # bf16=(config.DEVICE.type in ['cuda', 'xpu'] and config.DTYPE_TO_LOAD == torch.bfloat16),
            # fp16=(config.DEVICE.type == 'cuda' and config.DTYPE_TO_LOAD != torch.bfloat16), # Use fp16 if cuda and not bf16
            report_to="wandb" if wandb.run is not None else "none", # Report only if wandb initialized
            gradient_checkpointing=True, # Saves memory
            push_to_hub=False,
            # Required for some models when gradient checkpointing is enabled
            # find_unused_parameters=False # Set to True or False based on model/warnings
        )
        print("Training arguments configured.")
        return self.training_args

    def initialize_trainer(self) -> Trainer:
        """Initializes and returns the Trainer instance."""
        if self.training_args is None:
            print("Error: TrainingArguments not configured. Call configure_training_args() first.")
            return None

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset, # Use the appropriate split (e.g., validation or test)
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            # compute_metrics=compute_metrics, # Add if you have a compute_metrics function
        )
        print("Trainer initialized.")
        return self.trainer

    @staticmethod
    def finish_wandb():
        """Finishes the current WandB run if active."""
        if wandb.run is not None:
            wandb.finish()
            print("WandB run finished.")
        else:
            print("No active WandB run to finish.")
