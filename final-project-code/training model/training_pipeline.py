import os
import gc
import wandb
import torch
from transformers import TrainingArguments, Trainer
from typing import Dict, Any

# Import refactored components
from config_manager import ConfigManager
from model_manager import ModelManager
from dataset_manager import DatasetManager, PromptMaskingDataCollator
from inference_engine import InferenceEngine # For cleanup

class TrainingPipeline:
    """Encapsulates the standard fine-tuning process."""

    def __init__(self, run_config: Dict[str, Any]):
        """
        Initializes the pipeline with a specific run configuration.

        Args:
            run_config: Dictionary containing all configuration parameters for the run.
        """
        self.config = run_config
        self.model_manager = None
        self.dataset_manager = None
        self.tokenizer = None
        self.model = None
        self.tokenized_dataset = None
        self.data_collator = None
        self.trainer = None
        self.wandb_run = None
        self.force_cpu_fp32 = self.config.get('FORCE_CPU_FLOAT32_FOR_TRAINING', False)
        print("TrainingPipeline initialized.")

    def _setup(self):
        """Loads model, tokenizer, dataset, and prepares for training."""
        print("\n--- Pipeline Setup ---")
        # 1. Initialize Managers
        effective_device = torch.device("cpu") if self.force_cpu_fp32 else self.config['DEVICE']
        effective_dtype = torch.float32 if self.force_cpu_fp32 else self.config['DTYPE_TO_LOAD']

        self.model_manager = ModelManager(
            model_name=self.config['MODEL_NAME'],
            device=self.config['DEVICE'], # Pass the original default device
            dtype=self.config['DTYPE_TO_LOAD'], # Pass the original default dtype
            force_cpu_fp32_for_training=self.force_cpu_fp32 # Pass the override flag
        )

        # 2. Load Tokenizer
        self.tokenizer = self.model_manager.load_tokenizer()
        print(f"Tokenizer chat template:\n{self.tokenizer.chat_template}")

        # 3. Initialize Dataset Manager - Pass max length config values
        self.dataset_manager = DatasetManager(
            tokenizer=self.tokenizer,
            config_max_length=self.config['MAX_INPUT_LENGTH'],
            fallback_max_length=self.config['DEFAULT_FALLBACK_MAX_LENGTH']
        )

        # 4. Load and Prepare Dataset
        base_dataset = self.dataset_manager.load_base_dataset(
            dataset_name=self.config['BASE_DATASET_NAME'],
            dataset_config=self.config['BASE_DATASET_CONFIG']
        )
        self.tokenized_dataset = self.dataset_manager.prepare_and_tokenize(
            base_dataset=base_dataset,
            training_type='finetune', # Specify type
            train_json_path=self.config['dataset_json_path']
        )

        # 5. Get Data Collator
        self.data_collator = self.dataset_manager.get_data_collator(
            training_type='finetune',
            assistant_marker_style='no_think' # Or 'think' based on desired masking
        )

        # 6. Load Model (for training) - ModelManager now handles the CPU/FP32 override internally
        print(f"Loading model for training (effective device: {effective_device}, effective dtype: {effective_dtype})...")
        self.model = self.model_manager.load_model(for_training=True)

        # 7. Setup WandB
        self._setup_wandb()

        # 8. Configure Training Arguments
        training_args = self._configure_training_args()

        # 9. Initialize Trainer
        self._initialize_trainer(training_args)

        print("--- Setup Complete ---")


    def _setup_wandb(self):
        """Initializes and logs in to WandB."""
        print("Initializing WandB...")
        try:
            wandb.login() # Ensure you are logged in
            self.wandb_run = wandb.init(
                project=self.config.get('WANDB_PROJECT', 'NLP_FineTuning'),
                config=self.config, # Log the entire run config
                name=self.config.get('WANDB_RUN_NAME', 'default-finetune-run'),
            )
            print("WandB initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize WandB: {e}")
            print("Proceeding without WandB logging.")
            self.wandb_run = None


    def _configure_training_args(self) -> TrainingArguments:
        """Configures and returns the TrainingArguments."""
        # Determine if running on CPU
        is_cpu_training = self.force_cpu_fp32 or self.config['DEVICE'].type == 'cpu'
        print(f"Configuring TrainingArguments. Is CPU training: {is_cpu_training}")

        training_args = TrainingArguments(
            output_dir=self.config['OUTPUT_DIR'],
            learning_rate=self.config['LEARNING_RATE'],
            per_device_train_batch_size=self.config['TRAIN_BATCH_SIZE'],
            per_device_eval_batch_size=self.config['EVAL_BATCH_SIZE'],
            gradient_accumulation_steps=self.config['GRADIENT_ACCUMULATION_STEPS'],
            num_train_epochs=self.config['EPOCHS'],
            weight_decay=self.config['WEIGHT_DECAY'],
            eval_strategy="steps",
            eval_steps=self.config['EVALUATION_STEPS'],
            save_strategy="steps",
            save_steps=self.config['SAVE_STEPS'],
            load_best_model_at_end=False,
            logging_dir=self.config['LOGGING_DIR'],
            logging_steps=10,
            # Disable mixed precision if training on CPU
            bf16=False if is_cpu_training else (self.config['DEVICE'].type in ['cuda', 'xpu'] and self.config['DTYPE_TO_LOAD'] == torch.bfloat16),
            fp16=False if is_cpu_training else (self.config['DEVICE'].type == 'cuda' and self.config['DTYPE_TO_LOAD'] != torch.bfloat16),
            report_to="wandb" if self.wandb_run else "none",
            gradient_checkpointing=True, # Saves memory, adjust if causing issues
            push_to_hub=False,
            # Explicitly set no_cuda if forcing CPU, otherwise let Trainer detect
            no_cuda=is_cpu_training,
        )
        print("Training arguments configured.")
        if is_cpu_training:
            print("  Mixed precision (bf16/fp16) disabled for CPU training.")
        return training_args

    def _initialize_trainer(self, training_args: TrainingArguments):
        """Initializes the Trainer instance."""
        if not self.model or not self.tokenizer or not self.data_collator:
            raise RuntimeError("Model, tokenizer, or data collator not initialized.")
        if 'train' not in self.tokenized_dataset or 'validation' not in self.tokenized_dataset:
             raise RuntimeError("Tokenized dataset missing 'train' or 'validation' split.")

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset['train'],
            eval_dataset=self.tokenized_dataset['validation'],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            # compute_metrics=compute_metrics, # Add if needed
        )
        print("Trainer initialized.")

    def _train(self):
        """Runs the training loop."""
        print("\n--- Starting Training ---")
        if not self.trainer:
            raise RuntimeError("Trainer not initialized. Cannot start training.")
        try:
            train_result = self.trainer.train()
            print("--- Training Finished ---")
            # Log metrics
            metrics = train_result.metrics
            self.trainer.log_metrics("train", metrics)
            self.trainer.save_metrics("train", metrics)
            return True # Indicate success
        except Exception as e:
            print(f"An error occurred during training: {e}")
            return False # Indicate failure

    def _evaluate(self):
        """Evaluates the trained model."""
        print("\n--- Evaluating Final Model ---")
        if not self.trainer:
            print("Trainer not available. Skipping evaluation.")
            return
        try:
            eval_metrics = self.trainer.evaluate()
            self.trainer.log_metrics("eval", eval_metrics)
            self.trainer.save_metrics("eval", eval_metrics)
            print(f"Evaluation metrics: {eval_metrics}")
        except Exception as e:
            print(f"An error occurred during evaluation: {e}")

    def _save_model(self):
        """Saves the final model and tokenizer."""
        print("\n--- Saving Final Model ---")
        if not self.trainer:
            print("Trainer not available. Skipping model saving.")
            return
        try:
            save_path = self.config['SAVED_MODEL_PATH']
            print(f"Saving model and tokenizer to {save_path}...")
            self.trainer.save_model(save_path)
            self.tokenizer.save_pretrained(save_path)
            print("Model and tokenizer saved successfully.")
        except Exception as e:
            print(f"An error occurred during model saving: {e}")

    def _finish_wandb(self):
        """Finishes the WandB run."""
        if self.wandb_run:
            wandb.finish()
            print("WandB run finished.")

    def cleanup(self):
        """Cleans up resources like model, trainer, and GPU memory."""
        print("\n--- Cleaning Up Pipeline Resources ---")
        del self.model
        del self.trainer
        del self.tokenizer
        del self.dataset_manager
        del self.model_manager
        del self.tokenized_dataset
        del self.data_collator
        self.model = None
        self.trainer = None
        # ... reset other attributes ...
        gc.collect()
        InferenceEngine.cleanup_memory()
        print("Pipeline resources cleaned up.")

    def run(self):
        """Executes the full training pipeline."""
        training_successful = False
        try:
            self._setup()
            training_successful = self._train()
            if training_successful:
                self._evaluate()
                self._save_model()
            else:
                print("Skipping evaluation and saving due to training failure.")
        except Exception as e:
            print(f"An error occurred during the pipeline execution: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._finish_wandb()
            self.cleanup()

print("TrainingPipeline loaded.")
