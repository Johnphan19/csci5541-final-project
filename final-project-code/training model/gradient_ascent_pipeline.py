import os
import gc
import wandb
import torch
from transformers import TrainingArguments, Trainer
from typing import Dict, Any, List, Optional

# Import refactored components
from config_manager import ConfigManager
from model_manager import ModelManager
from dataset_manager import DatasetManager # Standard collator used here
from inference_engine import InferenceEngine # For cleanup
from utils import find_last_sublist # Import helper

# --- Custom Trainer for Gradient Ascent ---
class GradientAscentTrainer(Trainer):
    """Custom Trainer for Gradient Ascent with prompt masking and EOS scaling."""
    def __init__(self, *args, assistant_marker_str: str, eos_loss_scale_factor: float, **kwargs):
        super().__init__(*args, **kwargs)
        # Encode the assistant marker once
        self.assistant_prompt_marker_ids = self.tokenizer.encode(assistant_marker_str, add_special_tokens=False)
        if not self.assistant_prompt_marker_ids:
            print(f"Warning: Could not encode assistant marker '{assistant_marker_str}'. Masking will fail.")
        else:
            print(f"GradientAscentTrainer: Using marker '{assistant_marker_str}' (IDs: {self.assistant_prompt_marker_ids}) for loss masking.")
        self.eos_loss_scale_factor = eos_loss_scale_factor
        print(f"GradientAscentTrainer: Using EOS loss scale factor: {self.eos_loss_scale_factor}")

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss only on the assistant's response part, negate it,
        and scale down the loss magnitude for predicted EOS tokens.
        """
        input_ids = inputs.get("input_ids")
        labels = inputs.get("labels")
        if labels is None:
             raise ValueError("Labels must be provided.")

        masked_labels = labels.clone()

        for i in range(input_ids.size(0)):
            input_ids_list = input_ids[i].tolist()
            mask_until_idx = -1

            if self.assistant_prompt_marker_ids:
                start_idx = find_last_sublist(input_ids_list, self.assistant_prompt_marker_ids)
                if start_idx != -1:
                    mask_until_idx = start_idx + len(self.assistant_prompt_marker_ids)
                else:
                    print(f"Warning: Assistant marker not found in example {i}. Loss calculated on full sequence.")
                    mask_until_idx = 0
            else:
                 print(f"Warning: assistant_prompt_marker_ids is empty. Loss calculated on full sequence.")
                 mask_until_idx = 0

            if mask_until_idx > 0 and mask_until_idx < masked_labels.size(1):
                masked_labels[i, :mask_until_idx] = -100
            elif mask_until_idx >= masked_labels.size(1):
                 print(f"Warning: Mask index {mask_until_idx} out of bounds for labels length {masked_labels.size(1)}. Masking entire sequence.")
                 masked_labels[i, :] = -100

        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        per_token_loss = loss_fct(logits.view(-1, self.model.config.vocab_size), masked_labels.view(-1))
        per_token_loss = per_token_loss.view(masked_labels.size(0), masked_labels.size(1))

        predicted_token_ids = torch.argmax(logits, dim=-1)
        scaling_factors = torch.ones_like(masked_labels, dtype=logits.dtype)
        eos_predicted_mask = (predicted_token_ids == self.tokenizer.eos_token_id)
        scaling_factors[eos_predicted_mask] = self.eos_loss_scale_factor
        scaled_per_token_loss = per_token_loss * scaling_factors

        valid_labels_mask = (masked_labels != -100)
        valid_scaled_losses = scaled_per_token_loss[valid_labels_mask]

        if valid_scaled_losses.numel() > 0:
             mean_scaled_loss = valid_scaled_losses.mean()
        else:
             mean_scaled_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        neg_loss = -mean_scaled_loss
        return (neg_loss, outputs) if return_outputs else neg_loss

# --- Gradient Ascent Pipeline Class ---
class GradientAscentPipeline:
    """Encapsulates the Gradient Ascent training process."""

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
        print("GradientAscentPipeline initialized.")

    def _setup(self):
        """Loads model, tokenizer, dataset, and prepares for training."""
        print("\n--- GA Pipeline Setup ---")
        # 1. Initialize Managers
        self.force_cpu_fp32 = self.config.get('FORCE_CPU_FLOAT32_FOR_TRAINING', False)
        effective_device = torch.device("cpu") if self.force_cpu_fp32 else self.config['DEVICE']
        effective_dtype = torch.float32 if self.force_cpu_fp32 else self.config['DTYPE_TO_LOAD']

        self.model_manager = ModelManager(
            model_name=self.config['MODEL_NAME'],
            device=self.config['DEVICE'], # Pass original default
            dtype=self.config['DTYPE_TO_LOAD'], # Pass original default
            force_cpu_fp32_for_training=self.force_cpu_fp32 # Pass override flag
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

        # 4. Load and Prepare Dataset (Use original train split for GA)
        base_dataset = self.dataset_manager.load_base_dataset(
            dataset_name=self.config['BASE_DATASET_NAME'],
            dataset_config=self.config['BASE_DATASET_CONFIG']
        )
        self.tokenized_dataset = self.dataset_manager.prepare_and_tokenize(
            base_dataset=base_dataset,
            training_type='gradient_ascent', # Specify type
            train_json_path=None # Not needed for GA
        )

        # 5. Get Data Collator (Standard LM collator for GA)
        self.data_collator = self.dataset_manager.get_data_collator(
            training_type='gradient_ascent'
        )

        # 6. Load Model (for training) - ModelManager handles override
        print(f"Loading model for GA training (effective device: {effective_device}, effective dtype: {effective_dtype})...")
        self.model = self.model_manager.load_model(for_training=True)

        # 7. Setup WandB
        self._setup_wandb()

        # 8. Configure Training Arguments
        training_args = self._configure_training_args()

        # 9. Initialize Custom Trainer
        self._initialize_trainer(training_args)

        print("--- GA Setup Complete ---")

    def _setup_wandb(self):
        """Initializes and logs in to WandB."""
        print("Initializing WandB for Gradient Ascent...")
        try:
            wandb.login()
            self.wandb_run = wandb.init(
                project=self.config.get('WANDB_PROJECT', 'NLP_Gradient_Ascent'),
                config=self.config,
                name=self.config.get('WANDB_RUN_NAME', 'default-ga-run'),
            )
            print("WandB initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize WandB: {e}. Proceeding without logging.")
            self.wandb_run = None

    def _configure_training_args(self) -> TrainingArguments:
        """Configures and returns the TrainingArguments for GA."""
        # Determine if running on CPU
        is_cpu_training = self.force_cpu_fp32 or self.config['DEVICE'].type == 'cpu'
        print(f"Configuring GA TrainingArguments. Is CPU training: {is_cpu_training}")

        training_args = TrainingArguments(
            output_dir=self.config['OUTPUT_DIR'],
            learning_rate=self.config['LEARNING_RATE'],
            per_device_train_batch_size=self.config['TRAIN_BATCH_SIZE'],
            per_device_eval_batch_size=self.config['EVAL_BATCH_SIZE'],
            gradient_accumulation_steps=self.config['GRADIENT_ACCUMULATION_STEPS'],
            num_train_epochs=self.config['EPOCHS'],
            weight_decay=self.config['WEIGHT_DECAY'],
            evaluation_strategy="steps",
            eval_steps=self.config['EVALUATION_STEPS'],
            save_strategy="steps",
            save_steps=self.config['SAVE_STEPS'],
            load_best_model_at_end=False, # Don't load best for GA
            logging_dir=self.config['LOGGING_DIR'],
            logging_steps=10,
            # Disable mixed precision if training on CPU
            bf16=False if is_cpu_training else (self.config['DEVICE'].type in ['cuda', 'xpu'] and self.config['DTYPE_TO_LOAD'] == torch.bfloat16),
            fp16=False if is_cpu_training else (self.config['DEVICE'].type == 'cuda' and self.config['DTYPE_TO_LOAD'] != torch.bfloat16),
            report_to="wandb" if self.wandb_run else "none",
            gradient_checkpointing=True,
            push_to_hub=False,
            # Explicitly set no_cuda if forcing CPU, otherwise let Trainer detect
            no_cuda=is_cpu_training,
        )
        print("GA Training arguments configured.")
        if is_cpu_training:
            print("  Mixed precision (bf16/fp16) disabled for CPU training.")
        return training_args

    def _initialize_trainer(self, training_args: TrainingArguments):
        """Initializes the custom GradientAscentTrainer instance."""
        if not self.model or not self.tokenizer or not self.data_collator:
            raise RuntimeError("Model, tokenizer, or data collator not initialized.")
        if 'train' not in self.tokenized_dataset or 'validation' not in self.tokenized_dataset:
             raise RuntimeError("Tokenized dataset missing 'train' or 'validation' split.")

        # Get GA specific params from config
        assistant_marker = self.config.get('ASSISTANT_MARKER_STR', '<｜Assistant｜>')
        eos_scale = self.config.get('EOS_LOSS_SCALE_FACTOR', 0.1)

        self.trainer = GradientAscentTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset['train'],
            eval_dataset=self.tokenized_dataset['validation'],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            # Pass GA specific params
            assistant_marker_str=assistant_marker,
            eos_loss_scale_factor=eos_scale,
        )
        print("GradientAscentTrainer initialized.")

    def _train(self):
        """Runs the training loop."""
        print("\n--- Starting Gradient Ascent Training ---")
        if not self.trainer:
            raise RuntimeError("Trainer not initialized. Cannot start training.")
        try:
            train_result = self.trainer.train()
            print("--- GA Training Finished ---")
            metrics = train_result.metrics
            self.trainer.log_metrics("train", metrics)
            self.trainer.save_metrics("train", metrics)
            return True
        except Exception as e:
            print(f"An error occurred during GA training: {e}")
            return False

    def _evaluate(self):
        """Evaluates the trained model (eval loss should increase)."""
        print("\n--- Evaluating Final GA Model ---")
        if not self.trainer:
            print("Trainer not available. Skipping evaluation.")
            return
        try:
            # Note: eval_loss reported by Trainer.evaluate will be the standard
            # (positive) masked cross-entropy loss, which we expect to increase during GA.
            eval_metrics = self.trainer.evaluate()
            self.trainer.log_metrics("eval", eval_metrics)
            self.trainer.save_metrics("eval", eval_metrics)
            print(f"GA Evaluation metrics (expect high loss): {eval_metrics}")
        except Exception as e:
            print(f"An error occurred during GA evaluation: {e}")

    def _save_model(self):
        """Saves the final model and tokenizer."""
        print("\n--- Saving Final GA Model ---")
        if not self.trainer:
            print("Trainer not available. Skipping model saving.")
            return
        try:
            save_path = self.config['SAVED_MODEL_PATH']
            print(f"Saving GA model and tokenizer to {save_path}...")
            self.trainer.save_model(save_path)
            self.tokenizer.save_pretrained(save_path)
            print("GA Model and tokenizer saved successfully.")
        except Exception as e:
            print(f"An error occurred during GA model saving: {e}")

    def _finish_wandb(self):
        """Finishes the WandB run."""
        if self.wandb_run:
            wandb.finish()
            print("WandB run finished.")

    def cleanup(self):
        """Cleans up resources like model, trainer, and GPU memory."""
        print("\n--- Cleaning Up GA Pipeline Resources ---")
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
        print("GA Pipeline resources cleaned up.")

    def run(self):
        """Executes the full gradient ascent pipeline."""
        training_successful = False
        try:
            self._setup()
            training_successful = self._train()
            if training_successful:
                self._evaluate()
                self._save_model()
            else:
                print("Skipping GA evaluation and saving due to training failure.")
        except Exception as e:
            print(f"An error occurred during the GA pipeline execution: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._finish_wandb()
            self.cleanup()

print("GradientAscentPipeline loaded.")
