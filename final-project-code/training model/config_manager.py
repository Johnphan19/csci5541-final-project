import os
import importlib
import config as base_config_module # Import the base config file
from typing import Dict, Any, Optional

class ConfigManager:
    """Manages loading and merging configurations for training runs."""

    def __init__(self):
        """Initializes the ConfigManager by loading the base configuration."""
        self.base_config = self._load_base_config()
        print("ConfigManager initialized with base configuration.")

    def _load_base_config(self) -> Dict[str, Any]:
        """Loads attributes from the base config module."""
        # Reload the module in case it was changed
        importlib.reload(base_config_module)
        config_dict = {
            attr: getattr(base_config_module, attr)
            for attr in dir(base_config_module)
            if not attr.startswith("__") and not callable(getattr(base_config_module, attr))
        }
        return config_dict

    def get_base_config(self) -> Dict[str, Any]:
        """Returns a copy of the base configuration."""
        return self.base_config.copy()

    def get_base_value(self, key: str, default: Any = None) -> Any:
        """Gets a specific value from the base config."""
        return self.base_config.get(key, default)

    def get_config(self, experiment_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Creates a run-specific configuration by merging base config with experiment parameters.
        Also derives necessary paths based on the experiment name.

        Args:
            experiment_params: Dictionary of parameters specific to this run, overriding base config.

        Returns:
            A dictionary containing the final configuration for the run.
        """
        run_config = self.get_base_config()

        if experiment_params:
            # Override base config with experiment-specific parameters
            run_config.update(experiment_params)

        # --- Derive Paths and Names ---
        experiment_name = run_config.get("experiment_name", "default_experiment")
        training_type = run_config.get("training_type", "unknown_type") # e.g., 'finetune', 'gradient_ascent'

        # Ensure base output directory exists
        base_output_dir = run_config.get("DEFAULT_OUTPUT_BASE_DIR", "training_outputs")
        os.makedirs(base_output_dir, exist_ok=True)

        # Create experiment-specific output directory
        experiment_output_dir = os.path.join(base_output_dir, f"{training_type}_{experiment_name}")
        run_config["OUTPUT_DIR"] = experiment_output_dir
        os.makedirs(experiment_output_dir, exist_ok=True)

        # Define paths within the experiment directory
        run_config["LOGGING_DIR"] = os.path.join(experiment_output_dir, "logs")
        run_config["SAVED_MODEL_PATH"] = os.path.join(experiment_output_dir, "final_model")
        os.makedirs(run_config["LOGGING_DIR"], exist_ok=True)
        # Don't create final_model dir here, Trainer/save will do it

        # --- WandB Run Name ---
        # Create a more informative default WandB run name if not provided
        if "WANDB_RUN_NAME" not in run_config:
             model_short_name = run_config.get("MODEL_NAME", "unknown_model").split('/')[-1]
             lr = run_config.get("LEARNING_RATE", "unk_lr")
             epochs = run_config.get("EPOCHS", "unk_ep")
             run_config["WANDB_RUN_NAME"] = f"{training_type}-{model_short_name}-{experiment_name}-lr{lr}-ep{epochs}"

        # --- Dataset Path ---
        # Use default if not specified in experiment params
        if "dataset_json_path" not in run_config or run_config["dataset_json_path"] is None:
             run_config["dataset_json_path"] = run_config.get("DEFAULT_DATASET_JSON_PATH")

        # --- Max Length ---
        # Ensure MAX_INPUT_LENGTH is present (can be None)
        if "MAX_INPUT_LENGTH" not in run_config:
            run_config["MAX_INPUT_LENGTH"] = None # Default to None if missing
        # Ensure fallback is present
        if "DEFAULT_FALLBACK_MAX_LENGTH" not in run_config:
            run_config["DEFAULT_FALLBACK_MAX_LENGTH"] = 4096 # Hardcoded fallback if missing in base

        # --- Compile Flag ---
        if "COMPILE_MODEL_FOR_EVALUATION" not in run_config:
            run_config["COMPILE_MODEL_FOR_EVALUATION"] = False # Default if missing

        print(f"Generated run config for experiment: {experiment_name}")
        print(f"  Output Dir: {run_config['OUTPUT_DIR']}")
        print(f"  WandB Run Name: {run_config['WANDB_RUN_NAME']}")
        print(f"  Dataset JSON: {run_config['dataset_json_path']}")
        print(f"  Max Input Length (Config): {run_config['MAX_INPUT_LENGTH']}")
        print(f"  Compile Model for Eval: {run_config['COMPILE_MODEL_FOR_EVALUATION']}")

        return run_config

print("ConfigManager loaded.")
