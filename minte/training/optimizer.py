# minte/training/optimizer.py
"""Hyperparameter optimization using Optuna."""

import gc
import json
import logging
import os
import time
from typing import Any, Dict, Optional

import optuna
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader

from ..config import Config
from ..data import DataModule, collate_fn
from ..models import ModelFactory
from ..utils import convert_to_json_serializable
from .trainer import Trainer

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """Class for hyperparameter optimization using Optuna."""
    
    def __init__(self, config: Config, data_module: DataModule):
        """Initialize hyperparameter optimizer with configuration and data module."""
        self.config = config
        self.data_module = data_module
        
    def objective_for_model(self, trial: optuna.Trial, model_type: str) -> float:
        """Objective function for Optuna optimization with enhanced resource management."""
        # Sample hyperparameters
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
        batch_size = self.config.batch_size
        hidden_size = trial.suggest_categorical("hidden_size", [128, 256, 512])
        num_layers = trial.suggest_int("num_layers", 1, 4)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        lookback = trial.suggest_categorical("lookback", [13, 26, 52, 78]) 
        
        # Set device
        device = torch.device(self.config.device)
            
        # Print hyperparameters for this trial
        logger.info(f"\nTrial #{trial.number} for {model_type.upper()} with hyperparameters:")
        logger.info(f"    Learning Rate: {lr:.6f}")
        logger.info(f"    Hidden Size: {hidden_size}")
        logger.info(f"    Number of Layers: {num_layers}")
        logger.info(f"    Dropout: {dropout:.2f}")
        logger.info(f"    Lookback: {lookback}")
        
        # Define output size
        output_size = 1
        
        datasets = None
        train_loader = None
        val_loader = None
        model = None
        optimizer = None
        scheduler = None
        trainer = None
        
        try:
            # Build datasets and create data loaders with current lookback
            datasets = self.data_module.create_datasets(lookback)
            train_dataset = datasets["train"]
            val_dataset = datasets["val"]
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=self.config.num_workers,
                pin_memory=True,
                persistent_workers=True if self.config.num_workers > 0 else False,
                prefetch_factor=4 if self.config.num_workers > 0 else None
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=self.config.num_workers,
                pin_memory=True,
                persistent_workers=True if self.config.num_workers > 0 else False,
                prefetch_factor=4 if self.config.num_workers > 0 else None
            )
            
            # Initialize model with predictor type
            model = ModelFactory.create_model(
                model_type, 
                self.data_module.input_size, 
                hidden_size, 
                output_size, 
                dropout, 
                num_layers,
                self.config.predictor
            ).to(device)
            
            # Initialize optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            
            # Initialize scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )
            
            # Create temporary directory for trial outputs
            trial_dir = os.path.join(self.config.tuning_output_dir, f"{model_type}_trial_{trial.number}")
            os.makedirs(trial_dir, exist_ok=True)
            
            # Set the number of epochs for hyperparameter tuning to be shorter than the default
            tuning_epochs = min(self.config.epochs, 32)
            
            # Train model
            trainer = Trainer(model, self.config, trial_dir, model_type)
            model, best_val_loss, best_epoch = trainer.train(
                train_loader, val_loader, optimizer, scheduler, tuning_epochs
            )
            
            # Save trial results
            trial_results = {
                'trial_number': trial.number,
                'model_type': model_type,
                'hyperparameters': {
                    'learning_rate': lr,
                    'weight_decay': weight_decay,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'dropout': dropout,
                    'lookback': lookback
                },
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch
            }
            
            with open(os.path.join(trial_dir, f"{model_type}_trial_results.json"), 'w') as f:
                json.dump(convert_to_json_serializable(trial_results), f, indent=4)
            
            # Pre-cleanup before returning
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return best_val_loss
        
        except (RuntimeError, ValueError, torch.cuda.OutOfMemoryError) as e:
            # Handle errors (like OOM)
            logger.error(f"Trial failed due to: {str(e)}")
            return float('inf')  # Return a bad score
        
        finally:
            # Comprehensive cleanup to prevent resource leaks
            # Close and clean up DataLoaders first
            if train_loader is not None:
                try:
                    train_loader._iterator = None
                    for w in getattr(train_loader, '_workers', []):
                        if hasattr(w, 'is_alive') and w.is_alive():
                            w.terminate()
                except:
                    pass
            
            if val_loader is not None:
                try:
                    val_loader._iterator = None
                    for w in getattr(val_loader, '_workers', []):
                        if hasattr(w, 'is_alive') and w.is_alive():
                            w.terminate()
                except:
                    pass
            
            # Explicit deletion of objects
            del train_loader
            del val_loader
            del model
            del optimizer
            del scheduler
            del trainer
            del datasets
            
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Monitor resources periodically
            if trial.number % 5 == 0:  # Every 5 trials
                try:
                    import resource
                    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
                    logger.info(f"Resource check - File descriptor limit: {soft_limit}/{hard_limit}")
                except:
                    pass
            
    def run_optimization(self) -> Dict[str, Dict[str, Any]]:
        """Run hyperparameter optimization for both GRU and LSTM models."""
        logger.info("Starting hyperparameter optimization with Optuna for both GRU and LSTM models")
        
        # Dictionary to store best parameters for each model
        best_params = {}
        
        # Run optimization for each model type separately
        for model_type in ["gru", "lstm"]:
            logger.info(f"\nStarting optimization for {model_type.upper()} model")
            
            # Create study name based on dataset and model configuration
            study_name = f"{model_type}_{self.config.predictor}_optimization_{int(time.time())}"
            
            # Create sampler with seed for reproducibility
            sampler = TPESampler(seed=self.config.seed)
            
            # Create pruner to terminate unpromising trials early
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
            
            # Create Optuna study with in-memory storage
            study = optuna.create_study(
                study_name=study_name,
                direction="minimize",
                sampler=sampler,
                pruner=pruner
            )
            
            # Run optimization
            study.optimize(
                lambda trial: self.objective_for_model(trial, model_type),
                n_trials=self.config.tuning_trials,
                timeout=self.config.tuning_timeout // 2,
                gc_after_trial=True,
                show_progress_bar=True,
                n_jobs=4  # Run multiple trials in parallel (but don't overload GPU)
            )
            
            # Get best parameters
            model_best_params = study.best_params
            model_best_value = study.best_value
            logger.info(f"\nBest hyperparameters for {model_type.upper()}:")
            for param, value in model_best_params.items():
                logger.info(f"    {param}: {value}")
            logger.info(f"Best validation loss: {model_best_value:.6f}")
            
            # Save best parameters and full study results
            model_tuning_results = {
                'model_type': model_type,
                'predictor': self.config.predictor,
                'best_params': model_best_params,
                'best_validation_loss': model_best_value,
                'study_name': study_name,
                'n_trials': len(study.trials),
                'optimization_time': time.time(),
                'trials': [
                    {
                        'number': t.number,
                        'params': t.params,
                        'value': t.value,
                        'state': str(t.state),
                        'datetime_start': str(t.datetime_start),
                        'datetime_complete': str(t.datetime_complete)
                    }
                    for t in study.trials
                ]
            }
            
            # Store best parameters for this model
            best_params[model_type] = model_best_params
            
            # Create model-specific directory
            model_tuning_dir = os.path.join(self.config.tuning_output_dir, model_type)
            os.makedirs(model_tuning_dir, exist_ok=True)
            
            # Save results to JSON file
            model_best_params_path = os.path.join(model_tuning_dir, "best_params.json")
            with open(model_best_params_path, 'w') as f:
                json.dump(convert_to_json_serializable(model_tuning_results), f, indent=4)
            logger.info(f"Best hyperparameters for {model_type.upper()} saved to {model_best_params_path}")
            
            # Plot optimization history
            try:
                history_fig = optuna.visualization.plot_optimization_history(study)
                history_fig.write_image(os.path.join(model_tuning_dir, "optimization_history.png"))
                param_importances = optuna.visualization.plot_param_importances(study)
                param_importances.write_image(os.path.join(model_tuning_dir, "param_importances.png"))
                parallel_coordinate = optuna.visualization.plot_parallel_coordinate(study)
                parallel_coordinate.write_image(os.path.join(model_tuning_dir, "parallel_coordinate.png"))
            except Exception as e:
                logger.warning(f"Could not generate visualization plots: {str(e)}")
        
        # Save combined best parameters
        combined_best_params_path = os.path.join(self.config.tuning_output_dir, "best_params.json")
        with open(combined_best_params_path, 'w') as f:
            json.dump(convert_to_json_serializable({"gru": best_params["gru"], "lstm": best_params["lstm"]}), f, indent=4)
        logger.info(f"Combined best hyperparameters saved to {combined_best_params_path}")
        
        return best_params

    @staticmethod
    def load_best_params(tuning_output_dir: str) -> Optional[Dict[str, Dict[str, Any]]]:
        """Load best hyperparameters from previous tuning runs."""
        best_params = {}
        
        # Check for combined best params file first
        combined_params_path = os.path.join(tuning_output_dir, "best_params.json")
        if os.path.exists(combined_params_path):
            with open(combined_params_path, 'r') as f:
                combined_params = json.load(f)
                if "gru" in combined_params and "lstm" in combined_params:
                    return combined_params
        
        # If combined file doesn't exist or is incomplete, try model-specific files
        for model_type in ["gru", "lstm"]:
            model_params_path = os.path.join(tuning_output_dir, model_type, "best_params.json")
            if os.path.exists(model_params_path):
                with open(model_params_path, 'r') as f:
                    tuning_results = json.load(f)
                    best_params[model_type] = tuning_results.get('best_params', None)
            else:
                logger.warning(f"No tuned parameters found for {model_type.upper()} model")
                best_params[model_type] = None
        
        # Return the parameters we found, or None if we didn't find any
        if best_params.get("gru") is not None or best_params.get("lstm") is not None:
            return best_params
        else:
            return None