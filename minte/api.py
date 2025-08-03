# minte/api.py
"""High-level API for the malaria forecast package."""

import json
import logging
import os
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from .config import Config
from .data import DataModule
from .models import ModelFactory
from .training import HyperparameterOptimizer, Trainer
from .utils import convert_to_json_serializable, set_seed
from .visualization import Visualizer

logger = logging.getLogger(__name__)


class MalariaForecast:
    """High-level API for malaria forecasting."""
    
    def __init__(self, db_path: str, table_name: str = "simulation_results",
                 predictor: str = "prevalence", window_size: int = 7, 
                 device: Optional[str] = None, seed: int = 42):
        """
        Initialize the malaria forecast API.
        
        Args:
            db_path: Path to DuckDB database file
            table_name: Table name inside DuckDB
            predictor: What to predict - 'prevalence' or 'cases'
            window_size: Window size for rolling average
            device: Device to use - 'cuda' or 'cpu'. If None, auto-detect
            seed: Random seed for reproducibility
        """
        self.db_path = db_path
        self.table_name = table_name
        self.predictor = predictor
        self.window_size = window_size
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        
        # Initialize components
        self.config = None
        self.data_module = None
        self.models = {}
        self.trainers = {}
        self.results = {}
        
        # Set random seed
        set_seed(seed)
        
    def run(self, 
            param_limit: Union[str, int] = "all",
            sim_limit: Union[str, int] = "all",
            min_prevalence: float = 0.01,
            use_cyclical_time: bool = False,
            hidden_size: int = 64,
            num_layers: int = 1,
            dropout: float = 0.1,
            lookback: int = 30,
            epochs: int = 64,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-1,
            batch_size: int = 4096,
            patience: int = 16,
            num_workers: int = 0,
            output_dir: str = "results",
            use_existing_split: bool = False,
            split_file: Optional[str] = None,
            run_tuning: bool = False,
            tuning_timeout: int = 86400,
            tuning_trials: int = 32,
            use_tuned_parameters: bool = False,
            tuning_output_dir: str = "results_tuned") -> Dict[str, Any]:
        """
        Run the full malaria forecasting pipeline.
        
        Returns:
            Dictionary containing results for both GRU and LSTM models
        """
        # Create configuration
        self.config = Config(
            db_path=self.db_path,
            table_name=self.table_name,
            window_size=self.window_size,
            param_limit=str(param_limit),
            sim_limit=str(sim_limit),
            min_prevalence=min_prevalence,
            use_cyclical_time=use_cyclical_time,
            predictor=self.predictor,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            lookback=lookback,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            patience=patience,
            num_workers=num_workers,
            device=self.device,
            output_dir=output_dir,
            tuning_output_dir=tuning_output_dir,
            use_existing_split=use_existing_split,
            split_file=split_file,
            run_tuning=run_tuning,
            tuning_timeout=tuning_timeout,
            tuning_trials=tuning_trials,
            use_tuned_parameters=use_tuned_parameters,
            seed=self.seed
        )
        
        # Save configuration
        os.makedirs(self.config.output_dir, exist_ok=True)
        self.config.save(os.path.join(self.config.output_dir, "config.json"))
        logger.info(f"Configuration saved to {os.path.join(self.config.output_dir, 'config.json')}")
        
        # Create data module and prepare data
        logger.info("Preparing data...")
        self.data_module = DataModule(self.config)
        self.data_module.prepare_data()
        
        # Handle hyperparameter tuning
        model_hyperparams = self._handle_hyperparameter_tuning()
        
        # Train and evaluate both models
        for model_type in ["gru", "lstm"]:
            logger.info(f"\n========== Training {model_type.upper()} Model for {self.config.predictor} ==========")
            
            # Get model hyperparameters
            params = model_hyperparams[model_type]
            
            # Train model
            model_results = self._train_model(model_type, params)
            self.results[model_type] = model_results
            
        # Save combined results
        self._save_results()
        
        # Create visualizations
        self._create_visualizations()
        
        logger.info(f"\nExperiment for {self.config.predictor} completed successfully!")
        
        return self.results
        
    def _handle_hyperparameter_tuning(self) -> Dict[str, Dict[str, Any]]:
        """Handle hyperparameter tuning or loading tuned parameters."""
        # Initialize default hyperparameters
        model_hyperparams = {
            "gru": {
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "hidden_size": self.config.hidden_size,
                "num_layers": self.config.num_layers,
                "dropout": self.config.dropout,
                "lookback": self.config.lookback,
                "batch_size": self.config.batch_size
            },
            "lstm": {
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "hidden_size": self.config.hidden_size,
                "num_layers": self.config.num_layers,
                "dropout": self.config.dropout,
                "lookback": self.config.lookback,
                "batch_size": self.config.batch_size
            }
        }
        
        if self.config.run_tuning:
            logger.info("Running hyperparameter optimization for both models")
            hyperparameter_optimizer = HyperparameterOptimizer(self.config, self.data_module)
            best_params = hyperparameter_optimizer.run_optimization()
            
            # Update hyperparameters for both models
            for model_type in ["gru", "lstm"]:
                if model_type in best_params:
                    model_hyperparams[model_type].update(best_params[model_type])
                    model_hyperparams[model_type]["batch_size"] = self.config.batch_size
                    
        elif self.config.use_tuned_parameters:
            logger.info("Attempting to load previously tuned parameters")
            loaded_params = HyperparameterOptimizer.load_best_params(self.config.tuning_output_dir)
            
            if loaded_params:
                # Update parameters for each model if available
                for model_type in ["gru", "lstm"]:
                    if model_type in loaded_params and loaded_params[model_type] is not None:
                        model_hyperparams[model_type].update(loaded_params[model_type])
                        logger.info(f"Loaded parameters for {model_type.upper()}")
                    else:
                        logger.warning(f"No tuned parameters found for {model_type.upper()}. Using default parameters.")
            else:
                logger.warning("No tuned parameters found. Using default parameters for both models.")
                
        return model_hyperparams
        
    def _train_model(self, model_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Train a single model with given parameters."""
        # Create datasets and dataloaders
        datasets = self.data_module.create_datasets(params["lookback"])
        dataloaders = self.data_module.create_dataloaders(datasets, params["batch_size"])
        
        # Initialize model
        output_size = 1
        model = ModelFactory.create_model(
            model_type, 
            self.data_module.input_size, 
            params["hidden_size"], 
            output_size, 
            params["dropout"], 
            params["num_layers"],
            self.config.predictor
        )
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=params["learning_rate"], 
            weight_decay=params["weight_decay"]
        )
        
        # Initialize scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Initialize trainer
        trainer = Trainer(model, self.config, self.config.output_dir, model_type)
        
        # Train model
        model, best_loss, best_epoch = trainer.train(
            dataloaders["train"], dataloaders["val"], optimizer, scheduler
        )
        
        # Store model and trainer
        self.models[model_type] = model
        self.trainers[model_type] = trainer
        
        # Evaluate model on test set
        logger.info(f"\n========== Evaluating {model_type.upper()} Model on Test Set ==========")
        test_metrics, test_preds, test_targets = trainer.evaluate(
            dataloaders["test"], f"{model_type.upper()} Test"
        )
        
        return {
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "test_metrics": test_metrics,
            "test_preds": test_preds,
            "test_targets": test_targets,
            "hyperparameters": params
        }
        
    def _save_results(self) -> None:
        """Save test metrics and results."""
        json_serializable_results = convert_to_json_serializable({
            "predictor": self.config.predictor,
            "gru": {
                "test_metrics": self.results["gru"]["test_metrics"],
                "best_epoch": self.results["gru"]["best_epoch"],
                "best_loss": self.results["gru"]["best_loss"],
                "hyperparameters": self.results["gru"]["hyperparameters"]
            },
            "lstm": {
                "test_metrics": self.results["lstm"]["test_metrics"],
                "best_epoch": self.results["lstm"]["best_epoch"],
                "best_loss": self.results["lstm"]["best_loss"],
                "hyperparameters": self.results["lstm"]["hyperparameters"]
            }
        })

        # Save to JSON file
        with open(os.path.join(self.config.output_dir, "results.json"), 'w') as f:
            json.dump(json_serializable_results, f, indent=4)
            
    def _create_visualizations(self) -> None:
        """Create visualizations for the results."""
        visualizer = Visualizer(self.config, self.data_module)
        
        # Extract model results for visualization
        model_results = {
            "gru": {
                "best_epoch": self.results["gru"]["best_epoch"],
                "best_loss": self.results["gru"]["best_loss"]
            },
            "lstm": {
                "best_epoch": self.results["lstm"]["best_epoch"],
                "best_loss": self.results["lstm"]["best_loss"]
            }
        }
        
        visualizer.plot_training_history(self.config.output_dir, model_results)
        visualizer.plot_model_comparison(self.config.output_dir, self.results)
        visualizer.plot_test_predictions(self.config.output_dir, self.models, self.trainers)
        
    def predict(self, data: np.ndarray, model_type: str = "gru") -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            data: Time series data array of shape (timesteps, features)
            model_type: Which model to use - 'gru' or 'lstm'
            
        Returns:
            Predictions array
        """
        if model_type not in self.trainers:
            raise ValueError(f"Model type '{model_type}' not available. Train the model first.")
            
        return self.trainers[model_type].predict_sequence(data)
        
    def load_models(self, output_dir: str) -> None:
        """
        Load previously trained models from disk.
        
        Args:
            output_dir: Directory containing saved models
        """
        # Load configuration
        config_path = os.path.join(output_dir, "config.json")
        self.config = Config.load(config_path)
        
        # Create data module
        self.data_module = DataModule(self.config)
        
        # Load models
        for model_type in ["gru", "lstm"]:
            model_path = os.path.join(output_dir, f"{model_type}_best.pt")
            if os.path.exists(model_path):
                # Load hyperparameters from results
                results_path = os.path.join(output_dir, "results.json")
                with open(results_path, 'r') as f:
                    results = json.load(f)
                    
                params = results[model_type]["hyperparameters"]
                
                # Create model
                model = ModelFactory.create_model(
                    model_type,
                    self.data_module.input_size,
                    params["hidden_size"],
                    1,  # output_size
                    params["dropout"],
                    params["num_layers"],
                    self.config.predictor
                )
                
                # Load model weights
                checkpoint = torch.load(model_path, map_location=self.config.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                
                # Create trainer
                trainer = Trainer(model, self.config, output_dir, model_type)
                
                self.models[model_type] = model
                self.trainers[model_type] = trainer
                
                logger.info(f"Loaded {model_type.upper()} model from {model_path}")


def run_forecast(**kwargs) -> Dict[str, Any]:
    """
    Convenience function to run malaria forecasting.
    
    This function provides a simple interface for running the full forecasting pipeline.
    
    Args:
        **kwargs: Keyword arguments passed to MalariaForecast.run()
        
    Returns:
        Dictionary containing results for both GRU and LSTM models
    """
    # Extract initialization parameters
    init_params = {
        'db_path': kwargs.pop('db_path'),
        'table_name': kwargs.pop('table_name', 'simulation_results'),
        'predictor': kwargs.pop('predictor', 'prevalence'),
        'window_size': kwargs.pop('window_size', 7),
        'device': kwargs.pop('device', None),
        'seed': kwargs.pop('seed', 42)
    }
    
    # Create forecaster
    forecaster = MalariaForecast(**init_params)
    
    # Run forecast with remaining parameters
    return forecaster.run(**kwargs)