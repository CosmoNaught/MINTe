# minte/config.py
"""Configuration management for the malaria forecast package."""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import torch


@dataclass
class Config:
    """Configuration class to store all parameters."""
    
    # Data parameters
    db_path: str
    table_name: str = "simulation_results"
    window_size: int = 7
    param_limit: str = "all"
    sim_limit: str = "all"
    min_prevalence: float = 0.01
    use_cyclical_time: bool = False
    predictor: str = "prevalence"  # "prevalence" or "cases"
    
    # Model parameters
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.1
    lookback: int = 30
    
    # Training parameters
    epochs: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-1
    batch_attensize: int = 4096
    patience: int = 16
    num_workers: int = 0
    device: Optional[str] = None
    
    # File paths
    output_dir: str = "results"
    tuning_output_dir: str = "results_tuned"
    use_existing_split: bool = False
    split_file: Optional[str] = None
    
    # Hyperparameter tuning
    run_tuning: bool = False
    tuning_timeout: int = 86400  # 24 hours
    tuning_trials: int = 32
    use_tuned_parameters: bool = False
    
    # Random seed
    seed: int = 42
    
    # Computed fields
    base_output_dir: str = field(init=False)
    base_tuning_output_dir: str = field(init=False)
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Save base directories
        self.base_output_dir = self.output_dir
        self.base_tuning_output_dir = self.tuning_output_dir
        
        # Create predictor-specific output directories
        self.output_dir = os.path.join(self.predictor, self.base_output_dir)
        self.tuning_output_dir = os.path.join(self.predictor, self.base_tuning_output_dir)
        
        # Set device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        # Set split file path
        if self.split_file is None:
            self.split_file = os.path.join(self.output_dir, "train_val_test_split.csv")
            
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        if self.run_tuning or self.use_tuned_parameters:
            os.makedirs(self.tuning_output_dir, exist_ok=True)
            
    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        config_dict = asdict(self)
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)
            
    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary."""
        return cls(**config_dict)