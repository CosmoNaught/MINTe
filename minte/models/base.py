# minte/models/base.py
"""Base model factory for creating different model types."""

import torch.nn as nn


class ModelFactory:
    """Factory class to create different types of models."""
    
    @staticmethod
    def create_model(model_type: str, input_size: int, hidden_size: int, output_size: int, 
                     dropout_prob: float, num_layers: int = 1, predictor: str = "prevalence") -> nn.Module:
        """Create a model of the specified type."""
        from .gru import GRUModel
        from .lstm import LSTMModel
        
        if model_type.lower() == "gru":
            return GRUModel(input_size, hidden_size, output_size, dropout_prob, num_layers, predictor)
        elif model_type.lower() == "lstm":
            return LSTMModel(input_size, hidden_size, output_size, dropout_prob, num_layers, predictor)
        else:
            raise ValueError(f"Unknown model type: {model_type}")