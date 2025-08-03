# tests/test_models.py
"""Tests for model components."""

import pytest
import torch
from minte.models import ModelFactory, GRUModel, LSTMModel


def test_model_factory():
    """Test model factory creation."""
    input_size = 10
    hidden_size = 32
    output_size = 1
    dropout = 0.1
    num_layers = 2
    
    # Test GRU creation
    gru = ModelFactory.create_model(
        "gru", input_size, hidden_size, output_size, 
        dropout, num_layers, "prevalence"
    )
    assert isinstance(gru, GRUModel)
    assert gru.hidden_size == hidden_size
    assert gru.num_layers == num_layers
    
    # Test LSTM creation
    lstm = ModelFactory.create_model(
        "lstm", input_size, hidden_size, output_size,
        dropout, num_layers, "cases"
    )
    assert isinstance(lstm, LSTMModel)
    assert lstm.hidden_size == hidden_size
    assert lstm.num_layers == num_layers
    
    # Test invalid model type
    with pytest.raises(ValueError):
        ModelFactory.create_model("invalid", input_size, hidden_size, output_size, dropout)


def test_gru_forward_pass():
    """Test GRU model forward pass."""
    batch_size = 16
    seq_len = 30
    input_size = 10
    hidden_size = 32
    
    model = GRUModel(input_size, hidden_size, 1, 0.1, 2, "prevalence")
    
    # Create dummy input (seq_len, batch_size, input_size)
    x = torch.randn(seq_len, batch_size, input_size)
    
    # Forward pass
    output = model(x)
    
    # Check output shape
    assert output.shape == (seq_len, batch_size, 1)
    
    # Check that prevalence predictions are bounded [0, 1]
    assert torch.all(output >= 0) and torch.all(output <= 1)


def test_lstm_forward_pass():
    """Test LSTM model forward pass."""
    batch_size = 16
    seq_len = 30
    input_size = 10
    hidden_size = 32
    
    model = LSTMModel(input_size, hidden_size, 1, 0.1, 2, "cases")
    
    # Create dummy input
    x = torch.randn(seq_len, batch_size, input_size)
    
    # Forward pass
    output = model(x)
    
    # Check output shape
    assert output.shape == (seq_len, batch_size, 1)
    
    # Check that case predictions are non-negative
    assert torch.all(output >= 0)