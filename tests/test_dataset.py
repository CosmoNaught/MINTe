# tests/test_dataset.py
"""Tests for dataset components."""

import pytest
import numpy as np
import torch
from minte.data.dataset import TimeSeriesDataset, collate_fn


def test_time_series_dataset():
    """Test TimeSeriesDataset creation and indexing."""
    # Create dummy data
    groups = [
        {
            'time_series': np.random.randn(100, 10).astype(np.float32),
            'targets': np.random.rand(100).astype(np.float32),
            'length': 100,
            'param_sim_id': (0, 0)
        },
        {
            'time_series': np.random.randn(80, 10).astype(np.float32),
            'targets': np.random.rand(80).astype(np.float32),
            'length': 80,
            'param_sim_id': (1, 0)
        }
    ]
    
    lookback = 30
    dataset = TimeSeriesDataset(groups, lookback=lookback)
    
    # Check dataset length
    expected_length = (100 - lookback + 1) + (80 - lookback + 1)
    assert len(dataset) == expected_length
    
    # Check sample
    x, y = dataset[0]
    assert x.shape == (lookback, 10)
    assert y.shape == (lookback,)
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)


def test_collate_fn():
    """Test custom collate function."""
    batch_size = 4
    seq_len = 30
    features = 10
    
    # Create dummy batch
    batch = []
    for _ in range(batch_size):
        x = torch.randn(seq_len, features)
        y = torch.randn(seq_len)
        batch.append((x, y))
    
    # Collate
    X_batch, Y_batch = collate_fn(batch)
    
    # Check shapes (time, batch, features) and (time, batch)
    assert X_batch.shape == (seq_len, batch_size, features)
    assert Y_batch.shape == (seq_len, batch_size)