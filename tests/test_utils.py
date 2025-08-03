
# tests/test_utils.py
"""Tests for utility functions."""

import pytest
import numpy as np
import torch
import random
from minte.utils import set_seed, convert_to_json_serializable


def test_set_seed():
    """Test random seed setting."""
    # Set seed
    set_seed(123)
    
    # Check Python random
    val1 = random.random()
    set_seed(123)
    val2 = random.random()
    assert val1 == val2
    
    # Check NumPy
    arr1 = np.random.randn(5)
    set_seed(123)
    arr2 = np.random.randn(5)
    assert np.allclose(arr1, arr2)
    
    # Check PyTorch
    tensor1 = torch.randn(5)
    set_seed(123)
    tensor2 = torch.randn(5)
    assert torch.allclose(tensor1, tensor2)


def test_convert_to_json_serializable():
    """Test JSON serialization of NumPy types."""
    # Test various NumPy types
    data = {
        'int32': np.int32(42),
        'float64': np.float64(3.14),
        'array': np.array([1, 2, 3]),
        'nested': {
            'array': np.array([[1, 2], [3, 4]])
        },
        'list': [np.int64(1), np.float32(2.5)],
        'regular': 'string'
    }
    
    result = convert_to_json_serializable(data)
    
    assert isinstance(result['int32'], int)
    assert isinstance(result['float64'], float)
    assert isinstance(result['array'], list)
    assert isinstance(result['nested']['array'], list)
    assert isinstance(result['list'][0], int)
    assert isinstance(result['list'][1], float)
    assert result['regular'] == 'string'
    
    # Should be JSON serializable
    import json
    json_str = json.dumps(result)
    assert isinstance(json_str, str)