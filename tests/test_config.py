# tests/test_config.py
"""Tests for configuration module."""

import pytest
import tempfile
import json
import os
from minte.config import Config


def test_config_creation():
    """Test basic configuration creation."""
    config = Config(
        db_path="/path/to/db.duckdb",
        table_name="test_table",
        predictor="prevalence"
    )
    
    assert config.db_path == "/path/to/db.duckdb"
    assert config.table_name == "test_table"
    assert config.predictor == "prevalence"
    assert config.window_size == 7  # default
    assert config.seed == 42  # default


def test_config_predictor_directories():
    """Test that predictor-specific directories are created."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(
            db_path="/path/to/db.duckdb",
            predictor="cases",
            output_dir=tmpdir
        )
        
        assert "cases" in config.output_dir
        assert os.path.exists(config.output_dir)


def test_config_save_load():
    """Test saving and loading configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create config
        config1 = Config(
            db_path="/path/to/db.duckdb",
            predictor="prevalence",
            window_size=30,
            hidden_size=256
        )
        
        # Save config
        config_path = os.path.join(tmpdir, "config.json")
        config1.save(config_path)
        
        # Load config
        config2 = Config.load(config_path)
        
        assert config2.db_path == config1.db_path
        assert config2.predictor == config1.predictor
        assert config2.window_size == config1.window_size
        assert config2.hidden_size == config1.hidden_size