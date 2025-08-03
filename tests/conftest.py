# tests/conftest.py
"""Pytest configuration and fixtures."""

import pytest
import tempfile
import os


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_config():
    """Create a mock configuration for tests."""
    from minte.config import Config
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(
            db_path=os.path.join(tmpdir, "test.duckdb"),
            table_name="test_table",
            predictor="prevalence",
            output_dir=tmpdir,
            num_workers=0,  # Disable multiprocessing for tests
            epochs=2,  # Quick testing
            batch_size=32
        )
        yield config