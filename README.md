# Malaria Forecast

A Python package for time series forecasting of malaria prevalence and clinical cases using RNN models (GRU and LSTM) based off the malariasimulation framework.

## Features

- Support for both prevalence and case count prediction
- GRU and LSTM model architectures
- Automatic hyperparameter tuning with Optuna
- Comprehensive evaluation metrics
- Visualization of results
- GPU acceleration support
- Modular and extensible design

## Installation

### From Source

```bash
git clone https://github.com/CosmoNaught/minte.git
cd minte
pip install -e .
```

### Using pip

```bash
pip install minte
```

## Quick Start

### Command Line Interface

```bash
minte \
  --db-path /path/to/your/database.duckdb \
  --table-name simulation_results \
  --predictor cases \
  --window-size 30 \
  --param-limit 2048 \
  --sim-limit 8 \
  --use-cyclical-time \
  --device cuda \
  --output-dir results
```

### Python API

```python
from minte import MalariaForecast

# Initialize the forecaster
forecaster = MalariaForecast(
    db_path="/path/to/your/database.duckdb",
    table_name="simulation_results",
    predictor="cases",  # or "prevalence"
    window_size=30,
    device="cuda"
)

# Run the full pipeline with hyperparameter tuning
results = forecaster.run(
    param_limit=2048,
    sim_limit=8,
    use_cyclical_time=True,
    run_tuning=True,
    output_dir="results"
)

# Access results
print(f"GRU Test R²: {results['gru']['test_metrics']['r2']:.4f}")
print(f"LSTM Test R²: {results['lstm']['test_metrics']['r2']:.4f}")

# Make predictions on new data
predictions = forecaster.predict(data)
```

## Documentation (COMING SOON!)

For detailed documentation, see the [docs](docs/) directory.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
```