# Malaria INtervention Tool Emulator - Installation and Usage Guide

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Detailed Usage](#detailed-usage)
4. [API Reference](#api-reference)
5. [Command Line Interface](#command-line-interface)
6. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for faster training)
- At least 16GB RAM (24GB+ recommended for large datasets)

### From Source

1. Clone the repository:
```bash
git clone https://github.com/CosmoNaught/minte.git
cd minte
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package in development mode:
```bash
pip install -e .
```

### Using pip (when published)

```bash
pip install minte
```

### Verify Installation

```python
import minte
print(minte.__version__)
```

## Quick Start

### Using the Python API

```python
from minte import MalariaForecast

# Initialize forecaster
forecaster = MalariaForecast(
    db_path="path/to/your/data.duckdb",
    table_name="simulation_results",
    predictor="cases",  # or "prevalence"
    window_size=30,
    device="cuda"  # or "cpu"
)

# Run with default parameters
results = forecaster.run(
    param_limit=1000,
    sim_limit=10,
    output_dir="results/my_experiment"
)

# Print results
print(f"GRU R²: {results['gru']['test_metrics']['r2']:.4f}")
print(f"LSTM R²: {results['lstm']['test_metrics']['r2']:.4f}")
```

### Using the Command Line

```bash
minte \
  --db-path /path/to/data.duckdb \
  --table-name simulation_results \
  --predictor cases \
  --window-size 30 \
  --param-limit 2048 \
  --sim-limit 8 \
  --use-cyclical-time \
  --device cuda \
  --output-dir results
```

## Detailed Usage

### Data Requirements

Your DuckDB database should contain a table with the following columns:
- `parameter_index`: Parameter set identifier
- `simulation_index`: Simulation run identifier
- `timesteps`: Time step in the simulation
- `n_detect_lm_0_1825` (for prevalence): Number of detected cases
- `n_age_0_1825` (for prevalence): Population size
- `n_inc_clinical_0_36500` (for cases): Clinical incidence count
- `n_age_0_36500` (for cases): Population size for cases
- Static covariates: `eir`, `dn0_use`, `dn0_future`, `Q0`, `phi_bednets`, `seasonal`, `routine`, `itn_use`, `irs_use`, `itn_future`, `irs_future`, `lsm`

### Hyperparameter Tuning

Run automatic hyperparameter optimization:

```python
results = forecaster.run(
    param_limit=2048,
    sim_limit=8,
    run_tuning=True,
    tuning_trials=50,
    tuning_timeout=3600*6,  # 6 hours
    output_dir="results/tuned",
    tuning_output_dir="tuning_trials"
)
```

### Using Pre-tuned Parameters

```python
# First run: tune parameters
results = forecaster.run(
    run_tuning=True,
    tuning_output_dir="tuning_results"
)

# Later runs: use tuned parameters
results = forecaster.run(
    use_tuned_parameters=True,
    tuning_output_dir="tuning_results"
)
```

### Making Predictions

```python
# Load trained models
forecaster = MalariaForecast(
    db_path="path/to/data.duckdb",
    predictor="cases"
)
forecaster.load_models("results/trained_models")

# Prepare input data (shape: timesteps × features)
import numpy as np
input_data = np.random.randn(100, 15).astype(np.float32)

# Make predictions
predictions = forecaster.predict(input_data, model_type="gru")
```

## API Reference

### MalariaForecast Class

```python
class MalariaForecast:
    def __init__(
        self,
        db_path: str,
        table_name: str = "simulation_results",
        predictor: str = "prevalence",
        window_size: int = 7,
        device: Optional[str] = None,
        seed: int = 42
    )
```

**Parameters:**
- `db_path`: Path to DuckDB database
- `table_name`: Name of the table containing simulation data
- `predictor`: Target variable - "prevalence" or "cases"
- `window_size`: Window size for data aggregation
- `device`: Computing device - "cuda" or "cpu" (auto-detected if None)
- `seed`: Random seed for reproducibility

### run() Method

```python
def run(
    self,
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
    tuning_output_dir: str = "results_tuned"
) -> Dict[str, Any]
```

**Key Parameters:**
- `param_limit`: Maximum parameter index to include (or "all")
- `sim_limit`: Number of simulations per parameter (or "all")
- `use_cyclical_time`: Use sin/cos encoding for time
- `lookback`: Sequence length for RNN input
- `run_tuning`: Enable hyperparameter optimization
- `use_tuned_parameters`: Use previously tuned parameters

**Returns:**
Dictionary containing results for both GRU and LSTM models with metrics, predictions, and hyperparameters.

## Command Line Interface

### Basic Options

```bash
minte --help  # Show all options
```

### Common Use Cases

1. **Basic training with prevalence prediction:**
```bash
minte \
  --db-path data.duckdb \
  --predictor prevalence \
  --output-dir results/prevalence
```

2. **Cases prediction with cyclical time encoding:**
```bash
minte \
  --db-path data.duckdb \
  --predictor cases \
  --use-cyclical-time \
  --window-size 30 \
  --output-dir results/cases
```

3. **Hyperparameter tuning:**
```bash
minte \
  --db-path data.duckdb \
  --predictor cases \
  --run-tuning \
  --tuning-trials 100 \
  --tuning-timeout 43200
```

4. **Using multiple GPUs (data parallel):**
```bash
CUDA_VISIBLE_DEVICES=0,1 minte \
  --db-path data.duckdb \
  --device cuda \
  --batch-size 8192
```

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM) Errors:**
   - Reduce `batch_size`
   - Reduce `hidden_size` or `num_layers`
   - Use fewer workers (`--num-workers 0`)
   - Use CPU instead of GPU for small datasets

2. **Slow Data Loading:**
   - Ensure DuckDB file is on fast storage (SSD)
   - Reduce `param_limit` or `sim_limit` for testing
   - Increase `num_workers` (if not using GPU)

3. **Poor Model Performance:**
   - Check data quality and prevalence thresholds
   - Try hyperparameter tuning
   - Increase `lookback` window
   - Use `use_cyclical_time` for seasonal patterns

### Performance Tips

1. **For Large Datasets:**
   ```python
   results = forecaster.run(
       batch_size=8192,
       num_workers=8,
       device="cuda"
   )
   ```

2. **For Limited Memory:**
   ```python
   results = forecaster.run(
       batch_size=512,
       hidden_size=128,
       num_workers=0
   )
   ```

3. **For Quick Testing:**
   ```python
   results = forecaster.run(
       param_limit=100,
       sim_limit=2,
       epochs=10,
       output_dir="test_run"
   )
   ```

### Logging

Enable detailed logging:

```python
import logging
logging.getLogger('minte').setLevel(logging.DEBUG)
```

### Getting Help

- Check the [examples/](examples/) directory for more usage examples
- Report issues on GitHub
- Check existing models in the output directory for reuse

## Next Steps

1. Explore the [examples/](examples/) directory for advanced usage
2. Read the API documentation for detailed parameter descriptions
3. Check the visualization outputs in your results directory
4. Consider contributing to the project on GitHub