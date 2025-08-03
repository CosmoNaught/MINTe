# examples/basic_usage.py
"""Basic example of using the malaria forecast package."""

from minte import MalariaForecast

# Initialize the forecaster
forecaster = MalariaForecast(
    db_path="/path/to/your/database.duckdb",
    table_name="simulation_results",
    predictor="cases",  # or "prevalence"
    window_size=30,
    device="cuda"  # or "cpu"
)

# Run the full pipeline with default parameters
results = forecaster.run(
    param_limit=100,  # Use first 100 parameters for testing
    sim_limit=5,      # Use 5 simulations per parameter
    use_cyclical_time=True,
    output_dir="results/basic_example"
)

# Print results
print(f"GRU Model - Test R²: {results['gru']['test_metrics']['r2']:.4f}")
print(f"LSTM Model - Test R²: {results['lstm']['test_metrics']['r2']:.4f}")