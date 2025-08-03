# examples/advanced_usage.py
"""Advanced example with hyperparameter tuning."""

from minte import MalariaForecast
import json

# Initialize the forecaster
forecaster = MalariaForecast(
    db_path="/path/to/your/database.duckdb",
    table_name="simulation_results",
    predictor="prevalence",
    window_size=7,
    device="cuda"
)

# Run with hyperparameter tuning
results = forecaster.run(
    param_limit=500,
    sim_limit=10,
    use_cyclical_time=True,
    min_prevalence=0.02,
    num_workers=8,
    batch_size=2048,
    epochs=100,
    patience=20,
    # Hyperparameter tuning settings
    run_tuning=True,
    tuning_trials=50,
    tuning_timeout=3600*6,  # 6 hours
    output_dir="results/tuned_example",
    tuning_output_dir="results/tuning_trials"
)

# Save detailed results
with open("results/tuned_example/detailed_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Access tuned hyperparameters
print("\nTuned Hyperparameters:")
for model_type in ["gru", "lstm"]:
    print(f"\n{model_type.upper()}:")
    for param, value in results[model_type]["hyperparameters"].items():
        print(f"  {param}: {value}")