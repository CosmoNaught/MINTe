# examples/batch_experiments.py
"""Example of running batch experiments with different configurations."""

from minte import run_forecast
import itertools
import json
import os

# Define parameter grid
param_grid = {
    'window_size': [7, 14, 30],
    'use_cyclical_time': [True, False],
    'hidden_size': [128, 256],
    'num_layers': [2, 3]
}

# Base configuration
base_config = {
    'db_path': '/path/to/your/database.duckdb',
    'table_name': 'simulation_results',
    'predictor': 'prevalence',
    'param_limit': 100,
    'sim_limit': 5,
    'epochs': 50,
    'batch_size': 1024,
    'device': 'cuda'
}

# Run experiments
results_summary = []

for i, params in enumerate(itertools.product(*param_grid.values())):
    param_dict = dict(zip(param_grid.keys(), params))
    
    # Create unique output directory
    output_dir = f"results/experiment_{i:03d}"
    
    # Merge configurations
    config = {**base_config, **param_dict, 'output_dir': output_dir}
    
    print(f"\nRunning experiment {i+1} with parameters:")
    for k, v in param_dict.items():
        print(f"  {k}: {v}")
    
    try:
        # Run forecast
        results = run_forecast(**config)
        
        # Store summary
        results_summary.append({
            'experiment_id': i,
            'parameters': param_dict,
            'gru_r2': results['gru']['test_metrics']['r2'],
            'lstm_r2': results['lstm']['test_metrics']['r2'],
            'gru_rmse': results['gru']['test_metrics']['rmse'],
            'lstm_rmse': results['lstm']['test_metrics']['rmse']
        })
        
    except Exception as e:
        print(f"Experiment {i} failed: {str(e)}")
        continue

# Save summary
with open("results/experiments_summary.json", "w") as f:
    json.dump(results_summary, f, indent=2)

# Find best configuration
best_gru = max(results_summary, key=lambda x: x['gru_r2'])
best_lstm = max(results_summary, key=lambda x: x['lstm_r2'])

print("\nBest configurations:")
print(f"GRU: {best_gru['parameters']} - R²: {best_gru['gru_r2']:.4f}")
print(f"LSTM: {best_lstm['parameters']} - R²: {best_lstm['lstm_r2']:.4f}")