# examples/prediction_example.py
"""Example of making predictions with trained models."""

import numpy as np
from minte import MalariaForecast

# Initialize and load existing models
forecaster = MalariaForecast(
    db_path="/path/to/your/database.duckdb",
    table_name="simulation_results",
    predictor="cases",
    window_size=30
)

# Load previously trained models
forecaster.load_models("results/trained_models")

# Create sample input data
# Shape: (timesteps, features)
# Features depend on your configuration (time encoding + static covariates)
sample_data = np.random.randn(100, 15).astype(np.float32)

# Make predictions with GRU model
gru_predictions = forecaster.predict(sample_data, model_type="gru")
print(f"GRU predictions shape: {gru_predictions.shape}")

# Make predictions with LSTM model
lstm_predictions = forecaster.predict(sample_data, model_type="lstm")
print(f"LSTM predictions shape: {lstm_predictions.shape}")