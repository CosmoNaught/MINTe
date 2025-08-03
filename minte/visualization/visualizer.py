# minte/visualization/visualizer.py
"""Visualization utilities for model results."""

import json
import logging
import math
import os
import random
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..config import Config
from ..data import DataModule
from ..training import Trainer

logger = logging.getLogger(__name__)


class Visualizer:
    """Class for visualizing model results."""
    
    def __init__(self, config: Config, data_module: DataModule):
        """Initialize visualizer with configuration and data module."""
        self.config = config
        self.data_module = data_module
        
    def plot_training_history(self, output_dir: str, model_results: Dict[str, Dict[str, Any]]) -> None:
        """Plot combined training history for GRU and LSTM models."""
        gru_history_path = os.path.join(output_dir, "gru_training_history.json")
        lstm_history_path = os.path.join(output_dir, "lstm_training_history.json")
        
        if os.path.exists(gru_history_path) and os.path.exists(lstm_history_path):
            with open(gru_history_path, 'r') as f:
                gru_history = json.load(f)
            
            with open(lstm_history_path, 'r') as f:
                lstm_history = json.load(f)
            
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.plot(gru_history['epochs'], gru_history['train_loss'], label='Train Loss')
            plt.plot(gru_history['epochs'], gru_history['val_loss'], label='Validation Loss')
            plt.axvline(x=model_results["gru"]["best_epoch"], color='r', linestyle='--', alpha=0.5, 
                       label=f'Best Epoch ({model_results["gru"]["best_epoch"]})')
            plt.title('GRU Training History')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.plot(lstm_history['epochs'], lstm_history['train_loss'], label='Train Loss')
            plt.plot(lstm_history['epochs'], lstm_history['val_loss'], label='Validation Loss')
            plt.axvline(x=model_results["lstm"]["best_epoch"], color='r', linestyle='--', alpha=0.5, 
                       label=f'Best Epoch ({model_results["lstm"]["best_epoch"]})')
            plt.title('LSTM Training History')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(alpha=0.3)
            
            plt.tight_layout()
            plot_path = os.path.join(output_dir, "training_history.png")
            plt.savefig(plot_path)
            logger.info(f"Training history plot saved to {plot_path}")
            plt.close()
            
    def plot_model_comparison(self, output_dir: str, model_metrics: Dict[str, Dict[str, Any]]) -> None:
        """Plot comparison of model metrics."""
        plt.figure(figsize=(10, 6))
        
        metrics_to_plot = ['mse', 'rmse', 'mae', 'r2']
        x = np.arange(len(metrics_to_plot))
        width = 0.35
        
        gru_values = [model_metrics["gru"]["test_metrics"][m] for m in metrics_to_plot]
        lstm_values = [model_metrics["lstm"]["test_metrics"][m] for m in metrics_to_plot]
        
        plt.bar(x - width/2, gru_values, width, label='GRU')
        plt.bar(x + width/2, lstm_values, width, label='LSTM')
        
        plt.xlabel('Metric')
        plt.ylabel('Value')
        
        # Update title based on predictor type
        title_suffix = "Prevalence" if self.config.predictor == "prevalence" else "Cases"
        plt.title(f'Model Performance Comparison on Test Set ({title_suffix})')
        
        plt.xticks(x, metrics_to_plot)
        plt.legend()
        plt.grid(alpha=0.3)
        
        plot_path = os.path.join(output_dir, "model_comparison.png")
        plt.savefig(plot_path)
        logger.info(f"Model comparison plot saved to {plot_path}")
        plt.close()
            
    def plot_test_predictions(self, output_dir: str, models: Dict[str, nn.Module], trainers: Dict[str, Trainer]) -> None:
        """Visualize test set predictions."""
        # Fetch test data for visualization
        df_test_sims = self.data_module.fetch_data()
        
        # Get test parameters
        test_param_indices = list(self.data_module.test_params)
        random.shuffle(test_param_indices)
        subset_for_plot = test_param_indices[:9] if len(test_param_indices) >= 9 else test_param_indices
        
        # Filter to only include chosen test parameters
        df_test_sims = df_test_sims[df_test_sims["parameter_index"].isin(subset_for_plot)]
        param_groups = df_test_sims.groupby("parameter_index")

        n_plots = len(subset_for_plot)
        cols = 3
        rows = (n_plots + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 10), sharex=False, sharey=False)
        axes = axes.flatten() if n_plots > 1 else [axes]

        # Create a list to store all plot data
        all_plot_data = []

        # Set y-axis label based on predictor type
        y_label = "Prevalence" if self.config.predictor == "prevalence" else "Cases per 1000"

        for i, param_idx in enumerate(subset_for_plot):
            if param_idx not in param_groups.groups:
                logger.warning(f"Parameter index {param_idx} not found in test data, skipping")
                continue
                
            subdf_param = param_groups.get_group(param_idx)
            sim_groups = subdf_param.groupby("simulation_index")
            
            if len(sim_groups) == 0:
                logger.warning(f"No simulation data for parameter {param_idx}, skipping")
                continue

            ax = axes[i]
            raw_param_index = subdf_param['parameter_index'].iloc[0]

            # Get the target column name based on predictor type
            target_col = "prevalence" if self.config.predictor == "prevalence" else "cases"

            # Plot all simulations for this parameter
            for sim_idx, sim_df in sim_groups:
                # Check if this exact parameter-sim combination was in test set
                if (param_idx, sim_idx) in self.data_module.test_param_sims:
                    linestyle = "-"  # Solid line for test sims
                    alpha = 0.7
                    label = f"True {y_label} (Test)" if sim_idx == list(sim_groups.groups.keys())[0] else None
                else:
                    linestyle = "--"  # Dashed line for other sims
                    alpha = 0.3
                    label = f"True {y_label} (Other)" if sim_idx == list(sim_groups.groups.keys())[0] else None
                    
                t = sim_df["timesteps"].values.astype(np.float32)
                
                # Scale the x-axis for cases - each timestep represents 30 days
                if self.config.predictor == "cases":
                    t = t * self.config.window_size
                    
                y_true = sim_df[target_col].values.astype(np.float32)
                ax.plot(t, y_true, color="black", alpha=alpha, linewidth=1, linestyle=linestyle, label=label)

            # For prediction, use only a test simulation
            valid_sim_indices = [sim_idx for sim_idx in sim_groups.groups.keys() 
                                if (param_idx, sim_idx) in self.data_module.test_param_sims]
            
            if not valid_sim_indices:
                logger.warning(f"No test simulations found for parameter {param_idx}, using first available sim")
                first_sim_idx = list(sim_groups.groups.keys())[0]
            else:
                first_sim_idx = valid_sim_indices[0]
                
            sim_df = sim_groups.get_group(first_sim_idx).sort_values("timesteps")
            
            # Get the global index for this param-sim
            global_idx = sim_df['global_index'].iloc[0]

            t = sim_df["timesteps"].values.astype(np.float32)
            static_vals = sim_df.iloc[0][self.data_module.static_covars].values.astype(np.float32)
            # Normalize static values
            static_vals = self.data_module.static_scaler.transform(static_vals.reshape(1, -1)).flatten()
            
            T = len(sim_df)

            if self.config.use_cyclical_time:
                if self.config.predictor == "cases":
                    day_of_year = (t * self.config.window_size) % 365.0
                else:
                    day_of_year = t % 365.0
                    
                sin_t = np.sin(2 * math.pi * day_of_year / 365.0)
                cos_t = np.cos(2 * math.pi * day_of_year / 365.0)
                X_full = np.zeros((T, 2 + len(self.data_module.static_covars)), dtype=np.float32)
                for j in range(T):
                    X_full[j, 0] = sin_t[j]
                    X_full[j, 1] = cos_t[j]
                    X_full[j, 2:] = static_vals
            else:
                # Normalize timesteps
                t_min, t_max = np.min(t), np.max(t)
                t_norm = (t - t_min) / (t_max - t_min) if t_max > t_min else t
                
                X_full = np.zeros((T, 1 + len(self.data_module.static_covars)), dtype=np.float32)
                for j in range(T):
                    X_full[j, 0] = t_norm[j]
                    X_full[j, 1:] = static_vals

            # Create base entry for plot data
            plot_data_entry = {
                'parameter_index': raw_param_index,
                'simulation_index': first_sim_idx,
                'global_index': global_idx,
                'is_test': (param_idx, first_sim_idx) in self.data_module.test_param_sims
            }

            # Scale t for plotting if using cases (30-day intervals)
            if self.config.predictor == "cases":
                t_plot = t * self.config.window_size
            else:
                t_plot = t

            # Make predictions with both models
            for model_type in ["gru", "lstm"]:
                y_pred = trainers[model_type].predict_sequence(X_full)
                
                # Transform predictions back from log space for cases
                if self.config.predictor == "cases":
                    y_pred = np.expm1(y_pred)  # inverse of log1p
                
                # Plot predictions
                color = "red" if model_type == "gru" else "blue"
                ax.plot(t_plot, y_pred, label=model_type.upper(), color=color)
                
                # Add to plot data
                for j in range(len(t)):
                    if j == 0:  # First iteration, create entry
                        if 'timestep' not in plot_data_entry:
                            plot_data_entry['timestep'] = t_plot[j]
                            plot_data_entry[f'true_{self.config.predictor}'] = sim_df[target_col].values[j]
                        plot_data_entry[f'{model_type}_prediction'] = y_pred[j]
                    else:  # Subsequent iterations, update existing entry
                        entry_copy = plot_data_entry.copy()
                        entry_copy['timestep'] = t_plot[j]
                        entry_copy[f'true_{self.config.predictor}'] = sim_df[target_col].values[j]
                        entry_copy[f'{model_type}_prediction'] = y_pred[j]
                        all_plot_data.append(entry_copy)
            
            # Only add the first entry once (since we've already added the rest in the loop)
            all_plot_data.append(plot_data_entry)
            
            test_status = "(Test)" if param_idx in self.data_module.test_params else "(Non-Test)"
            ax.set_title(f"Parameter Index = {raw_param_index} {test_status}")
            ax.set_xlabel("Time Step" if self.config.predictor == "prevalence" else "Days")
            ax.set_ylabel(y_label)
            ax.legend()

        # For any empty subplots, hide them
        for i in range(len(subset_for_plot), len(axes)):
            axes[i].axis('off')

        # Save the plot
        plot_path = os.path.join(output_dir, "test_predictions.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        logger.info(f"Saved test plot to {plot_path}")
        plt.close()

        # Save the plot data to CSV
        plot_data_df = pd.DataFrame(all_plot_data)
        csv_path = os.path.join(output_dir, "test_plot_data.csv")
        plot_data_df.to_csv(csv_path, index=False)
        logger.info(f"Saved test plot data to {csv_path}")