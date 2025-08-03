# minte/training/trainer.py
"""Trainer class for training and evaluating models."""

import json
import logging
import os
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import Config

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer class for training and evaluating models."""
    
    def __init__(self, model: nn.Module, config: Config, output_dir: str, model_name: str):
        """Initialize trainer with model and configuration."""
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.output_dir = output_dir
        self.model_name = model_name
        self.criterion = nn.MSELoss()
        self.model.to(self.device)
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              optimizer: torch.optim.Optimizer, scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
              epochs: Optional[int] = None) -> Tuple[nn.Module, float, int]:
        """Train model with early stopping and best model checkpointing."""
        epochs = epochs or self.config.epochs
        scaler = GradScaler()
        
        # Initialize best validation loss and patience counter for early stopping
        best_val_loss = float('inf')
        patience = self.config.patience
        patience_counter = 0
        best_epoch = 0
        
        # Path to save best model
        best_model_path = os.path.join(self.output_dir, f"{self.model_name}_best.pt")
        training_history = {'train_loss': [], 'val_loss': [], 'epochs': []}
        
        for epoch in range(1, epochs + 1):
            logger.info(f"Starting Epoch {epoch}/{epochs}")
            epoch_start_time = time.time()

            # Training phase
            self.model.train()
            total_train_loss = 0.0
            train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch} - Training", leave=False)

            for X, Y in train_loader_tqdm:
                X = X.to(self.device, non_blocking=True)
                Y = Y.to(self.device, non_blocking=True)

                optimizer.zero_grad()

                with autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu'):
                    pred = self.model(X).squeeze(-1)
                    loss = self.criterion(pred, Y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_train_loss += loss.item()
                train_loader_tqdm.set_postfix(loss=loss.item())

            avg_train_loss = total_train_loss / len(train_loader)
            logger.info(f"Epoch {epoch} - Average Training Loss: {avg_train_loss:.6f}")

            # Validation phase
            self.model.eval()
            total_val_loss = 0.0
            val_predictions = []
            val_targets = []
            val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch} - Validation", leave=False)

            with torch.no_grad():
                for X_val, Y_val in val_loader_tqdm:
                    X_val = X_val.to(self.device, non_blocking=True)
                    Y_val = Y_val.to(self.device, non_blocking=True)

                    with autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu'):
                        pred_val = self.model(X_val).squeeze(-1)
                        loss_val = self.criterion(pred_val, Y_val)

                    total_val_loss += loss_val.item()
                    val_loader_tqdm.set_postfix(val_loss=loss_val.item())
                    
                    # Collect predictions for additional metrics
                    val_predictions.append(pred_val.cpu().numpy())
                    val_targets.append(Y_val.cpu().numpy())

            avg_val_loss = total_val_loss / len(val_loader)
            epoch_duration = time.time() - epoch_start_time
            
            # Record training history
            training_history['train_loss'].append(avg_train_loss)
            training_history['val_loss'].append(avg_val_loss)
            training_history['epochs'].append(epoch)

            # Apply learning rate scheduler if provided
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()

            # Check if this is the best model so far
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                patience_counter = 0
                
                # Save the best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                    'train_loss': avg_train_loss
                }, best_model_path)
                
                logger.info(f"New best model saved at epoch {epoch} with validation loss: {avg_val_loss:.6f}")
            else:
                patience_counter += 1
                logger.info(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")

            logger.info(f"Epoch {epoch} Completed - Avg Validation Loss: {avg_val_loss:.6f} | Duration: {epoch_duration:.2f}s")
            
            # Early stopping check
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs. Best epoch was {best_epoch} with validation loss: {best_val_loss:.6f}")
                break
        
        # Save training history
        history_path = os.path.join(self.output_dir, f"{self.model_name}_training_history.json")
        with open(history_path, 'w') as f:
            json.dump(training_history, f)
        
        logger.info(f"Training completed. Best model was at epoch {best_epoch} with validation loss: {best_val_loss:.6f}")
        
        # Load the best model for return
        checkpoint = torch.load(best_model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Also save a final model (in case we want both final and best)
        final_model_path = os.path.join(self.output_dir, f"{self.model_name}_final.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
            'train_loss': avg_train_loss,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss
        }, final_model_path)
        logger.info(f"Final model saved to {final_model_path}")
        
        return self.model, best_val_loss, best_epoch

    def evaluate(self, data_loader: DataLoader, dataset_name: str = "Test") -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """Evaluate model on a dataset with comprehensive metrics."""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        
        # Evaluate with no gradients
        with torch.no_grad():
            for X, Y in tqdm(data_loader, desc=f"Evaluating on {dataset_name} Set"):
                X = X.to(self.device, non_blocking=True)
                Y = Y.to(self.device, non_blocking=True)
                
                with autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu'):
                    pred = self.model(X).squeeze(-1)
                    loss = self.criterion(pred, Y)
                
                total_loss += loss.item()
                all_predictions.append(pred.cpu().numpy())
                all_targets.append(Y.cpu().numpy())
        
        # Concatenate predictions and targets
        all_predictions = np.concatenate([p.flatten() for p in all_predictions])
        all_targets = np.concatenate([t.flatten() for t in all_targets])
        
        # Transform back from log space for cases
        if self.config.predictor == "cases":
            all_predictions = np.expm1(all_predictions)  # inverse of log1p
            all_targets = np.expm1(all_targets)
            
        # Calculate standard metrics
        mse = mean_squared_error(all_targets, all_predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(all_targets, all_predictions)
        r2 = r2_score(all_targets, all_predictions)
        
        # Additional metrics depend on whether we're predicting prevalence or cases
        epsilon = 1e-7  # To avoid division by zero
        
        # Symmetric Mean Absolute Percentage Error
        smape = 100 * np.mean(2 * np.abs(all_predictions - all_targets) / 
                             (np.abs(all_predictions) + np.abs(all_targets) + epsilon))
        
        # Calculate bias
        bias = np.mean(all_predictions - all_targets)
        
        if self.config.predictor == "prevalence":
            # Log likelihood of Beta distribution (approximation for bounded variables)
            scaled_pred = np.clip(all_predictions, epsilon, 1-epsilon)
            scaled_targets = np.clip(all_targets, epsilon, 1-epsilon)
            log_likelihood = np.mean(np.log(scaled_pred) * scaled_targets + 
                                   np.log(1 - scaled_pred) * (1 - scaled_targets))
        else:
            log_likelihood = float('nan')
        
        # Create results dictionary
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'smape': smape,
            'bias': bias,
            'log_likelihood': log_likelihood,
            'avg_loss': total_loss / len(data_loader)
        }
        
        # Print metrics
        logger.info(f"\n{dataset_name} Set Evaluation Results:")
        logger.info(f"    MSE:  {mse:.6f}")
        logger.info(f"    RMSE: {rmse:.6f}")
        logger.info(f"    MAE:  {mae:.6f}")
        logger.info(f"    R²:   {r2:.6f}")
        logger.info(f"    SMAPE: {smape:.2f}%")
        logger.info(f"    Bias: {bias:.6f}")
        if not np.isnan(log_likelihood):
            logger.info(f"    Log-Likelihood: {log_likelihood:.6f}")
        
        return metrics, all_predictions, all_targets
    
    def predict_sequence(self, full_ts: np.ndarray) -> np.ndarray:
        """Predict a full time series sequence."""
        self.model.eval()
        with torch.no_grad(), autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu'):
            x_torch = torch.tensor(full_ts, dtype=torch.float32).unsqueeze(1).to(self.device)
            pred = self.model(x_torch).squeeze(-1).squeeze(-1).cpu().numpy()
        return pred
