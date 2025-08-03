# minte/data/dataset.py
"""Dataset and DataLoader utilities for time series data."""

from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """Dataset for time series data."""
    
    def __init__(self, groups: List[Dict], lookback: int = 30):
        """Initialize time series dataset."""
        self.samples = []
        self.lookback = lookback
        for g in groups:
            ts = g['time_series']  
            y  = g['targets']      
            T  = g['length']

            for start in range(0, T - lookback + 1):
                end = start + lookback
                self.samples.append({
                    'X': ts[start:end], 
                    'Y': y[start:end],   
                    'param_sim_id': g['param_sim_id']
                })

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        item = self.samples[idx]
        X = item['X']
        Y = item['Y']
        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Custom collate function for batching time series data."""
    Xs = []
    Ys = []
    for (x, y) in batch:
        Xs.append(x)
        Ys.append(y)

    Xs = torch.stack(Xs, dim=0)
    Xs = Xs.permute(1, 0, 2)  # (time, batch, features)

    Ys = torch.stack(Ys, dim=0)
    Ys = Ys.permute(1, 0)  # (time, batch)

    return Xs, Ys