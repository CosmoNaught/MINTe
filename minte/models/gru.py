"""GRU model implementation."""
import torch
import torch.nn as nn

class GRUModel(nn.Module):
    """GRU Model for time series prediction."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 dropout_prob: float, num_layers: int = 1, predictor: str = "prevalence"):
        """Initialize GRU model."""
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.predictor = predictor
        
        # Enhanced architecture for cases
        if predictor == "cases":
            # Use larger hidden size internally for cases
            internal_hidden = hidden_size * 2
            
            # Input projection for cases
            self.input_projection = nn.Sequential(
                nn.Linear(input_size, internal_hidden),
                nn.LayerNorm(internal_hidden),
                nn.GELU(),
                nn.Dropout(dropout_prob)
            )
            
            self.gru = nn.GRU(
                internal_hidden, internal_hidden,
                num_layers=num_layers,
                dropout=dropout_prob if num_layers > 1 else 0.0,
                bidirectional=True  # Bidirectional for cases
            )
            
            # Attention layer for cases
            self.attention = nn.MultiheadAttention(
                internal_hidden * 2,  # *2 for bidirectional
                num_heads=8,
                dropout=dropout_prob
            )
            
            # Output projection with skip connection
            self.fc = nn.Sequential(
                nn.Linear(internal_hidden * 2, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_size, output_size)
            )
            
            self.fc_skip = nn.Linear(internal_hidden * 2, output_size)
            self.ln = nn.LayerNorm(internal_hidden * 2)
            
        else:
            # Original architecture for prevalence
            self.gru = nn.GRU(
                input_size, hidden_size,
                num_layers=num_layers,
                dropout=dropout_prob if num_layers > 1 else 0.0
            )
            self.fc = nn.Linear(hidden_size, output_size)
            self.ln = nn.LayerNorm(hidden_size)
            self.dropout = nn.Dropout(dropout_prob)
            self.activation = nn.Sigmoid()  # Bounded between 0 and 1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        if self.predictor == "cases":
            # Project input for cases
            x = self.input_projection(x.transpose(0, 1)).transpose(0, 1)
            
            # Bidirectional GRU
            out, _ = self.gru(x)
            
            # Self-attention
            out_attended, _ = self.attention(out, out, out)
            out = out + out_attended  # Residual connection
            
            # Layer norm
            out = self.ln(out)
            
            # Output with skip connection
            deep_out = self.fc(out)
            skip_out = self.fc_skip(out)
            return deep_out + 0.1 * skip_out
            
        else:
            # Original implementation for prevalence
            out, _ = self.gru(x)
            out = self.ln(out)
            out = self.dropout(out)
            out = self.fc(out)
            return self.activation(out)