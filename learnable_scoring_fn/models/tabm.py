"""TabM (Tabular M) with parameter-efficient ensembling for interval prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List
from .base import BaseScoringFunction


class TabMScoringFunction(BaseScoringFunction):
    """TabM with BatchEnsemble-style parameter-efficient ensembling.
    
    TabM uses a shared backbone with lightweight ensemble-specific adapters,
    achieving ensemble benefits with minimal parameter overhead.
    """
    
    def __init__(self, input_dim: int = 17, hidden_dims: List[int] = [128, 64, 32], 
                 n_ensemble: int = 8, dropout: float = 0.15, use_skip: bool = True,
                 scoring_strategy: str = 'direct', output_constraint: str = 'natural'):
        """
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            n_ensemble: Number of ensemble members
            dropout: Dropout probability
            use_skip: Whether to use skip connections
            scoring_strategy: 'legacy' or 'direct' for adaptive scoring
            output_constraint: 'legacy', 'natural', or 'unconstrained'
        """
        super().__init__(input_dim, scoring_strategy, output_constraint)
        
        self.hidden_dims = hidden_dims
        self.n_ensemble = n_ensemble
        self.dropout_rate = dropout
        self.use_skip = use_skip
        
        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_dim)
        
        # Build shared backbone layers
        layers = []
        self.layer_norms = nn.ModuleList()
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layer_norms.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        
        self.shared_layers = nn.ModuleList(layers)
        
        # BatchEnsemble-style adapters (rank-1 factors)
        # Input adapter
        self.input_adapters = nn.Parameter(
            torch.ones(n_ensemble, input_dim) + 0.1 * torch.randn(n_ensemble, input_dim)
        )
        
        # Layer adapters
        self.layer_adapters = nn.ParameterList([
            nn.Parameter(
                torch.ones(n_ensemble, dim) + 0.1 * torch.randn(n_ensemble, dim)
            )
            for dim in hidden_dims
        ])
        
        # Ensemble-specific output heads
        self.ensemble_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1] // 2, 1)
            ) for _ in range(n_ensemble)
        ])
        
        # Optional: learnable ensemble weights for final aggregation
        self.ensemble_weights = nn.Parameter(torch.ones(n_ensemble) / n_ensemble)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights properly."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TabM ensemble.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            widths: Predicted interval widths [batch_size, 1]
        """
        batch_size = x.size(0)
        
        # Normalize input
        x = self.input_norm(x)
        
        # Process through each ensemble member
        ensemble_outputs = []
        
        for i in range(self.n_ensemble):
            # Apply input adapter
            h = x * self.input_adapters[i].unsqueeze(0)
            
            # Store for skip connections
            skip_connections = [h] if self.use_skip else []
            
            # Process through shared layers with adapters
            for j, (layer, norm, adapter) in enumerate(
                zip(self.shared_layers, self.layer_norms, self.layer_adapters)
            ):
                # Linear transformation
                h = layer(h)
                
                # Apply adapter (element-wise multiplication)
                h = h * adapter[i].unsqueeze(0)
                
                # Normalization
                h = norm(h)
                
                # Activation and dropout
                h = self.activation(h)
                h = self.dropout(h)
                
                # Skip connection (if enabled and dimensions match)
                if self.use_skip and j > 0 and j < len(self.shared_layers) - 1:
                    if h.size(-1) == skip_connections[-1].size(-1):
                        h = h + skip_connections[-1]
                    skip_connections.append(h)
            
            # Ensemble-specific output
            out = self.ensemble_heads[i](h)
            ensemble_outputs.append(out)
        
        # Stack ensemble outputs
        ensemble_outputs = torch.stack(ensemble_outputs, dim=1)  # [batch_size, n_ensemble, 1]
        
        # Weighted average with learnable weights
        weights = F.softmax(self.ensemble_weights, dim=0)
        weighted_output = (ensemble_outputs * weights.view(1, -1, 1)).sum(dim=1)  # [batch_size, 1]
        
        # Ensure positive output
        return self.ensure_positive_output(weighted_output)
    
    def get_config(self) -> Dict[str, Any]:
        """Return model configuration."""
        return {
            'hidden_dims': self.hidden_dims,
            'n_ensemble': self.n_ensemble,
            'dropout': self.dropout_rate,
            'use_skip': self.use_skip,
            'scoring_strategy': self.scoring_strategy,
            'output_constraint': self.output_constraint
        }
    
    @property
    def model_name(self) -> str:
        """Return model name."""
        return "TabM"