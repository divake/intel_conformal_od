"""MLP-based scoring function for conformal prediction intervals."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from .base import BaseScoringFunction


class MLPScoringFunction(BaseScoringFunction):
    """
    MLP-based regression scoring function for conformal prediction intervals.
    
    This network outputs the WIDTH of prediction intervals, not classification scores.
    The actual prediction interval is: pred ± (score * tau)
    """
    
    def __init__(self, input_dim: int = 17, hidden_dims: list = [256, 128, 64], 
                 dropout_rate: float = 0.15, activation: str = 'relu',
                 scoring_strategy: str = 'direct', output_constraint: str = 'natural'):
        """
        Args:
            input_dim: Dimension of input features (13 geometric + 4 uncertainty features)
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability for regularization
            activation: Activation function type ('relu', 'elu', 'leaky_relu')
            scoring_strategy: 'legacy' or 'direct' for adaptive scoring
            output_constraint: 'legacy', 'natural', or 'unconstrained'
        """
        super().__init__(input_dim, scoring_strategy, output_constraint)
        
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.activation_type = activation
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        # Hidden layers with proper regularization
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),  # Add batch normalization for stability
                self._get_activation(activation),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer - predict interval WIDTH (must be positive)
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights properly for regression
        self._initialize_weights()
    
    def _get_activation(self, activation: str):
        """Get activation function by name."""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'elu':
            return nn.ELU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.1)
        else:
            return nn.ReLU()
    
    def _initialize_weights(self):
        """Initialize network weights for regression task."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # He initialization for ReLU activations
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to predict interval width.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
               Features include coordinates, confidence, geometric features,
               and uncertainty indicators
        
        Returns:
            widths: Tensor of shape [batch_size, 1]
                   Predicted interval widths (always positive)
        """
        # Forward pass through network
        raw_output = self.network(x)
        
        # Ensure positive output using base class method
        widths = self.ensure_positive_output(raw_output)
        
        return widths
    
    def get_config(self) -> Dict[str, Any]:
        """Return model configuration for saving."""
        return {
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation_type,
            'scoring_strategy': self.scoring_strategy,
            'output_constraint': self.output_constraint
        }
    
    @property
    def model_name(self) -> str:
        """Return model name for logging."""
        return "MLP"


# Backward compatibility alias
RegressionScoringFunction = MLPScoringFunction