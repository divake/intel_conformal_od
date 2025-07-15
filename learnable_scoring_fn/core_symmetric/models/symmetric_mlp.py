"""Symmetric Adaptive MLP for conformal prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


class SymmetricAdaptiveMLP(nn.Module):
    """
    Predicts symmetric interval widths for each coordinate.
    
    Output: [wx0, wy0, wx1, wy1] where:
    - wx0: symmetric width for x0 coordinate (left/right)
    - wy0: symmetric width for y0 coordinate (top/bottom)
    - wx1: symmetric width for x1 coordinate (left/right)
    - wy1: symmetric width for y1 coordinate (top/bottom)
    
    The model learns to predict appropriate widths based on:
    - Object size and aspect ratio
    - Position in image
    - Confidence scores
    - Other geometric features
    """
    
    def __init__(
        self, 
        input_dim: int = 17,
        hidden_dims: list = None,
        dropout_rate: float = 0.1,
        activation: str = 'relu',
        use_batch_norm: bool = True
    ):
        """
        Initialize the symmetric MLP.
        
        Args:
            input_dim: Dimension of input features (default: 17)
            hidden_dims: List of hidden layer dimensions (default: [128, 128])
            dropout_rate: Dropout probability for regularization
            activation: Activation function type ('relu', 'elu', 'gelu')
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 128]
            
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.activation_type = activation
        self.use_batch_norm = use_batch_norm
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization (if enabled)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(self._get_activation(activation))
            
            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer - 4 width predictions
        self.feature_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dims[-1], 4)
        
        # Initialize weights
        self._initialize_weights()
        
        # For logging purposes
        self.training_step = 0
        
    def _get_activation(self, activation: str):
        """Get activation function by name."""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'elu':
            return nn.ELU()
        elif activation == 'gelu':
            return nn.GELU()
        else:
            return nn.ReLU()
    
    def _initialize_weights(self):
        """Initialize network weights properly."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/He initialization depending on activation
                if self.activation_type == 'relu':
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(module.weight)
                    
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        
        # CRITICAL: Initialize output layer to prevent collapse
        # Set different biases for each output to encourage variation
        with torch.no_grad():
            # Larger weight initialization to encourage variation
            nn.init.normal_(self.output_layer.weight, mean=0, std=0.1)
            # Different initial biases to break symmetry
            self.output_layer.bias.data = torch.tensor([0.5, 0.8, 0.6, 0.7])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to predict symmetric interval widths.
        
        Args:
            x: Input tensor of shape [batch_size, 17]
               Features include coordinates, confidence, geometric features,
               and uncertainty indicators
        
        Returns:
            widths: Tensor of shape [batch_size, 4]
                   Predicted symmetric widths for each coordinate
                   All values are positive (enforced by exponential transformation)
        """
        # Pass through feature layers
        features = self.feature_layers(x)
        
        # Get raw width predictions
        raw_widths = self.output_layer(features)
        
        # FIXED: Use sigmoid-based transformation for stable gradients
        # Scale raw outputs to reasonable range
        # Sigmoid gives [0, 1], then scale to desired width range
        # This prevents both collapse and explosion
        
        # Apply sigmoid to get values in [0, 1]
        sigmoid_outputs = torch.sigmoid(raw_widths)
        
        # Scale to desired range: [3, 30] pixels
        # This gives adaptive widths while preventing collapse
        min_width = 3.0
        max_width = 30.0
        widths = min_width + sigmoid_outputs * (max_width - min_width)
        
        # Add size-based modulation to encourage adaptation
        if x.shape[1] >= 7:  # Ensure we have area feature
            area_feature = x[:, 6:7]  # Box area as fraction of image
            # Larger objects get potentially larger widths
            size_factor = 1.0 + area_feature * 2.0  # [1.0, 3.0] range
            widths = widths * size_factor
        
        # Add small noise during training to prevent identical outputs
        if self.training:
            noise = torch.randn_like(widths) * 0.1
            widths = widths + noise.abs()  # Keep widths positive
        
        # Log statistics during training (every 100 steps)
        if self.training and self.training_step % 100 == 0:
            with torch.no_grad():
                mean_widths = widths.mean(dim=0)
                std_widths = widths.std(dim=0)
                print(f"Step {self.training_step} - Width stats:")
                print(f"  Mean: {mean_widths.detach().cpu().numpy()}")
                print(f"  Std:  {std_widths.detach().cpu().numpy()}")
                print(f"  Raw range: [{raw_widths.min().item():.2f}, {raw_widths.max().item():.2f}]")
        
        if self.training:
            self.training_step += 1
        
        return widths
    
    def get_config(self) -> Dict[str, Any]:
        """Return model configuration for saving/loading."""
        return {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation_type,
            'use_batch_norm': self.use_batch_norm
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """Create model from configuration dictionary."""
        return cls(**config)
    
    def predict_intervals(
        self, 
        features: torch.Tensor,
        predictions: torch.Tensor,
        tau: float = 1.0
    ) -> tuple:
        """
        Predict symmetric intervals given features and predictions.
        
        Args:
            features: Input features [batch_size, 17]
            predictions: Predicted boxes [batch_size, 4]
            tau: Calibration factor (default: 1.0)
            
        Returns:
            lower_bounds: Lower bounds of intervals [batch_size, 4]
            upper_bounds: Upper bounds of intervals [batch_size, 4]
        """
        # Get width predictions
        widths = self.forward(features)
        
        # Apply tau scaling
        calibrated_widths = widths * tau
        
        # Create symmetric intervals
        lower_bounds = predictions - calibrated_widths
        upper_bounds = predictions + calibrated_widths
        
        return lower_bounds, upper_bounds
    
    @property
    def model_name(self) -> str:
        """Return model name for logging."""
        return f"SymmetricMLP_h{'x'.join(map(str, self.hidden_dims))}"


class AdaptiveMLPV2(nn.Module):
    """
    Coverage Dropout with Decomposed Architecture.
    
    This model decomposes uncertainty into multiple components and uses
    coverage dropout during training to prevent collapse.
    """
    
    def __init__(self, input_dim=17, hidden_dims=[256, 128, 64]):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # Decomposed networks for different uncertainty aspects
        self.size_net = nn.Sequential(
            nn.Linear(4, 64),  # Box coordinates only
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()  # Always positive
        )
        
        self.difficulty_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 4),  # Per-coordinate difficulty
            nn.Sigmoid()  # [0, 1] range
        )
        
        self.coverage_conditioner = nn.Sequential(
            nn.Linear(input_dim + 1, 128),  # +1 for target coverage
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights to encourage diversity."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        # Special initialization for output layers
        with torch.no_grad():
            # Size net should output moderate values
            self.size_net[-2].bias.data.fill_(0.5)
            
            # Difficulty net should start near middle of range
            self.difficulty_net[-2].bias.data.fill_(0.0)
            
            # Coverage conditioner should have varied outputs
            nn.init.normal_(self.coverage_conditioner[-1].weight, std=0.2)
            self.coverage_conditioner[-1].bias.data = torch.randn(4) * 0.5
        
    def forward(self, features, target_coverage=0.9):
        """
        Forward pass with coverage conditioning.
        
        Args:
            features: Input features [batch_size, 17]
            target_coverage: Target coverage level (used during training)
            
        Returns:
            widths: Predicted interval widths [batch_size, 4]
        """
        # Extract components
        box_coords = features[:, :4]
        
        # Size-based factor (larger objects = larger base width)
        size_factor = self.size_net(box_coords) + 1.0  # [1, ∞)
        
        # Difficulty assessment
        difficulty = self.difficulty_net(features) * 2.0 + 0.5  # [0.5, 2.5]
        
        # Coverage-conditioned base width
        coverage_tensor = torch.full((features.shape[0], 1), target_coverage, 
                                   device=features.device, dtype=features.dtype)
        coverage_input = torch.cat([features, coverage_tensor], dim=1)
        base_widths = torch.exp(self.coverage_conditioner(coverage_input))  # Positive values
        
        # Combine factors
        final_widths = base_widths * size_factor * difficulty
        
        # Add noise during training to prevent identical outputs
        if self.training:
            noise = torch.randn_like(final_widths) * 0.1 * final_widths.detach()
            final_widths = final_widths + noise.abs()
            
        return final_widths
    
    def get_config(self):
        """Return model configuration."""
        return {
            'model_type': 'AdaptiveMLPV2',
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims
        }