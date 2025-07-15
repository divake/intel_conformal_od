"""Conditional Quantile Network for adaptive conformal prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple


class ConditionalQuantileNetwork(nn.Module):
    """
    Conditional Quantile Network (CQN) for learning adaptive prediction intervals.
    
    Key differences from previous approaches:
    1. Directly optimizes quantile loss (pinball loss)
    2. Learns quantile functions conditioned on features
    3. Simpler architecture - single network, not decomposed
    4. Naturally handles coverage through quantile regression
    
    The model learns to predict the α/2 and 1-α/2 quantiles of the
    absolute error distribution, conditioned on input features.
    """
    
    def __init__(
        self,
        input_dim: int = 17,
        hidden_dims: list = None,
        dropout_rate: float = 0.1,
        num_quantiles: int = 2,  # Lower and upper quantiles
        base_quantile: float = 0.9  # For 90% coverage
    ):
        """
        Initialize CQN.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: Hidden layer dimensions
            dropout_rate: Dropout rate
            num_quantiles: Number of quantiles to predict (2 for intervals)
            base_quantile: Base coverage level (e.g., 0.9 for 90%)
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.num_quantiles = num_quantiles
        self.base_quantile = base_quantile
        
        # Build the network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output heads for each coordinate and quantile
        # 4 coordinates × 2 quantiles = 8 outputs
        self.quantile_heads = nn.ModuleList([
            nn.Linear(hidden_dims[-1], 4)  # 4 coordinates
            for _ in range(num_quantiles)
        ])
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    # Initialize biases to reasonable starting values
                    # This helps avoid initial collapse
                    nn.init.constant_(module.bias, 0.1)
    
    def forward(
        self, 
        features: torch.Tensor,
        quantile_levels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass to predict quantiles.
        
        Args:
            features: Input features [batch_size, input_dim]
            quantile_levels: Optional quantile levels to use
                           If None, uses [α/2, 1-α/2] for base_quantile
        
        Returns:
            Dictionary containing:
                - 'lower_quantiles': Lower quantile predictions [batch_size, 4]
                - 'upper_quantiles': Upper quantile predictions [batch_size, 4]
                - 'widths': Predicted interval widths [batch_size, 4]
        """
        # Extract features
        hidden = self.feature_extractor(features)
        
        # Compute quantile levels if not provided
        if quantile_levels is None:
            alpha = 1.0 - self.base_quantile
            lower_q = alpha / 2
            upper_q = 1.0 - alpha / 2
            quantile_levels = [lower_q, upper_q]
        
        # Get quantile predictions
        quantile_preds = []
        for i, head in enumerate(self.quantile_heads):
            # Raw predictions
            raw_pred = head(hidden)
            # Ensure positive values using softplus
            # Add small epsilon to prevent zero predictions
            pred = F.softplus(raw_pred) + 1e-3
            quantile_preds.append(pred)
        
        # Lower and upper quantiles
        lower_quantiles = quantile_preds[0]
        upper_quantiles = quantile_preds[1]
        
        # Ensure upper > lower by construction
        # This is critical for valid intervals
        upper_quantiles = lower_quantiles + F.softplus(upper_quantiles - lower_quantiles) + 1e-3
        
        # Compute widths (for symmetric intervals, this would be the same as upper)
        # But we keep both for flexibility
        widths = upper_quantiles
        
        return {
            'lower_quantiles': lower_quantiles,
            'upper_quantiles': upper_quantiles,
            'widths': widths
        }
    
    def predict_intervals(
        self,
        features: torch.Tensor,
        pred_boxes: torch.Tensor,
        coverage_level: float = 0.9
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict prediction intervals.
        
        Args:
            features: Input features [batch_size, input_dim]
            pred_boxes: Predicted boxes [batch_size, 4]
            coverage_level: Desired coverage level
            
        Returns:
            lower_bounds: Lower bounds of intervals [batch_size, 4]
            upper_bounds: Upper bounds of intervals [batch_size, 4]
        """
        # Get quantile predictions
        alpha = 1.0 - coverage_level
        outputs = self.forward(features, quantile_levels=[alpha/2, 1-alpha/2])
        
        # For symmetric intervals (which we're using)
        widths = outputs['widths']
        
        # Create intervals
        lower_bounds = pred_boxes - widths
        upper_bounds = pred_boxes + widths
        
        return lower_bounds, upper_bounds
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'model_type': 'ConditionalQuantileNetwork',
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'num_quantiles': self.num_quantiles,
            'base_quantile': self.base_quantile
        }


class QuantileLoss(nn.Module):
    """
    Pinball loss for quantile regression.
    
    This loss naturally encourages the model to learn proper quantiles
    of the error distribution, which directly translates to coverage.
    """
    
    def __init__(self, quantile_levels: list = None):
        """
        Initialize quantile loss.
        
        Args:
            quantile_levels: List of quantile levels to optimize
                           Default: [0.05, 0.95] for 90% coverage
        """
        super().__init__()
        
        if quantile_levels is None:
            quantile_levels = [0.05, 0.95]  # For 90% coverage
        
        self.quantile_levels = torch.tensor(quantile_levels)
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        quantile_level: float
    ) -> torch.Tensor:
        """
        Compute pinball loss.
        
        Args:
            predictions: Predicted quantile values
            targets: True values (errors in our case)
            quantile_level: The quantile being predicted
            
        Returns:
            Pinball loss value
        """
        errors = targets - predictions
        
        # Pinball loss
        loss = torch.where(
            errors >= 0,
            quantile_level * errors,
            (quantile_level - 1) * errors
        )
        
        return loss.mean()
    
    def compute_interval_loss(
        self,
        pred_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
        lower_quantiles: torch.Tensor,
        upper_quantiles: torch.Tensor,
        alpha: float = 0.1  # For 90% coverage
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for interval predictions.
        
        Args:
            pred_boxes: Predicted boxes [batch_size, 4]
            gt_boxes: Ground truth boxes [batch_size, 4]
            lower_quantiles: Lower quantile predictions [batch_size, 4]
            upper_quantiles: Upper quantile predictions [batch_size, 4]
            alpha: Miscoverage rate (1 - coverage)
            
        Returns:
            Dictionary with loss components
        """
        # Compute absolute errors
        errors = torch.abs(gt_boxes - pred_boxes)
        
        # Lower quantile loss (should capture α/2 quantile of errors)
        lower_loss = self.forward(lower_quantiles, errors, alpha / 2)
        
        # Upper quantile loss (should capture 1-α/2 quantile of errors)
        upper_loss = self.forward(upper_quantiles, errors, 1 - alpha / 2)
        
        # Total quantile loss
        quantile_loss = lower_loss + upper_loss
        
        # Interval width penalty (efficiency)
        # We want tight intervals, so penalize large widths
        widths = upper_quantiles  # For symmetric intervals
        width_penalty = widths.mean()
        
        # Coverage check (for monitoring, not optimization)
        with torch.no_grad():
            # Check if errors are within predicted quantiles
            covered = errors <= upper_quantiles
            coverage_rate = covered.float().mean()
        
        # Total loss
        lambda_width = 0.01  # Small weight for width penalty
        total_loss = quantile_loss + lambda_width * width_penalty
        
        return {
            'total': total_loss,
            'quantile_loss': quantile_loss,
            'width_penalty': width_penalty,
            'coverage_rate': coverage_rate,
            'avg_width': widths.mean(),
            'lower_loss': lower_loss,
            'upper_loss': upper_loss
        }