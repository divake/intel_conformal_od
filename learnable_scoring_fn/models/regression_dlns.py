"""Regression DLNs (Differentiable Logic Networks) for interval prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List
from .base import BaseScoringFunction


class RegressionDLNScoringFunction(BaseScoringFunction):
    """Differentiable Logic Networks adapted for regression.
    
    DLNs learn logical rules over binarized features and combine them
    for prediction. This implementation adapts DLNs for continuous output.
    """
    
    def __init__(self, input_dim: int = 17, n_logic_layers: int = 2,
                 neurons_per_layer: List[int] = [32, 16], temperature: float = 0.5,
                 n_thresholds: int = 3, use_residual: bool = True,
                 scoring_strategy: str = 'direct', output_constraint: str = 'natural'):
        """
        Args:
            input_dim: Number of input features
            n_logic_layers: Number of logic layers
            neurons_per_layer: Number of neurons in each logic layer
            temperature: Temperature for differentiable operations
            n_thresholds: Number of thresholds per feature for binarization
            use_residual: Whether to use residual connections
                    scoring_strategy: 'legacy' or 'direct' for adaptive scoring
            output_constraint: 'legacy', 'natural', or 'unconstrained'
        """
        super().__init__(input_dim, scoring_strategy, output_constraint)
        
        self.n_logic_layers = n_logic_layers
        self.neurons_per_layer = neurons_per_layer
        self.temperature = temperature
        self.n_thresholds = n_thresholds
        self.use_residual = use_residual
        
        # Multi-threshold binarization layer
        self.threshold_layer = MultiThresholdBinarization(
            input_dim, n_thresholds, temperature
        )
        
        # Total binary features after thresholding
        binary_dim = input_dim * n_thresholds
        
        # Build logic layers
        self.logic_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        prev_size = binary_dim
        
        for i, n_neurons in enumerate(neurons_per_layer):
            layer = DifferentiableLogicLayer(
                prev_size, n_neurons, temperature, use_residual=(i > 0 and use_residual)
            )
            self.logic_layers.append(layer)
            self.layer_norms.append(nn.LayerNorm(n_neurons))
            prev_size = n_neurons
        
        # Continuous value estimation from logic outputs
        # Use a small MLP to map from logic space to continuous output
        self.value_estimator = nn.Sequential(
            nn.Linear(neurons_per_layer[-1], neurons_per_layer[-1] * 2),
            nn.LayerNorm(neurons_per_layer[-1] * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(neurons_per_layer[-1] * 2, neurons_per_layer[-1]),
            nn.ReLU(),
            nn.Linear(neurons_per_layer[-1], 1)
        )
        
        # Optional: Direct path from input to output for residual learning
        if use_residual:
            self.direct_path = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights appropriately."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Regression DLNs.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            widths: Predicted interval widths [batch_size, 1]
        """
        batch_size = x.size(0)
        
        # Multi-threshold binarization
        binary_features = self.threshold_layer(x)  # [batch_size, input_dim * n_thresholds]
        
        # Process through logic layers
        h = binary_features
        logic_outputs = []
        
        for i, (layer, norm) in enumerate(zip(self.logic_layers, self.layer_norms)):
            h = layer(h)
            h = norm(h)
            logic_outputs.append(h)
        
        # Use final logic layer output for value estimation
        final_logic = logic_outputs[-1]
        
        # Estimate continuous value from logic outputs
        raw_output = self.value_estimator(final_logic)
        
        # Optional: Add direct path
        if self.use_residual:
            direct_output = self.direct_path(x)
            raw_output = raw_output + 0.1 * direct_output  # Small weight on direct path
        
        # Ensure positive output
        return self.ensure_positive_output(raw_output)
    
    def get_config(self) -> Dict[str, Any]:
        """Return model configuration."""
        return {
            'n_logic_layers': self.n_logic_layers,
            'neurons_per_layer': self.neurons_per_layer,
            'temperature': self.temperature,
            'n_thresholds': self.n_thresholds,
            'use_residual': self.use_residual,
            'scoring_strategy': self.scoring_strategy,
            'output_constraint': self.output_constraint
        }
    
    @property
    def model_name(self) -> str:
        """Return model name."""
        return "Regression-DLNs"


class MultiThresholdBinarization(nn.Module):
    """Learnable multi-threshold binarization of continuous features."""
    
    def __init__(self, input_dim: int, n_thresholds: int, temperature: float):
        super().__init__()
        self.input_dim = input_dim
        self.n_thresholds = n_thresholds
        self.temperature = temperature
        
        # Learnable thresholds for each feature
        # Initialize thresholds to span reasonable range
        self.thresholds = nn.Parameter(torch.zeros(input_dim, n_thresholds))
        
        # Learnable scaling factors for each feature
        self.scales = nn.Parameter(torch.ones(input_dim))
        
        # Initialize thresholds to percentiles
        self._init_thresholds()
    
    def _init_thresholds(self):
        """Initialize thresholds to span the feature range."""
        with torch.no_grad():
            # Initialize to equally spaced values
            for i in range(self.input_dim):
                percentiles = torch.linspace(-1, 1, self.n_thresholds)
                self.thresholds[i] = percentiles
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-threshold binarization.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            binary_features: Binarized features [batch_size, input_dim * n_thresholds]
        """
        batch_size = x.size(0)
        binary_features = []
        
        for i in range(self.input_dim):
            # Scale feature
            scaled_feature = x[:, i:i+1] * self.scales[i]  # [batch_size, 1]
            
            # Compare against multiple thresholds
            thresholds = self.thresholds[i].unsqueeze(0)  # [1, n_thresholds]
            
            # Soft binarization using sigmoid
            differences = scaled_feature - thresholds  # [batch_size, n_thresholds]
            binary_vals = torch.sigmoid(differences / self.temperature)
            
            binary_features.append(binary_vals)
        
        # Concatenate all binary features
        return torch.cat(binary_features, dim=1)  # [batch_size, input_dim * n_thresholds]


class DifferentiableLogicLayer(nn.Module):
    """Single differentiable logic layer."""
    
    def __init__(self, input_size: int, output_size: int, temperature: float, 
                 use_residual: bool = False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.temperature = temperature
        self.use_residual = use_residual
        
        # Learnable logic function selection (conjunctions/disjunctions)
        self.conjunction_weights = nn.Parameter(torch.randn(output_size, input_size))
        self.disjunction_weights = nn.Parameter(torch.randn(output_size, input_size))
        
        # Negation weights (which inputs to negate)
        self.negation_weights = nn.Parameter(torch.zeros(output_size, input_size))
        
        # Output combination weights
        self.combination_weights = nn.Parameter(torch.ones(output_size, 2) * 0.5)
        
        # Optional residual connection
        if use_residual and input_size == output_size:
            self.residual_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through logic layer.
        
        Args:
            x: Binary features [batch_size, input_size]
            
        Returns:
            output: Logic layer output [batch_size, output_size]
        """
        batch_size = x.size(0)
        
        # Apply learnable negations
        negation_probs = torch.sigmoid(self.negation_weights / self.temperature)
        # x: [batch_size, input_size]
        # negation_probs: [output_size, input_size]
        x_expanded = x.unsqueeze(1).expand(-1, self.output_size, -1)  # [batch_size, output_size, input_size]
        negation_probs_expanded = negation_probs.unsqueeze(0)  # [1, output_size, input_size]
        x_negated = x_expanded * (1 - negation_probs_expanded) + (1 - x_expanded) * negation_probs_expanded
        
        # Compute weighted conjunctions (AND operations)
        conj_weights = F.softmax(self.conjunction_weights / self.temperature, dim=1)  # [output_size, input_size]
        # Soft AND: weighted geometric mean
        log_inputs = torch.log(x_negated + 1e-8)  # [batch_size, output_size, input_size]
        weighted_log_inputs = log_inputs * conj_weights.unsqueeze(0)  # [batch_size, output_size, input_size]
        conjunctions = torch.exp(weighted_log_inputs.sum(dim=2))  # [batch_size, output_size]
        
        # Compute weighted disjunctions (OR operations)
        disj_weights = F.softmax(self.disjunction_weights / self.temperature, dim=1)  # [output_size, input_size]
        # Soft OR: 1 - weighted geometric mean of (1 - inputs)
        log_complements = torch.log(1 - x_negated + 1e-8)  # [batch_size, output_size, input_size]
        weighted_log_complements = log_complements * disj_weights.unsqueeze(0)  # [batch_size, output_size, input_size]
        disjunctions = 1 - torch.exp(weighted_log_complements.sum(dim=2))  # [batch_size, output_size]
        
        # Combine conjunctions and disjunctions
        combo_weights = F.softmax(self.combination_weights / self.temperature, dim=1)  # [output_size, 2]
        outputs = conjunctions * combo_weights[:, 0] + disjunctions * combo_weights[:, 1]  # [batch_size, output_size]
        
        # Apply residual connection if applicable
        if self.use_residual and hasattr(self, 'residual_weight'):
            outputs = outputs + self.residual_weight * x
        
        # Ensure outputs are in [0, 1] range
        outputs = torch.clamp(outputs, 0, 1)
        
        return outputs