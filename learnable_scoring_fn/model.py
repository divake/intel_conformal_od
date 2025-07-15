import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class RegressionScoringFunction(nn.Module):
    """
    Regression-based scoring function for conformal prediction intervals.
    
    This network outputs the WIDTH of prediction intervals, not classification scores.
    The actual prediction interval is: pred ± (score * tau)
    """
    
    def __init__(self, input_dim: int = 17, hidden_dims: list = [256, 128, 64], 
                 dropout_rate: float = 0.15, activation: str = 'relu'):
        """
        Args:
            input_dim: Dimension of input features (13 geometric + 4 uncertainty features)
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability for regularization
            activation: Activation function type ('relu', 'elu', 'leaky_relu')
        """
        super(RegressionScoringFunction, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
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
        
        # Stored tau value for inference
        self.tau = None
        
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
        
        # Ensure positive width using softplus
        # Initialize to produce widths around 30-35 pixels for higher coverage
        # F.softplus(3.5) + 25.0 ≈ 35 pixels, which should give 90%+ coverage
        widths = F.softplus(raw_output + 3.5) + 25.0  # Start around ~35 pixels
        
        # Clamp to reasonable range for bounding box coordinates
        widths = torch.clamp(widths, min=5.0, max=100.0)
        
        return widths
    
    def set_tau(self, tau: torch.Tensor):
        """Store tau value for inference."""
        self.tau = tau
    
    def get_prediction_intervals(self, predictions: torch.Tensor, widths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate prediction intervals given predictions and learned widths.
        
        Args:
            predictions: [batch_size, 4] predicted bounding box coordinates
            widths: [batch_size, 1] learned interval widths from this model
            
        Returns:
            lower_bounds: [batch_size, 4] lower bounds of prediction intervals
            upper_bounds: [batch_size, 4] upper bounds of prediction intervals
        """
        if self.tau is None:
            raise ValueError("Must set tau using set_tau() before computing intervals")
        
        # Expand widths to match coordinate dimensions
        interval_widths = widths * self.tau  # [batch_size, 1]
        interval_widths = interval_widths.expand(-1, 4)  # [batch_size, 4]
        
        # Calculate intervals
        lower_bounds = predictions - interval_widths
        upper_bounds = predictions + interval_widths
        
        return lower_bounds, upper_bounds


class RegressionCoverageLoss(nn.Module):
    """
    Coverage loss for regression-based conformal prediction.
    
    This loss ensures that ground truth falls within the predicted intervals
    while minimizing interval width for efficiency.
    """
    
    def __init__(self, target_coverage: float = 0.9, efficiency_weight: float = 0.1,
                 calibration_weight: float = 0.05):
        """
        Args:
            target_coverage: Target coverage level (e.g., 0.9 for 90%)
            efficiency_weight: Weight for interval width penalty
            calibration_weight: Weight for calibration loss
        """
        super(RegressionCoverageLoss, self).__init__()
        self.target_coverage = target_coverage
        self.efficiency_weight = efficiency_weight
        self.calibration_weight = calibration_weight
    
    def forward(self, widths: torch.Tensor, gt_coords: torch.Tensor, 
                pred_coords: torch.Tensor, tau: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute regression coverage loss with CORRECT interval coverage.
        
        Args:
            widths: [batch_size, 1] predicted interval widths
            gt_coords: [batch_size, 4] ground truth coordinates
            pred_coords: [batch_size, 4] predicted coordinates
            tau: Current tau value (scalar or tensor)
            
        Returns:
            losses: Dictionary containing individual loss components
        """
        batch_size = widths.size(0)
        
        # Calculate actual errors
        errors = torch.abs(gt_coords - pred_coords)  # [batch_size, 4]
        
        # Calculate prediction interval bounds
        interval_half_widths = widths * tau  # [batch_size, 1]
        interval_half_widths_expanded = interval_half_widths.expand(-1, 4)  # [batch_size, 4]
        
        # CORRECT: Form prediction intervals
        lower_bounds = pred_coords - interval_half_widths_expanded
        upper_bounds = pred_coords + interval_half_widths_expanded
        
        # CORRECT: Check if ground truth falls within intervals
        # Coverage = 1 if gt is within [lower, upper], 0 otherwise
        covered_per_coord = (gt_coords >= lower_bounds) & (gt_coords <= upper_bounds)  # [batch_size, 4]
        
        # For bounding boxes: ALL coordinates must be covered
        sample_covered = covered_per_coord.all(dim=1).float()  # [batch_size]
        actual_coverage = sample_covered.mean()
        
        # 1. Coverage Loss - penalize under-coverage more than over-coverage
        coverage_error = actual_coverage - self.target_coverage
        if coverage_error < 0:  # Under-coverage
            coverage_loss = coverage_error ** 2 * 10.0  # Heavily penalize
        else:  # Over-coverage
            coverage_loss = coverage_error ** 2
        
        # 2. Efficiency Loss - directly minimize average interval width
        # No normalization by error - we want absolute efficiency
        efficiency_loss = widths.mean()
        
        # 3. Calibration Loss - encourage proportionality between widths and actual errors
        # Widths should be proportional to the expected error magnitude
        avg_errors_per_sample = errors.mean(dim=1, keepdim=True)  # [batch_size, 1]
        
        # Use correlation-based calibration loss
        # High correlation means widths adapt to error patterns
        error_mean = avg_errors_per_sample.mean()
        width_mean = widths.mean()
        
        error_centered = avg_errors_per_sample - error_mean
        width_centered = widths - width_mean
        
        covariance = (error_centered * width_centered).mean()
        error_std = error_centered.pow(2).mean().sqrt() + 1e-6
        width_std = width_centered.pow(2).mean().sqrt() + 1e-6
        
        correlation = covariance / (error_std * width_std)
        calibration_loss = 1.0 - correlation  # Want high correlation
        
        # Combine losses with adaptive weighting
        if actual_coverage < self.target_coverage - 0.3:  # Way under coverage (< 60%)
            # Heavily prioritize coverage, almost ignore efficiency
            total_loss = coverage_loss + 0.0001 * self.efficiency_weight * efficiency_loss
        elif actual_coverage < self.target_coverage - 0.1:  # Under coverage (< 80%)
            # Prioritize coverage, some efficiency
            total_loss = coverage_loss + 0.01 * self.efficiency_weight * efficiency_loss
        else:
            # Normal weighting
            total_loss = (coverage_loss + 
                         self.efficiency_weight * efficiency_loss +
                         self.calibration_weight * calibration_loss)
        
        # Return detailed losses for monitoring
        losses = {
            'total': total_loss,
            'coverage': coverage_loss,
            'efficiency': efficiency_loss,
            'calibration': calibration_loss,
            'actual_coverage': actual_coverage,
            'avg_width': widths.mean(),
            'correlation': correlation
        }
        
        return losses


def calculate_tau_regression(widths: torch.Tensor, errors: torch.Tensor, 
                            target_coverage: float = 0.9) -> torch.Tensor:
    """
    Calculate tau for regression conformal prediction without circular dependency.
    
    In the fixed approach, we use tau=1.0 and let the model learn appropriate widths.
    This avoids the circular dependency where tau depends on the widths being learned.
    
    Args:
        widths: [n_cal, 1] predicted interval widths from scoring function (not used)
        errors: [n_cal, 4] absolute errors between predictions and ground truth
        target_coverage: Desired coverage level
        
    Returns:
        tau: Fixed value of 1.0
    """
    # Use fixed tau = 1.0
    # The model will learn to output widths that achieve target coverage
    # when multiplied by tau = 1.0
    return torch.tensor(1.0, device=widths.device)


class UncertaintyFeatureExtractor:
    """
    Extract uncertainty-related features to augment geometric features.
    
    These features help the scoring function understand prediction uncertainty.
    """
    
    def __init__(self):
        self.error_stats = None
    
    def fit_error_distribution(self, train_errors: torch.Tensor):
        """
        Fit error distribution statistics from training data.
        
        Args:
            train_errors: [n_train, 4] training set errors
        """
        self.error_stats = {
            'mean': train_errors.mean(dim=0),
            'std': train_errors.std(dim=0),
            'quantiles': {
                'q25': torch.quantile(train_errors, 0.25, dim=0),
                'q50': torch.quantile(train_errors, 0.50, dim=0),
                'q75': torch.quantile(train_errors, 0.75, dim=0),
                'q90': torch.quantile(train_errors, 0.90, dim=0)
            }
        }
    
    def extract_uncertainty_features(self, pred_coords: torch.Tensor, 
                                   confidence: torch.Tensor,
                                   ensemble_std: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract uncertainty-related features.
        
        Args:
            pred_coords: [batch_size, 4] predicted coordinates
            confidence: [batch_size] confidence scores
            ensemble_std: [batch_size, 4] ensemble standard deviations (if available)
            
        Returns:
            features: [batch_size, 4] uncertainty features
        """
        batch_size = pred_coords.size(0)
        device = pred_coords.device
        
        features = []
        
        # 1. Confidence-based uncertainty (1 - confidence)
        uncertainty_score = (1 - confidence).unsqueeze(1)  # [batch_size, 1]
        features.append(uncertainty_score)
        
        # 2. Coordinate variance if ensemble predictions available
        if ensemble_std is not None:
            # Average ensemble uncertainty across coordinates
            avg_ensemble_uncertainty = ensemble_std.mean(dim=1, keepdim=True)  # [batch_size, 1]
            features.append(avg_ensemble_uncertainty)
        else:
            # Use confidence-based proxy
            features.append(uncertainty_score * 10.0)  # Scale up for similar magnitude
        
        # 3. Expected error based on confidence (learned mapping)
        if self.error_stats is not None:
            # Simple linear mapping from confidence to expected error
            expected_error = (1 - confidence) * self.error_stats['mean'].mean()
            features.append(expected_error.unsqueeze(1))
        else:
            features.append(uncertainty_score * 50.0)  # Default scaling
        
        # 4. Difficulty score based on box characteristics
        # Smaller boxes and extreme aspect ratios are typically harder
        width = pred_coords[:, 2] - pred_coords[:, 0]
        height = pred_coords[:, 3] - pred_coords[:, 1]
        area = width * height
        aspect_ratio = width / (height + 1e-6)
        
        # Normalize area (smaller boxes = higher difficulty)
        area_difficulty = 1.0 / (area + 1.0)
        
        # Aspect ratio difficulty (extreme ratios = higher difficulty)
        aspect_difficulty = torch.abs(torch.log(aspect_ratio + 1e-6))
        
        difficulty_score = (area_difficulty + aspect_difficulty) / 2
        features.append(difficulty_score.unsqueeze(1))
        
        # Concatenate all uncertainty features
        uncertainty_features = torch.cat(features, dim=1)  # [batch_size, 4]
        
        return uncertainty_features
    
    def extract_features(self, pred_coords: torch.Tensor, gt_coords: torch.Tensor, 
                        confidence: torch.Tensor, ensemble_std: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract uncertainty features based on predictions and ground truth.
        This is a wrapper that calls extract_uncertainty_features for compatibility.
        
        Args:
            pred_coords: [batch_size, 4] predicted coordinates
            gt_coords: [batch_size, 4] ground truth coordinates (not used currently)
            confidence: [batch_size] confidence scores
            ensemble_std: [batch_size, 4] ensemble standard deviations (if available)
            
        Returns:
            features: [batch_size, 4] uncertainty features
        """
        return self.extract_uncertainty_features(pred_coords, confidence, ensemble_std)


def save_regression_model(model: RegressionScoringFunction, optimizer: torch.optim.Optimizer,
                         epoch: int, losses: dict, tau: float, filepath: str,
                         feature_stats: dict = None, error_stats: dict = None):
    """Save regression model checkpoint with all necessary information."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'tau': tau,
        'model_config': {
            'input_dim': model.input_dim,
            'hidden_dims': model.hidden_dims,
            'dropout_rate': model.dropout_rate
        },
        'feature_stats': feature_stats,
        'error_stats': error_stats
    }
    torch.save(checkpoint, filepath)


def load_regression_model(filepath: str, device: torch.device = None) -> Tuple:
    """Load regression model checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint with map_location to handle device placement properly
    try:
        checkpoint = torch.load(filepath, map_location=device)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and device.type == "cuda":
            print(f"GPU out of memory while loading checkpoint, retrying on CPU...")
            device = torch.device("cpu")
            checkpoint = torch.load(filepath, map_location=device)
        else:
            raise
    
    # Get model type and config
    model_type = checkpoint.get('model_type', 'mlp')  # Default to mlp for backward compatibility
    model_config = checkpoint['model_config'].copy()  # Make a copy to avoid modifying original
    
    # Import factory to create the correct model type
    from learnable_scoring_fn.models.factory import create_model
    
    # Extract input_dim separately from config
    input_dim = model_config.pop('input_dim')
    
    # Create model using factory
    model = create_model(
        model_type=model_type,
        input_dim=input_dim,
        config=model_config
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Set tau if available
    if 'tau' in checkpoint:
        model.set_tau(checkpoint['tau'])
    
    return model, checkpoint