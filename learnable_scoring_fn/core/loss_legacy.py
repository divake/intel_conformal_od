"""Loss functions for learnable scoring functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


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


class AdaptiveCoverageLoss(nn.Module):
    """
    Adaptive coverage loss that properly balances coverage and efficiency.
    
    Key improvements over RegressionCoverageLoss:
    1. No adaptive weighting that kills efficiency optimization
    2. Includes ranking loss to ensure proper score ordering
    3. Variance penalty to encourage adaptive behavior
    4. Smooth coverage loss using quantile regression
    """
    
    def __init__(self, target_coverage: float = 0.9, 
                 efficiency_weight: float = 0.1,
                 ranking_weight: float = 0.05,
                 variance_weight: float = 0.02,
                 smoothness_weight: float = 0.01):
        """
        Args:
            target_coverage: Target coverage level (e.g., 0.9 for 90%)
            efficiency_weight: Weight for efficiency (score magnitude) penalty
            ranking_weight: Weight for ranking loss (ensures proper ordering)
            variance_weight: Weight for variance penalty (encourages adaptation)
            smoothness_weight: Weight for smoothness loss (stable predictions)
        """
        super().__init__()
        self.target_coverage = target_coverage
        self.efficiency_weight = efficiency_weight
        self.ranking_weight = ranking_weight
        self.variance_weight = variance_weight
        self.smoothness_weight = smoothness_weight
        
    def forward(self, scores: torch.Tensor, gt_coords: torch.Tensor, 
                pred_coords: torch.Tensor, features: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Compute adaptive coverage loss.
        
        Args:
            scores: Model predictions [batch_size, 1] or [batch_size, 4]
            gt_coords: Ground truth coordinates [batch_size, 4]
            pred_coords: Predicted coordinates [batch_size, 4]
            features: Input features for smoothness loss [batch_size, feature_dim]
            
        Returns:
            losses: Dictionary containing individual loss components
        """
        batch_size = scores.size(0)
        
        # Handle coordinate-specific scores
        if scores.dim() == 2 and scores.size(1) == 4:
            # Use mean for overall score
            score_mean = scores.mean(dim=1, keepdim=True)
        else:
            score_mean = scores
            
        # Calculate actual errors
        errors = torch.abs(gt_coords - pred_coords)  # [batch_size, 4]
        max_error = errors.max(dim=1, keepdim=True)[0]  # [batch_size, 1]
        
        # 1. Coverage Loss using smooth quantile regression (pinball loss)
        # This provides smoother gradients than hard coverage constraints
        quantile_level = self.target_coverage
        
        # For conformal prediction, we want P(error <= score) >= target_coverage
        # This translates to quantile regression at level target_coverage
        residuals = max_error - score_mean  # [batch_size, 1]
        
        # Pinball loss
        coverage_loss = torch.mean(
            torch.max(
                quantile_level * residuals,
                (quantile_level - 1) * residuals
            )
        )
        
        # 2. Efficiency Loss - minimize average score magnitude
        # But not too aggressively - we want some headroom
        efficiency_loss = score_mean.mean()
        
        # 3. Ranking Loss - ensure scores are ordered by difficulty
        if batch_size > 1:
            # Sample pairs for ranking loss
            n_pairs = min(batch_size * (batch_size - 1) // 2, 100)  # Limit pairs for efficiency
            
            # Random pairs
            idx1 = torch.randperm(batch_size)[:n_pairs]
            idx2 = torch.randperm(batch_size)[:n_pairs]
            
            # Ensure different indices
            mask = idx1 != idx2
            idx1 = idx1[mask]
            idx2 = idx2[mask]
            
            if len(idx1) > 0:
                errors1 = max_error[idx1]
                errors2 = max_error[idx2]
                scores1 = score_mean[idx1]
                scores2 = score_mean[idx2]
                
                # Where error1 > error2, we want score1 > score2
                # Use margin ranking loss
                margin = 0.1
                target = torch.sign(errors1 - errors2)
                ranking_loss = F.margin_ranking_loss(
                    scores1, scores2, target, margin=margin, reduction='mean'
                )
            else:
                ranking_loss = torch.tensor(0.0, device=scores.device)
        else:
            ranking_loss = torch.tensor(0.0, device=scores.device)
        
        # 4. Variance Penalty - encourage diverse scores, not constant output
        score_std = score_mean.std()
        score_mean_val = score_mean.mean()
        
        # Target coefficient of variation of at least 20%
        target_cv = 0.2
        target_std = target_cv * score_mean_val
        variance_loss = F.relu(target_std - score_std)
        
        # 5. Smoothness Loss - similar features should have similar scores
        if features is not None and batch_size > 1:
            # Ensure features and scores have correct dimensions for cdist
            if features.dim() == 1:
                features = features.unsqueeze(0)
            if score_mean.dim() == 1:
                score_mean = score_mean.unsqueeze(0)
                
            # Compute pairwise distances in feature space
            feature_dists = torch.cdist(features, features, p=2)
            score_dists = torch.cdist(score_mean, score_mean, p=2)
            
            # Normalize distances
            feature_dists = feature_dists / (feature_dists.mean() + 1e-6)
            score_dists = score_dists / (score_dists.mean() + 1e-6)
            
            # Mask out diagonal
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=features.device)
            feature_dists = feature_dists[mask]
            score_dists = score_dists[mask]
            
            # Smoothness loss - scores should vary smoothly with features
            smoothness_loss = F.mse_loss(score_dists, feature_dists)
        else:
            smoothness_loss = torch.tensor(0.0, device=scores.device)
        
        # Combine losses WITHOUT adaptive weighting
        total_loss = (
            coverage_loss + 
            self.efficiency_weight * efficiency_loss +
            self.ranking_weight * ranking_loss +
            self.variance_weight * variance_loss +
            self.smoothness_weight * smoothness_loss
        )
        
        # Calculate actual coverage for monitoring
        with torch.no_grad():
            covered = (max_error <= score_mean).float()
            actual_coverage = covered.mean()
        
        # Return detailed losses
        losses = {
            'total': total_loss,
            'coverage': coverage_loss,
            'efficiency': efficiency_loss,
            'ranking': ranking_loss,
            'variance': variance_loss,
            'smoothness': smoothness_loss,
            'actual_coverage': actual_coverage,
            'score_mean': score_mean.mean(),
            'score_std': score_std,
            'score_cv': score_std / (score_mean_val + 1e-6)
        }
        
        return losses