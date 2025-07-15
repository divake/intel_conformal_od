"""Size-aware loss function for symmetric adaptive conformal prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class SizeAwareSymmetricLoss(nn.Module):
    """
    Size-aware loss function that targets different coverage levels for different object sizes.
    
    Key insight: Small objects can afford higher coverage (90%) with minimal MPIW increase,
    while large objects should have lower coverage (85%) to significantly reduce MPIW.
    """
    
    def __init__(
        self,
        small_target_coverage: float = 0.90,   # Target 90% for small objects
        medium_target_coverage: float = 0.89,  # Target 89% for medium objects
        large_target_coverage: float = 0.85,   # Target 85% for large objects
        lambda_efficiency: float = 0.35,
        coverage_loss_type: str = 'smooth_l1',
        size_normalization: bool = True,
        # Size thresholds (based on sqrt of area)
        small_threshold: float = 32.0,    # Objects smaller than 32x32
        large_threshold: float = 96.0,    # Objects larger than 96x96
    ):
        """
        Initialize the size-aware loss function.
        
        Args:
            small_target_coverage: Target coverage for small objects
            medium_target_coverage: Target coverage for medium objects
            large_target_coverage: Target coverage for large objects
            lambda_efficiency: Weight for efficiency loss
            coverage_loss_type: Type of coverage loss
            size_normalization: Whether to normalize MPIW by object size
            small_threshold: Size threshold for small objects
            large_threshold: Size threshold for large objects
        """
        super().__init__()
        
        self.small_target_coverage = small_target_coverage
        self.medium_target_coverage = medium_target_coverage
        self.large_target_coverage = large_target_coverage
        self.lambda_efficiency = lambda_efficiency
        self.coverage_loss_type = coverage_loss_type
        self.size_normalization = size_normalization
        self.small_threshold = small_threshold
        self.large_threshold = large_threshold
        
        # For smooth L1 and Huber losses
        self.smooth_l1_beta = 1.0
        self.huber_delta = 1.0
        
    def get_size_category(self, gt_boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Categorize objects by size and return per-object target coverage.
        
        Returns:
            size_categories: 0=small, 1=medium, 2=large [batch_size]
            target_coverages: Target coverage for each object [batch_size]
        """
        # Compute box dimensions
        box_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
        box_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
        
        # Use sqrt of area as size metric
        box_sizes = torch.sqrt(box_widths * box_heights)
        
        # Initialize with medium category
        size_categories = torch.ones_like(box_sizes, dtype=torch.long)
        target_coverages = torch.full_like(box_sizes, self.medium_target_coverage)
        
        # Small objects
        small_mask = box_sizes < self.small_threshold
        size_categories[small_mask] = 0
        target_coverages[small_mask] = self.small_target_coverage
        
        # Large objects
        large_mask = box_sizes > self.large_threshold
        size_categories[large_mask] = 2
        target_coverages[large_mask] = self.large_target_coverage
        
        return size_categories, target_coverages
        
    def forward(
        self,
        pred_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
        predicted_widths: torch.Tensor,
        tau: float = 1.0,
        return_components: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the size-aware symmetric adaptive loss.
        """
        batch_size = pred_boxes.shape[0]
        device = pred_boxes.device
        
        # Get size categories and target coverages
        size_categories, target_coverages = self.get_size_category(gt_boxes)
        
        # Apply tau scaling to get calibrated widths
        scaled_widths = predicted_widths * tau
        
        # Form symmetric intervals
        lower_bounds = pred_boxes - scaled_widths
        upper_bounds = pred_boxes + scaled_widths
        
        # Compute coverage violations
        violations_lower = F.relu(gt_boxes - upper_bounds)
        violations_upper = F.relu(lower_bounds - gt_boxes)
        violations = violations_lower + violations_upper
        
        # Coverage loss based on selected type
        if self.coverage_loss_type == 'smooth_l1':
            coverage_loss_per_coord = F.smooth_l1_loss(
                violations,
                torch.zeros_like(violations),
                reduction='none',
                beta=self.smooth_l1_beta
            )
        elif self.coverage_loss_type == 'huber':
            coverage_loss_per_coord = F.huber_loss(
                violations,
                torch.zeros_like(violations),
                reduction='none',
                delta=self.huber_delta
            )
        else:  # 'mse'
            coverage_loss_per_coord = violations ** 2
        
        # Average across coordinates
        coverage_loss_per_box = coverage_loss_per_coord.mean(dim=1)  # [batch_size]
        
        # Weight coverage loss by size category
        # Give more weight to coverage violations for small objects (we want high coverage)
        # Give less weight to coverage violations for large objects (we allow lower coverage)
        coverage_weights = torch.ones_like(coverage_loss_per_box)
        coverage_weights[size_categories == 0] = 1.5  # Small objects - enforce coverage
        coverage_weights[size_categories == 2] = 0.7  # Large objects - allow flexibility
        
        coverage_loss = (coverage_loss_per_box * coverage_weights).mean()
        
        # Efficiency loss
        interval_widths = 2 * scaled_widths
        mpiw_per_box = interval_widths.mean(dim=1)
        
        if self.size_normalization:
            box_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
            box_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
            object_sizes = (box_widths + box_heights) / 2 + 1.0
            normalized_mpiw = mpiw_per_box / object_sizes
            
            # Size-specific efficiency weights
            # Penalize large objects more heavily for their MPIW
            efficiency_weights = torch.ones_like(normalized_mpiw)
            efficiency_weights[size_categories == 2] = 2.0  # Double penalty for large objects
            efficiency_weights[size_categories == 0] = 0.5  # Half penalty for small objects
            
            efficiency_loss = (normalized_mpiw * efficiency_weights).mean()
        else:
            efficiency_loss = mpiw_per_box.mean()
            normalized_mpiw = mpiw_per_box
        
        # Compute actual coverage rate per size category
        with torch.no_grad():
            covered_lower = gt_boxes >= lower_bounds
            covered_upper = gt_boxes <= upper_bounds
            covered = covered_lower & covered_upper
            box_covered = covered.all(dim=1).float()  # [batch_size]
            
            # Overall coverage
            coverage_rate = box_covered.mean()
            
            # Size-specific coverage rates
            small_coverage = box_covered[size_categories == 0].mean() if (size_categories == 0).any() else torch.tensor(0.0)
            medium_coverage = box_covered[size_categories == 1].mean() if (size_categories == 1).any() else torch.tensor(0.0)
            large_coverage = box_covered[size_categories == 2].mean() if (size_categories == 2).any() else torch.tensor(0.0)
        
        # Size-specific coverage penalties
        coverage_penalty = 0.0
        
        # Small objects penalty (want ~90% coverage)
        if (size_categories == 0).any() and small_coverage < 0.88:
            coverage_penalty += (0.90 - small_coverage) ** 2 * 5.0
        
        # Medium objects penalty (want ~89% coverage)
        if (size_categories == 1).any():
            if medium_coverage < 0.87:
                coverage_penalty += (0.89 - medium_coverage) ** 2 * 3.0
            elif medium_coverage > 0.91:
                coverage_penalty += (medium_coverage - 0.89) ** 2 * 3.0
        
        # Large objects penalty (want ~85% coverage)
        if (size_categories == 2).any() and large_coverage > 0.87:
            coverage_penalty += (large_coverage - 0.85) ** 2 * 5.0
        
        # Total loss
        total_loss = coverage_loss + self.lambda_efficiency * efficiency_loss + coverage_penalty
        
        # Additional statistics
        with torch.no_grad():
            avg_mpiw = mpiw_per_box.mean()
            avg_normalized_mpiw = normalized_mpiw.mean()
            
            # Size-specific MPIW
            small_mpiw = mpiw_per_box[size_categories == 0].mean() if (size_categories == 0).any() else torch.tensor(0.0)
            medium_mpiw = mpiw_per_box[size_categories == 1].mean() if (size_categories == 1).any() else torch.tensor(0.0)
            large_mpiw = mpiw_per_box[size_categories == 2].mean() if (size_categories == 2).any() else torch.tensor(0.0)
        
        # Return results
        result = {
            'total': total_loss,
            'coverage': coverage_loss,
            'efficiency': efficiency_loss,
            'coverage_penalty': torch.tensor(coverage_penalty),
            'coverage_rate': coverage_rate,
            'avg_mpiw': avg_mpiw,
            'normalized_mpiw': avg_normalized_mpiw,
            # Size-specific metrics
            'small_coverage': small_coverage,
            'medium_coverage': medium_coverage,
            'large_coverage': large_coverage,
            'small_mpiw': small_mpiw,
            'medium_mpiw': medium_mpiw,
            'large_mpiw': large_mpiw,
        }
        
        if return_components:
            result.update({
                'coord_coverage': covered.float().mean(dim=0),
                'avg_widths': scaled_widths.mean(dim=0),
                'min_widths': scaled_widths.min(dim=0)[0],
                'max_widths': scaled_widths.max(dim=0)[0]
            })
        
        return result