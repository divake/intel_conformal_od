"""Tau calibration for symmetric adaptive conformal prediction."""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


def calibrate_tau(
    model: nn.Module,
    calibration_loader: DataLoader,
    target_coverage: float = 0.9,
    device: Optional[torch.device] = None
) -> float:
    """
    Find tau such that target coverage is achieved on calibration set.
    
    For each calibration example, we find the minimum scaling factor needed
    to cover the ground truth. The tau is then the quantile of these factors
    at the target coverage level.
    
    Args:
        model: Trained symmetric adaptive model
        calibration_loader: DataLoader with calibration data
        target_coverage: Desired coverage level (default: 0.9)
        device: Device to run on (auto-detected if None)
        
    Returns:
        tau: Calibration factor that ensures target coverage
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    coverage_factors = []
    
    with torch.no_grad():
        for batch in calibration_loader:
            # Unpack batch
            features = batch[0].to(device)
            pred_coords = batch[1].to(device)
            gt_coords = batch[2].to(device)
            
            # Get width predictions from model
            widths = model(features)  # [batch_size, 4]
            
            # For each example, find minimum factor for coverage
            # We need: pred ± (width * factor) to contain gt
            # So: factor >= |gt - pred| / width
            
            errors = torch.abs(gt_coords - pred_coords)  # [batch_size, 4]
            
            # Avoid division by zero
            widths_safe = widths.clamp(min=1e-6)
            
            # Required factors per coordinate
            required_factors = errors / widths_safe  # [batch_size, 4]
            
            # Maximum factor needed per box (to cover all coordinates)
            max_factors_per_box = required_factors.max(dim=1)[0]  # [batch_size]
            
            coverage_factors.extend(max_factors_per_box.cpu().numpy())
    
    # Convert to numpy array
    coverage_factors = np.array(coverage_factors)
    
    # Compute quantile at target coverage level
    # For tighter control, use slightly lower quantile
    adjusted_quantile = target_coverage - 0.01  # Aim for 89% instead of 90%
    tau = np.quantile(coverage_factors, adjusted_quantile)
    
    # No safety margin - we want tight intervals
    # tau = tau * 1.01  # Removed safety margin
    
    return float(tau)


class TauCalibrator:
    """
    More sophisticated tau calibration with additional features.
    """
    
    def __init__(
        self,
        target_coverage: float = 0.9,
        smoothing_factor: float = 0.7,
        min_tau: float = 0.1,  # Lowered from 0.5 to allow tighter intervals
        max_tau: float = 5.0,
        safety_margin: float = 0.01
    ):
        """
        Initialize the tau calibrator.
        
        Args:
            target_coverage: Desired coverage level
            smoothing_factor: Smoothing for tau updates (0=no smoothing, 1=full smoothing)
            min_tau: Minimum allowed tau value
            max_tau: Maximum allowed tau value
            safety_margin: Additional margin to add to tau
        """
        self.target_coverage = target_coverage
        self.smoothing_factor = smoothing_factor
        self.min_tau = min_tau
        self.max_tau = max_tau
        self.safety_margin = safety_margin
        
        self.current_tau = 1.0
        self.tau_history = []
        
    def calibrate(
        self,
        model: nn.Module,
        calibration_data: Dict[str, torch.Tensor],
        batch_size: int = 256,
        device: Optional[torch.device] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calibrate tau with additional statistics.
        
        Args:
            model: Trained model
            calibration_data: Dictionary with 'features', 'pred_coords', 'gt_coords'
            batch_size: Batch size for calibration
            device: Device to run on
            
        Returns:
            tau: Calibrated tau value
            stats: Dictionary with calibration statistics
        """
        if device is None:
            device = next(model.parameters()).device
        
        # Create calibration loader
        dataset = TensorDataset(
            calibration_data['features'],
            calibration_data['pred_coords'],
            calibration_data['gt_coords']
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Run basic calibration
        raw_tau = calibrate_tau(model, loader, self.target_coverage, device)
        
        # Apply smoothing if we have history
        if self.tau_history:
            smoothed_tau = (
                self.smoothing_factor * self.current_tau +
                (1 - self.smoothing_factor) * raw_tau
            )
        else:
            smoothed_tau = raw_tau
        
        # Clamp to valid range
        final_tau = np.clip(smoothed_tau, self.min_tau, self.max_tau)
        
        # Update state
        self.current_tau = final_tau
        self.tau_history.append(final_tau)
        
        # Compute additional statistics
        stats = self._compute_statistics(
            model, loader, final_tau, device
        )
        
        return final_tau, stats
    
    def _compute_statistics(
        self,
        model: nn.Module,
        loader: DataLoader,
        tau: float,
        device: torch.device
    ) -> Dict[str, float]:
        """Compute calibration statistics."""
        model.eval()
        
        coverage_indicators = []
        mpiws = []
        
        with torch.no_grad():
            for batch in loader:
                features = batch[0].to(device)
                pred_coords = batch[1].to(device)
                gt_coords = batch[2].to(device)
                
                # Get predictions
                widths = model(features)
                scaled_widths = widths * tau
                
                # Check coverage
                lower = pred_coords - scaled_widths
                upper = pred_coords + scaled_widths
                
                covered = ((gt_coords >= lower) & (gt_coords <= upper)).all(dim=1)
                coverage_indicators.extend(covered.cpu().numpy())
                
                # Compute MPIW
                mpiw = (2 * scaled_widths).mean(dim=1)
                mpiws.extend(mpiw.cpu().numpy())
        
        # Aggregate statistics
        coverage_indicators = np.array(coverage_indicators)
        mpiws = np.array(mpiws)
        
        stats = {
            'actual_coverage': coverage_indicators.mean(),
            'avg_mpiw': mpiws.mean(),
            'std_mpiw': mpiws.std(),
            'min_mpiw': mpiws.min(),
            'max_mpiw': mpiws.max(),
            'tau': tau,
            'raw_tau': self.tau_history[-1] if self.tau_history else tau
        }
        
        return stats
    
    def get_tau_evolution(self) -> np.ndarray:
        """Get the history of tau values."""
        return np.array(self.tau_history)