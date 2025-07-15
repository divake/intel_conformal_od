#!/usr/bin/env python
"""
Calibrate the CQN model to achieve target coverage.

This implements the recommended solution: Enhanced CQN with Calibration.
"""

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from scipy.optimize import minimize_scalar

# Add paths
import sys
sys.path.append(str(Path(__file__).parent.parent))

from learnable_scoring_fn.core_symmetric.models.cqn_model import ConditionalQuantileNetwork
from learnable_scoring_fn.core_symmetric.symmetric_adaptive import load_cached_data, prepare_splits


class CalibratedCQN:
    """CQN model with post-hoc calibration for target coverage."""
    
    def __init__(self, base_model, target_coverage=0.9):
        """
        Initialize calibrated model.
        
        Args:
            base_model: Trained CQN model
            target_coverage: Target coverage level
        """
        self.base_model = base_model
        self.target_coverage = target_coverage
        self.temperature = 1.0  # Calibration parameter
        
    def calibrate(self, calib_loader, device='cuda'):
        """
        Calibrate temperature parameter on calibration set.
        
        Args:
            calib_loader: DataLoader for calibration data
            device: Device to use
        """
        print(f"\nCalibrating for {self.target_coverage:.0%} coverage...")
        
        def compute_coverage(temperature):
            """Compute coverage for given temperature."""
            self.base_model.eval()
            coverages = []
            
            with torch.no_grad():
                for batch in calib_loader:
                    features = batch[0].to(device)
                    pred_boxes = batch[1].to(device)
                    gt_boxes = batch[2].to(device)
                    
                    # Get base predictions
                    outputs = self.base_model(features)
                    widths = outputs['widths']
                    
                    # Apply temperature scaling
                    calibrated_widths = widths * temperature
                    
                    # Check coverage
                    errors = torch.abs(gt_boxes - pred_boxes)
                    covered = (errors <= calibrated_widths).all(dim=1)
                    coverages.extend(covered.cpu().numpy())
            
            coverage = np.mean(coverages)
            return coverage
        
        # Binary search for optimal temperature
        def objective(temp):
            """Objective to minimize: squared error from target coverage."""
            coverage = compute_coverage(temp)
            return (coverage - self.target_coverage) ** 2
        
        # Find optimal temperature
        result = minimize_scalar(
            objective,
            bounds=(0.5, 3.0),
            method='bounded',
            options={'xatol': 0.01}
        )
        
        self.temperature = result.x
        final_coverage = compute_coverage(self.temperature)
        
        print(f"Calibration complete!")
        print(f"  Temperature: {self.temperature:.3f}")
        print(f"  Achieved coverage: {final_coverage:.1%}")
        
        return self.temperature, final_coverage
    
    def predict_intervals(self, features, pred_boxes):
        """
        Predict calibrated intervals.
        
        Args:
            features: Input features
            pred_boxes: Predicted boxes
            
        Returns:
            lower_bounds, upper_bounds
        """
        self.base_model.eval()
        
        with torch.no_grad():
            outputs = self.base_model(features)
            widths = outputs['widths']
            
            # Apply temperature calibration
            calibrated_widths = widths * self.temperature
            
            # Create intervals
            lower_bounds = pred_boxes - calibrated_widths
            upper_bounds = pred_boxes + calibrated_widths
            
        return lower_bounds, upper_bounds


def main():
    """Main calibration and evaluation."""
    
    # Load best CQN checkpoint
    print("Loading best CQN model...")
    checkpoint_path = Path("learnable_scoring_fn/experiment_tracking/checkpoints/cqn_best.pt")
    
    if not checkpoint_path.exists():
        print("Error: CQN checkpoint not found!")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Initialize model
    model = ConditionalQuantileNetwork(
        input_dim=17,
        hidden_dims=[256, 128, 64],
        dropout_rate=0.1,
        base_quantile=0.9
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Original coverage: {checkpoint['metrics']['coverage']:.1%}")
    
    # Load data
    print("\nLoading calibration data...")
    cache_dir = "/ssd_4TB/divake/conformal-od/learnable_scoring_fn/cache_base_model"
    raw_data = load_cached_data(cache_dir)
    
    val_data_dict = {
        'features': raw_data['val_features'],
        'pred_coords': raw_data['val_data']['pred_coords'],
        'gt_coords': raw_data['val_data']['gt_coords'],
        'confidence': raw_data['val_data']['confidence']
    }
    
    # Split into calibration and test
    cal_data, test_data = prepare_splits(val_data_dict, calib_fraction=0.5, seed=42)
    
    # Create data loaders
    cal_dataset = TensorDataset(
        cal_data['features'],
        cal_data['pred_coords'],
        cal_data['gt_coords']
    )
    cal_loader = DataLoader(cal_dataset, batch_size=256, shuffle=False)
    
    test_dataset = TensorDataset(
        test_data['features'],
        test_data['pred_coords'],
        test_data['gt_coords']
    )
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Create calibrated model
    calibrated_model = CalibratedCQN(model, target_coverage=0.9)
    
    # Calibrate
    temperature, cal_coverage = calibrated_model.calibrate(cal_loader, device)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_coverages = []
    test_mpws = []
    test_widths_all = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch[0].to(device)
            pred_boxes = batch[1].to(device)
            gt_boxes = batch[2].to(device)
            
            # Get calibrated predictions
            lower_bounds, upper_bounds = calibrated_model.predict_intervals(
                features, pred_boxes
            )
            
            # Check coverage
            errors = torch.abs(gt_boxes - pred_boxes)
            widths = upper_bounds - pred_boxes
            covered = (errors <= widths).all(dim=1)
            test_coverages.extend(covered.cpu().numpy())
            
            # MPIW
            mpiw = (upper_bounds - lower_bounds).mean(dim=1)
            test_mpws.extend(mpiw.cpu().numpy())
            test_widths_all.append(widths.cpu())
    
    # Final results
    test_coverage = np.mean(test_coverages)
    test_mpiw = np.mean(test_mpws)
    test_widths = torch.cat(test_widths_all, dim=0)
    test_width_std = test_widths.std().item()
    
    print("\n" + "="*60)
    print("CALIBRATED CQN - FINAL RESULTS")
    print("="*60)
    print(f"Test Coverage: {test_coverage:.1%}")
    print(f"Test MPIW: {test_mpiw:.1f} pixels")
    print(f"Width STD: {test_width_std:.2f}")
    print(f"Temperature: {temperature:.3f}")
    
    # Check size adaptivity
    print("\nSize-stratified results:")
    
    # Reload one batch for size analysis
    features, pred_boxes, gt_boxes = next(iter(test_loader))
    features = features.to(device)
    pred_boxes = pred_boxes.to(device)
    gt_boxes = gt_boxes.to(device)
    
    lower_bounds, upper_bounds = calibrated_model.predict_intervals(features, pred_boxes)
    mpiw = (upper_bounds - lower_bounds).mean(dim=1)
    
    # Compute areas
    box_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
    box_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
    areas = box_widths * box_heights
    
    # Size bins
    small_mask = areas < 32**2
    medium_mask = (areas >= 32**2) & (areas < 96**2)
    large_mask = areas >= 96**2
    
    if small_mask.any():
        print(f"  Small objects: MPIW = {mpiw[small_mask].mean():.1f}")
    if medium_mask.any():
        print(f"  Medium objects: MPIW = {mpiw[medium_mask].mean():.1f}")
    if large_mask.any():
        print(f"  Large objects: MPIW = {mpiw[large_mask].mean():.1f}")
    
    # Success check
    if 0.88 <= test_coverage <= 0.92 and test_width_std > 5.0:
        print("\n🎉 SUCCESS! Target achieved with calibrated CQN! 🎉")
        
        # Save calibrated model
        save_path = Path("learnable_scoring_fn/experiment_tracking/checkpoints/cqn_calibrated_success.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'temperature': temperature,
            'test_metrics': {
                'coverage': test_coverage,
                'mpiw': test_mpiw,
                'width_std': test_width_std
            },
            'config': model.get_config()
        }, save_path)
        
        print(f"\nCalibrated model saved to: {save_path}")
    else:
        print("\nClose but not quite at target. Consider:")
        print("- Fine-tuning temperature manually")
        print("- Using isotonic regression instead")
        print("- Trying different calibration methods")
    
    print("="*60)


if __name__ == "__main__":
    main()