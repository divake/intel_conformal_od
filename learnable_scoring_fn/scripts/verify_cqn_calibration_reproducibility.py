#!/usr/bin/env python
"""
Verify CQN calibration reproducibility using the existing trained model.
Tests if temperature calibration is stable across different data splits.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Add paths
sys.path.append(str(Path(__file__).parent))

from learnable_scoring_fn.core_symmetric.models.cqn_model import ConditionalQuantileNetwork
from learnable_scoring_fn.calibrate_cqn import CalibratedCQN
from learnable_scoring_fn.core_symmetric.symmetric_adaptive import load_cached_data, prepare_splits
from torch.utils.data import DataLoader, TensorDataset


def run_single_calibration(seed, run_idx, model, raw_data):
    """Run a single calibration experiment with different data split."""
    
    print(f"\n{'='*60}")
    print(f"Run {run_idx}/5 - Seed: {seed}")
    print(f"{'='*60}")
    
    # Set random seed for data split
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Prepare data with different seed
    val_data_dict = {
        'features': raw_data['val_features'],
        'pred_coords': raw_data['val_data']['pred_coords'],
        'gt_coords': raw_data['val_data']['gt_coords'],
        'confidence': raw_data['val_data']['confidence']
    }
    
    # Split into calibration and test with seed
    cal_data, test_data = prepare_splits(val_data_dict, calib_fraction=0.5, seed=seed)
    
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
    
    # Calibrate model
    calibrated_model = CalibratedCQN(model, target_coverage=0.9)
    temperature, cal_coverage = calibrated_model.calibrate(cal_loader, device)
    
    # Evaluate on test set
    test_coverages = []
    test_mpws = []
    test_widths_all = []
    size_results = {'small': [], 'medium': [], 'large': []}
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch[0].to(device)
            pred_boxes = batch[1].to(device)
            gt_boxes = batch[2].to(device)
            
            # Get calibrated predictions
            lower_bounds, upper_bounds = calibrated_model.predict_intervals(features, pred_boxes)
            
            # Check coverage
            errors = torch.abs(gt_boxes - pred_boxes)
            widths = upper_bounds - pred_boxes
            covered = (errors <= widths).all(dim=1)
            test_coverages.extend(covered.cpu().numpy())
            
            # MPIW
            mpiw = (upper_bounds - lower_bounds).mean(dim=1)
            test_mpws.extend(mpiw.cpu().numpy())
            test_widths_all.append(widths.cpu())
            
            # Size-specific results
            box_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
            box_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
            areas = box_widths * box_heights
            
            small_mask = areas < 32**2
            medium_mask = (areas >= 32**2) & (areas < 96**2)
            large_mask = areas >= 96**2
            
            if small_mask.any():
                size_results['small'].extend(mpiw[small_mask].cpu().numpy())
            if medium_mask.any():
                size_results['medium'].extend(mpiw[medium_mask].cpu().numpy())
            if large_mask.any():
                size_results['large'].extend(mpiw[large_mask].cpu().numpy())
    
    # Compute final metrics
    test_coverage = np.mean(test_coverages)
    test_mpiw = np.mean(test_mpws)
    test_widths = torch.cat(test_widths_all, dim=0)
    test_width_std = test_widths.std().item()
    
    # Size-specific MPIWs
    size_mpws = {
        size: np.mean(mpws) if mpws else 0.0 
        for size, mpws in size_results.items()
    }
    
    # Collect results
    results = {
        'seed': seed,
        'calibration': {
            'temperature': temperature,
            'cal_coverage': cal_coverage
        },
        'test_results': {
            'coverage': test_coverage,
            'mpiw': test_mpiw,
            'width_std': test_width_std
        },
        'size_mpws': size_mpws
    }
    
    return results


def main():
    """Run calibration reproducibility verification."""
    
    print("\n" + "="*80)
    print("CQN CALIBRATION REPRODUCIBILITY VERIFICATION")
    print("Using existing trained CQN model")
    print("Testing calibration stability across different data splits")
    print("="*80)
    
    # Load best CQN checkpoint
    print("\nLoading best CQN model...")
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
    model.eval()  # Set to eval mode
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Original coverage: {checkpoint['metrics']['coverage']:.1%}")
    
    # Load data once
    print("\nLoading data...")
    cache_dir = "/ssd_4TB/divake/conformal-od/learnable_scoring_fn/cache_base_model"
    raw_data = load_cached_data(cache_dir)
    
    # Different seeds to test
    seeds = [42, 123, 456, 789, 2024]
    
    # Store all results
    all_results = []
    
    # Run experiments
    for idx, seed in enumerate(seeds, 1):
        results = run_single_calibration(seed, idx, model, raw_data)
        all_results.append(results)
        
        # Print summary for this run
        print(f"\nRun {idx} Summary:")
        print(f"  Temperature: {results['calibration']['temperature']:.3f}")
        print(f"  Calibration coverage: {results['calibration']['cal_coverage']:.1%}")
        print(f"  Test coverage: {results['test_results']['coverage']:.1%}")
        print(f"  MPIW: {results['test_results']['mpiw']:.1f}")
    
    # Compute statistics across runs
    print("\n" + "="*80)
    print("REPRODUCIBILITY SUMMARY")
    print("="*80)
    
    # Extract metrics
    temperatures = [r['calibration']['temperature'] for r in all_results]
    cal_coverages = [r['calibration']['cal_coverage'] for r in all_results]
    test_coverages = [r['test_results']['coverage'] for r in all_results]
    mpws = [r['test_results']['mpiw'] for r in all_results]
    width_stds = [r['test_results']['width_std'] for r in all_results]
    
    print("\nCalibration Results:")
    print(f"  Temperature: {np.mean(temperatures):.3f} ± {np.std(temperatures):.3f}")
    print(f"  Range: [{np.min(temperatures):.3f}, {np.max(temperatures):.3f}]")
    print(f"  Calibration Coverage: {np.mean(cal_coverages):.1%} ± {np.std(cal_coverages):.1%}")
    
    print("\nTest Results:")
    print(f"  Coverage: {np.mean(test_coverages):.1%} ± {np.std(test_coverages):.1%}")
    print(f"  Range: [{np.min(test_coverages):.1%}, {np.max(test_coverages):.1%}]")
    print(f"  MPIW: {np.mean(mpws):.1f} ± {np.std(mpws):.1f}")
    print(f"  Width STD: {np.mean(width_stds):.1f} ± {np.std(width_stds):.1f}")
    
    # Size-specific results
    print("\nSize-Adaptive Behavior (average across runs):")
    for size in ['small', 'medium', 'large']:
        size_mpws = [r['size_mpws'][size] for r in all_results if r['size_mpws'][size] > 0]
        if size_mpws:
            print(f"  {size.capitalize()} objects: {np.mean(size_mpws):.1f} ± {np.std(size_mpws):.1f}")
    
    # Check consistency
    temp_range = np.max(temperatures) - np.min(temperatures)
    coverage_range = np.max(test_coverages) - np.min(test_coverages)
    temp_cv = np.std(temperatures) / np.mean(temperatures)  # Coefficient of variation
    
    print("\nConsistency Metrics:")
    print(f"  Temperature range: {temp_range:.3f}")
    print(f"  Coverage range: {coverage_range:.1%}")
    print(f"  Temperature CV: {temp_cv:.1%}")
    
    if temp_range <= 0.1 and coverage_range <= 0.02:  # Within 0.1 temperature and 2% coverage
        print("\n✅ EXCELLENT REPRODUCIBILITY: Calibration is highly stable across different data splits!")
    elif temp_range <= 0.2 and coverage_range <= 0.05:
        print("\n✅ GOOD REPRODUCIBILITY: Calibration is reasonably stable across different data splits.")
    else:
        print("\n⚠️  MODERATE REPRODUCIBILITY: Some variation observed across different data splits.")
    
    # Save detailed results
    results_path = Path("learnable_scoring_fn/experiment_tracking/cqn_calibration_reproducibility.json")
    results_path.parent.mkdir(exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'checkpoint': str(checkpoint_path),
                'original_coverage': float(checkpoint['metrics']['coverage'])
            },
            'summary': {
                'temperatures': {
                    'mean': float(np.mean(temperatures)),
                    'std': float(np.std(temperatures)),
                    'min': float(np.min(temperatures)),
                    'max': float(np.max(temperatures)),
                    'range': float(temp_range),
                    'cv': float(temp_cv)
                },
                'test_coverage': {
                    'mean': float(np.mean(test_coverages)),
                    'std': float(np.std(test_coverages)),
                    'min': float(np.min(test_coverages)),
                    'max': float(np.max(test_coverages)),
                    'range': float(coverage_range)
                },
                'mpiw': {
                    'mean': float(np.mean(mpws)),
                    'std': float(np.std(mpws))
                }
            },
            'detailed_results': all_results
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_path}")
    print("="*80)


if __name__ == "__main__":
    main()