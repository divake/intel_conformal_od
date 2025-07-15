#!/usr/bin/env python
"""
Verify CQN reproducibility by running calibrated CQN 5 times with different seeds.
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

from learnable_scoring_fn.core_symmetric.models.cqn_model import ConditionalQuantileNetwork, QuantileLoss
from learnable_scoring_fn.train_cqn import train_cqn, evaluate_model
from learnable_scoring_fn.calibrate_cqn import CalibratedCQN
from learnable_scoring_fn.core_symmetric.symmetric_adaptive import load_cached_data, prepare_splits
from torch.utils.data import DataLoader, TensorDataset


def run_single_experiment(seed, run_idx):
    """Run a single CQN training and calibration experiment."""
    
    print(f"\n{'='*60}")
    print(f"Run {run_idx}/5 - Seed: {seed}")
    print(f"{'='*60}")
    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Train model
    model, best_metrics, history = train_cqn(
        epochs=100,
        batch_size=256,
        learning_rate=0.001,
        target_coverage=0.9,
        device=device
    )
    
    # Load data for calibration
    cache_dir = "/ssd_4TB/divake/conformal-od/learnable_scoring_fn/cache_base_model"
    raw_data = load_cached_data(cache_dir)
    
    val_data_dict = {
        'features': raw_data['val_features'],
        'pred_coords': raw_data['val_data']['pred_coords'],
        'gt_coords': raw_data['val_data']['gt_coords'],
        'confidence': raw_data['val_data']['confidence']
    }
    
    # Split into calibration and test
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
    
    # Get uncalibrated test metrics
    uncalibrated_metrics = evaluate_model(model, test_loader, 0.9, device)
    
    # Calibrate model
    calibrated_model = CalibratedCQN(model, target_coverage=0.9)
    temperature, cal_coverage = calibrated_model.calibrate(cal_loader, device)
    
    # Evaluate calibrated model on test set
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
        'uncalibrated': {
            'coverage': uncalibrated_metrics['coverage'],
            'mpiw': uncalibrated_metrics['mpiw'],
            'width_std': uncalibrated_metrics['width_std']
        },
        'calibrated': {
            'coverage': test_coverage,
            'mpiw': test_mpiw,
            'width_std': test_width_std,
            'temperature': temperature
        },
        'size_mpws': size_mpws,
        'training_history': {
            'best_val_coverage': best_metrics['coverage']
        }
    }
    
    return results


def main():
    """Run reproducibility verification."""
    
    print("\n" + "="*80)
    print("CQN REPRODUCIBILITY VERIFICATION")
    print("Running 5 experiments with different seeds")
    print("="*80)
    
    # Different seeds to test
    seeds = [42, 123, 456, 789, 2024]
    
    # Store all results
    all_results = []
    
    # Run experiments
    for idx, seed in enumerate(seeds, 1):
        results = run_single_experiment(seed, idx)
        all_results.append(results)
        
        # Print summary for this run
        print(f"\nRun {idx} Summary:")
        print(f"  Uncalibrated coverage: {results['uncalibrated']['coverage']:.1%}")
        print(f"  Calibrated coverage: {results['calibrated']['coverage']:.1%}")
        print(f"  Temperature: {results['calibrated']['temperature']:.3f}")
        print(f"  MPIW: {results['calibrated']['mpiw']:.1f}")
    
    # Compute statistics across runs
    print("\n" + "="*80)
    print("REPRODUCIBILITY SUMMARY")
    print("="*80)
    
    # Extract metrics
    uncalib_coverages = [r['uncalibrated']['coverage'] for r in all_results]
    calib_coverages = [r['calibrated']['coverage'] for r in all_results]
    temperatures = [r['calibrated']['temperature'] for r in all_results]
    mpws = [r['calibrated']['mpiw'] for r in all_results]
    width_stds = [r['calibrated']['width_std'] for r in all_results]
    
    print("\nUncalibrated Results:")
    print(f"  Coverage: {np.mean(uncalib_coverages):.1%} ± {np.std(uncalib_coverages):.1%}")
    print(f"  Range: [{np.min(uncalib_coverages):.1%}, {np.max(uncalib_coverages):.1%}]")
    
    print("\nCalibrated Results:")
    print(f"  Coverage: {np.mean(calib_coverages):.1%} ± {np.std(calib_coverages):.1%}")
    print(f"  Range: [{np.min(calib_coverages):.1%}, {np.max(calib_coverages):.1%}]")
    print(f"  Temperature: {np.mean(temperatures):.3f} ± {np.std(temperatures):.3f}")
    print(f"  MPIW: {np.mean(mpws):.1f} ± {np.std(mpws):.1f}")
    print(f"  Width STD: {np.mean(width_stds):.1f} ± {np.std(width_stds):.1f}")
    
    # Size-specific results
    print("\nSize-Adaptive Behavior (average across runs):")
    for size in ['small', 'medium', 'large']:
        size_mpws = [r['size_mpws'][size] for r in all_results if r['size_mpws'][size] > 0]
        if size_mpws:
            print(f"  {size.capitalize()} objects: {np.mean(size_mpws):.1f} ± {np.std(size_mpws):.1f}")
    
    # Check consistency
    coverage_range = np.max(calib_coverages) - np.min(calib_coverages)
    temp_cv = np.std(temperatures) / np.mean(temperatures)  # Coefficient of variation
    
    print("\nConsistency Metrics:")
    print(f"  Coverage range: {coverage_range:.1%}")
    print(f"  Temperature CV: {temp_cv:.1%}")
    
    if coverage_range <= 0.02 and temp_cv <= 0.05:  # Within 2% coverage and 5% temperature variation
        print("\n✅ EXCELLENT REPRODUCIBILITY: Results are highly consistent across seeds!")
    elif coverage_range <= 0.05 and temp_cv <= 0.10:
        print("\n✅ GOOD REPRODUCIBILITY: Results are reasonably consistent across seeds.")
    else:
        print("\n⚠️  MODERATE REPRODUCIBILITY: Some variation observed across seeds.")
    
    # Save detailed results
    results_path = Path("learnable_scoring_fn/experiment_tracking/cqn_reproducibility_results.json")
    results_path.parent.mkdir(exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'uncalibrated': {
                    'mean_coverage': float(np.mean(uncalib_coverages)),
                    'std_coverage': float(np.std(uncalib_coverages)),
                    'min_coverage': float(np.min(uncalib_coverages)),
                    'max_coverage': float(np.max(uncalib_coverages))
                },
                'calibrated': {
                    'mean_coverage': float(np.mean(calib_coverages)),
                    'std_coverage': float(np.std(calib_coverages)),
                    'min_coverage': float(np.min(calib_coverages)),
                    'max_coverage': float(np.max(calib_coverages)),
                    'mean_temperature': float(np.mean(temperatures)),
                    'std_temperature': float(np.std(temperatures)),
                    'mean_mpiw': float(np.mean(mpws)),
                    'std_mpiw': float(np.std(mpws))
                }
            },
            'detailed_results': all_results
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_path}")
    print("="*80)


if __name__ == "__main__":
    main()