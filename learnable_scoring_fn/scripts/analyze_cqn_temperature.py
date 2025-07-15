#!/usr/bin/env python
"""
Analyze the relationship between temperature and coverage for CQN model.
Plots coverage vs temperature to understand calibration behavior.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import json

# Add paths
import sys
sys.path.append(str(Path(__file__).parent))

from learnable_scoring_fn.core_symmetric.models.cqn_model import ConditionalQuantileNetwork
from learnable_scoring_fn.core_symmetric.symmetric_adaptive import load_cached_data, prepare_splits


def compute_coverage_at_temperature(model, data_loader, temperature, device='cuda'):
    """Compute coverage for a given temperature value."""
    
    model.eval()
    coverages = []
    mpws = []
    
    with torch.no_grad():
        for batch in data_loader:
            features = batch[0].to(device)
            pred_boxes = batch[1].to(device)
            gt_boxes = batch[2].to(device)
            
            # Get base predictions
            outputs = model(features)
            widths = outputs['widths']
            
            # Apply temperature scaling
            calibrated_widths = widths * temperature
            
            # Check coverage
            errors = torch.abs(gt_boxes - pred_boxes)
            covered = (errors <= calibrated_widths).all(dim=1)
            coverages.extend(covered.cpu().numpy())
            
            # Compute MPIW
            mpiw = (2 * calibrated_widths).mean(dim=1)  # Total interval width
            mpws.extend(mpiw.cpu().numpy())
    
    coverage = np.mean(coverages)
    avg_mpiw = np.mean(mpws)
    
    return coverage, avg_mpiw


def analyze_temperature_coverage_relationship():
    """Analyze how temperature affects coverage."""
    
    print("\n" + "="*80)
    print("TEMPERATURE-COVERAGE RELATIONSHIP ANALYSIS")
    print("="*80)
    
    # Load best CQN checkpoint
    print("\nLoading CQN model...")
    checkpoint_path = Path("learnable_scoring_fn/experiment_tracking/checkpoints/cqn_best.pt")
    
    if not checkpoint_path.exists():
        print("Error: CQN checkpoint not found! Please train CQN first.")
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
    
    print(f"Model loaded successfully")
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
    
    # Use calibration split
    cal_data, _ = prepare_splits(val_data_dict, calib_fraction=0.5, seed=42)
    
    cal_dataset = TensorDataset(
        cal_data['features'],
        cal_data['pred_coords'],
        cal_data['gt_coords']
    )
    cal_loader = DataLoader(cal_dataset, batch_size=256, shuffle=False)
    
    # Test different temperature values
    temperatures = np.linspace(0.5, 3.0, 51)  # 51 points from 0.5 to 3.0
    coverages = []
    mpws = []
    
    print("\nAnalyzing temperature values...")
    for i, temp in enumerate(temperatures):
        coverage, mpiw = compute_coverage_at_temperature(model, cal_loader, temp, device)
        coverages.append(coverage)
        mpws.append(mpiw)
        
        if i % 10 == 0:
            print(f"  Temperature {temp:.2f}: Coverage = {coverage:.1%}, MPIW = {mpiw:.1f}")
    
    # Find optimal temperature for 90% coverage
    target_coverage = 0.9
    coverage_array = np.array(coverages)
    idx_closest = np.argmin(np.abs(coverage_array - target_coverage))
    optimal_temp = temperatures[idx_closest]
    optimal_coverage = coverages[idx_closest]
    optimal_mpiw = mpws[idx_closest]
    
    print(f"\nOptimal temperature for {target_coverage:.0%} coverage:")
    print(f"  Temperature: {optimal_temp:.3f}")
    print(f"  Achieved coverage: {optimal_coverage:.1%}")
    print(f"  MPIW: {optimal_mpiw:.1f}")
    
    # Create visualizations
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Coverage vs Temperature
    ax1.plot(temperatures, coverages, 'b-', linewidth=2)
    ax1.axhline(y=0.9, color='r', linestyle='--', label='Target (90%)')
    ax1.axvline(x=optimal_temp, color='g', linestyle='--', label=f'Optimal T={optimal_temp:.3f}')
    ax1.scatter([optimal_temp], [optimal_coverage], color='g', s=100, zorder=5)
    ax1.set_xlabel('Temperature', fontsize=12)
    ax1.set_ylabel('Coverage', fontsize=12)
    ax1.set_title('Coverage vs Temperature', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([0.0, 1.0])
    
    # Add coverage bands
    ax1.axhspan(0.88, 0.92, alpha=0.2, color='green', label='Target range')
    
    # Plot 2: MPIW vs Temperature
    ax2.plot(temperatures, mpws, 'r-', linewidth=2)
    ax2.axvline(x=optimal_temp, color='g', linestyle='--', label=f'Optimal T={optimal_temp:.3f}')
    ax2.scatter([optimal_temp], [optimal_mpiw], color='g', s=100, zorder=5)
    ax2.set_xlabel('Temperature', fontsize=12)
    ax2.set_ylabel('Mean Prediction Interval Width (MPIW)', fontsize=12)
    ax2.set_title('MPIW vs Temperature', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path("learnable_scoring_fn/experiment_tracking/temperature_coverage_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    
    # Additional analysis: Temperature sensitivity
    print("\nTemperature Sensitivity Analysis:")
    
    # Find temperature range for 88-92% coverage
    valid_mask = (coverage_array >= 0.88) & (coverage_array <= 0.92)
    valid_temps = temperatures[valid_mask]
    
    if len(valid_temps) > 0:
        print(f"  Temperature range for 88-92% coverage: [{valid_temps.min():.3f}, {valid_temps.max():.3f}]")
        print(f"  Range width: {valid_temps.max() - valid_temps.min():.3f}")
    else:
        print("  No temperatures found that achieve 88-92% coverage")
    
    # Compute gradient around optimal temperature
    if idx_closest > 0 and idx_closest < len(temperatures) - 1:
        local_gradient = (coverages[idx_closest + 1] - coverages[idx_closest - 1]) / (temperatures[idx_closest + 1] - temperatures[idx_closest - 1])
        print(f"  Local gradient at optimal T: {local_gradient:.3f} coverage/temperature")
        print(f"  Interpretation: 1% change in T changes coverage by ~{abs(local_gradient * 0.01):.1%}")
    
    # Save detailed results
    results = {
        'temperatures': temperatures.tolist(),
        'coverages': [float(c) for c in coverages],
        'mpws': [float(m) for m in mpws],
        'optimal': {
            'temperature': float(optimal_temp),
            'coverage': float(optimal_coverage),
            'mpiw': float(optimal_mpiw)
        },
        'sensitivity': {
            'valid_temp_range': [float(valid_temps.min()), float(valid_temps.max())] if len(valid_temps) > 0 else None,
            'local_gradient': float(local_gradient) if 'local_gradient' in locals() else None
        }
    }
    
    results_path = Path("learnable_scoring_fn/experiment_tracking/temperature_analysis_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_path}")
    
    # Create secondary plot: Coverage error vs Temperature
    plt.figure(figsize=(10, 6))
    coverage_errors = np.abs(coverage_array - target_coverage)
    plt.plot(temperatures, coverage_errors, 'purple', linewidth=2)
    plt.axvline(x=optimal_temp, color='g', linestyle='--', label=f'Optimal T={optimal_temp:.3f}')
    plt.scatter([optimal_temp], [coverage_errors[idx_closest]], color='g', s=100, zorder=5)
    plt.xlabel('Temperature', fontsize=12)
    plt.ylabel('|Coverage - Target|', fontsize=12)
    plt.title('Absolute Coverage Error vs Temperature', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.yscale('log')
    
    error_plot_path = Path("learnable_scoring_fn/experiment_tracking/temperature_coverage_error.png")
    plt.savefig(error_plot_path, dpi=300, bbox_inches='tight')
    print(f"Error plot saved to: {error_plot_path}")
    
    plt.show()
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    analyze_temperature_coverage_relationship()