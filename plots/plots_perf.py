#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simplified Performance Plotting Script - Coverage and MPIW Only (3 Methods)

This script creates two specific plots:
1. Coverage violin plots comparing 3 methods
2. Coverage-Adjusted MPIW violin plots for fair comparison at 90% coverage

Dataset: COCO validation set
Methods: Standard, Ensemble, CQR

Features:
- Easy base model switching via configuration
- Coverage adjustment for fair MPIW comparison
- Clean 3-method comparison (learnable method removed)

Output Files:
- coco_val_coverage_violin.png
- coco_val_mpiw_adjusted_violin.png (MPIW adjusted to 90% coverage)
"""

# ============================================================================
# CONFIGURATION - EASY BASE MODEL SWITCHING
# ============================================================================

# Base model configuration - Change these to switch between models
BASE_MODEL_CONFIG = {
    # Current setup: R50-FPN for STD & ENS, X101-FPN for CQR
    'std': {
        'model_id': 'r50fpn',
        'model_name': 'ResNet-50-FPN',
        'model_file': 'faster_rcnn_R_50_FPN_3x.yaml'
    },
    'ens': {
        'model_id': 'r50fpn', 
        'model_name': 'ResNet-50-FPN',
        'model_file': 'faster_rcnn_R_50_FPN_3x.yaml'
    },
    'cqr': {
        'model_id': 'x101fpn',
        'model_name': 'ResNeXt-101-FPN', 
        'model_file': 'faster_rcnn_X_101_32x8d_FPN_3x.yaml'
    }
}

# Alternative configurations for easy switching:
# 
# ALL R50-FPN:
# BASE_MODEL_CONFIG = {
#     'std': {'model_id': 'r50fpn', 'model_name': 'ResNet-50-FPN', 'model_file': 'faster_rcnn_R_50_FPN_3x.yaml'},
#     'ens': {'model_id': 'r50fpn', 'model_name': 'ResNet-50-FPN', 'model_file': 'faster_rcnn_R_50_FPN_3x.yaml'},
#     'cqr': {'model_id': 'r50fpn', 'model_name': 'ResNet-50-FPN', 'model_file': 'faster_rcnn_R_50_FPN_3x.yaml'}
# }
#
# ALL X101-FPN:
# BASE_MODEL_CONFIG = {
#     'std': {'model_id': 'x101fpn', 'model_name': 'ResNeXt-101-FPN', 'model_file': 'faster_rcnn_X_101_32x8d_FPN_3x.yaml'},
#     'ens': {'model_id': 'x101fpn', 'model_name': 'ResNeXt-101-FPN', 'model_file': 'faster_rcnn_X_101_32x8d_FPN_3x.yaml'},
#     'cqr': {'model_id': 'x101fpn', 'model_name': 'ResNeXt-101-FPN', 'model_file': 'faster_rcnn_X_101_32x8d_FPN_3x.yaml'}
# }

# ============================================================================
# IMPORTS
# ============================================================================

import os
import sys
import torch
import numpy as np
from pathlib import Path
from scipy.stats import beta

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
matplotlib.rcParams.update({
    'text.usetex': False,
    'mathtext.default': 'regular',
    'font.family': ['DejaVu Sans', 'sans-serif'],
    'axes.unicode_minus': False,
    'figure.max_open_warning': 0
})

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_base_model_summary():
    """Get a summary of current base model configuration"""
    summary = []
    for method, config in BASE_MODEL_CONFIG.items():
        summary.append(f"{method.upper()}: {config['model_name']} ({config['model_id']})")
    return ", ".join(summary)

def get_unified_model_name():
    """Get unified model name if all methods use the same model, otherwise return mixed"""
    model_names = set(config['model_name'] for config in BASE_MODEL_CONFIG.values())
    if len(model_names) == 1:
        return list(model_names)[0]
    else:
        return "Mixed Models"

def configure_matplotlib_no_latex():
    """Ensure matplotlib is configured to not use LaTeX"""
    matplotlib.rcParams.update({
        'text.usetex': False,
        'mathtext.default': 'regular',
        'font.family': ['DejaVu Sans', 'sans-serif'],
        'axes.unicode_minus': False,
        'figure.max_open_warning': 0
    })
    plt.rcParams.update({
        'text.usetex': False,
        'mathtext.default': 'regular',
        'font.family': ['DejaVu Sans', 'sans-serif'],
        'axes.unicode_minus': False,
        'figure.max_open_warning': 0
    })

def save_fig(figname: str, **kwargs):
    """Save figure to file with given name"""
    plt.savefig(figname + ".png", format="png", dpi=300, bbox_inches='tight', **kwargs)
    plt.close()  # Close the figure to free memory

def adjust_mpiw_to_target_coverage(coverage_data, mpiw_data, target_coverage=0.9):
    """
    Adjust MPIW values to what they would be at target coverage level.
    
    This uses a linear scaling approach where:
    - Higher coverage typically means wider intervals (higher MPIW)
    - We scale MPIW proportionally to achieve target coverage
    
    Args:
        coverage_data: Array of coverage values for each trial
        mpiw_data: Array of MPIW values for each trial  
        target_coverage: Target coverage level (default 0.9)
        
    Returns:
        adjusted_mpiw: MPIW values adjusted to target coverage
    """
    # For each trial, calculate the scaling factor
    # If coverage > target: scale down MPIW (intervals were too wide)
    # If coverage < target: scale up MPIW (intervals were too narrow)
    
    # Simple linear scaling: 
    # If coverage is 95% and target is 90%, scale MPIW by 90/95 = 0.947
    # If coverage is 85% and target is 90%, scale MPIW by 90/85 = 1.059
    
    scaling_factors = target_coverage / coverage_data
    adjusted_mpiw = mpiw_data * scaling_factors
    
    return adjusted_mpiw

# ============================================================================
# DATA LOADING
# ============================================================================

def load_performance_data():
    """
    Load performance data from the 3 conformal prediction methods.
    
    Returns:
        dict: Dictionary containing coverage and MPIW data for each method
        
    Raises:
        FileNotFoundError: If any required result files are missing
        Exception: If any files cannot be loaded properly
    """
    print("Loading performance data from conformal prediction results...")
    print("Note: Learnable method removed - comparing only 3 methods")
    print(f"Base Model Configuration: {get_base_model_summary()}")
    
    # Generate dynamic base directories based on configuration
    base_dirs = {}
    data_files = {}
    
    for method in ['std', 'ens', 'cqr']:
        model_id = BASE_MODEL_CONFIG[method]['model_id']
        method_suffix = f"{method}_rank_class"
        
        base_dirs[method] = f'/ssd_4TB/divake/conformal-od/output/coco_val/{method}_conf_{model_id}_{method_suffix}'
        data_files[method] = f'{method}_conf_{model_id}_{method_suffix}_box_set.pt'
    
    # Method names and descriptions - Dynamic based on configuration
    method_info = {
        'std': {
            'name': 'Box-Std',
            'full_name': 'Standard Conformal (Absolute Residual)',
            'base_model': BASE_MODEL_CONFIG['std']['model_name'],
            'scoring': 'Absolute residual between predicted and ground truth boxes'
        },
        'ens': {
            'name': 'Box-Ens', 
            'full_name': 'Ensemble Conformal (Normalized Residual)',
            'base_model': f"{BASE_MODEL_CONFIG['ens']['model_name']} Ensemble (5 models)",
            'scoring': 'Normalized residual with ensemble uncertainty'
        },
        'cqr': {
            'name': 'Box-CQR',
            'full_name': 'Conformalized Quantile Regression', 
            'base_model': f"{BASE_MODEL_CONFIG['cqr']['model_name']} with Quantile Regression Head",
            'scoring': 'Quantile regression predictions (10th, 90th percentiles)'
        }
    }
    
    # Metric indices in the tensor data
    # Based on evaluation.results_table._idx_box_set_metrics
    metric_indices = {
        'cov_box': 5,    # Overall box coverage
        'mpiw': 2,       # Mean Prediction Interval Width
        'cov_area_s': 6, # Coverage for small objects
        'cov_area_m': 7, # Coverage for medium objects  
        'cov_area_l': 8  # Coverage for large objects
    }
    
    # First, check if all required files exist
    missing_files = []
    file_paths = {}
    
    for method in ['std', 'ens', 'cqr']:
        file_path = os.path.join(base_dirs[method], data_files[method])
        file_paths[method] = file_path
        
        if not os.path.exists(file_path):
            missing_files.append(f"  - {method.upper()}: {file_path}")
    
    # If any files are missing, show error and exit
    if missing_files:
        print("\n" + "="*80)
        print("❌ ERROR: REQUIRED RESULT FILES ARE MISSING!")
        print("="*80)
        print("The following result files are required but not found:")
        print()
        for missing in missing_files:
            print(missing)
        print()
        print("🔧 TO FIX THIS:")
        print("1. Run the conformal prediction experiments first using:")
        print("   /ssd_4TB/divake/conformal-od/run.sh")
        print("2. Or run the individual commands from:")
        print("   /ssd_4TB/divake/conformal-od/commands.txt")
        print()
        print("📋 COMMANDS TO GENERATE MISSING FILES:")
        
        cmd_templates = {
            'std': "/home/divake/miniconda3/envs/env_cu121/bin/python main.py --config_file=cfg_std_rank --config_path=config/coco_val --run_collect_pred --save_file_pred --risk_control=std_conf --alpha=0.1 --run_risk_control --save_file_control --run_eval --save_file_eval --device=cuda",
            'ens': "/home/divake/miniconda3/envs/env_cu121/bin/python main.py --config_file=cfg_ens_rank --config_path=config/coco_val --run_collect_pred --save_file_pred --risk_control=ens_conf --alpha=0.1 --run_risk_control --save_file_control --run_eval --save_file_eval --device=cuda",
            'cqr': "/home/divake/miniconda3/envs/env_cu121/bin/python main.py --config_file=cfg_cqr_rank --config_path=config/coco_val --run_collect_pred --save_file_pred --risk_control=cqr_conf --alpha=0.1 --run_risk_control --save_file_control --run_eval --save_file_eval --device=cuda"
        }
        
        for method in ['std', 'ens', 'cqr']:
            if any(method.upper() in missing for missing in missing_files):
                print(f"\n{method.upper()}:")
                print(f"  {cmd_templates[method]}")
        
        print("\n" + "="*80)
        raise FileNotFoundError(f"Missing {len(missing_files)} required result file(s). Please run the experiments first.")
    
    # All files exist, proceed with loading
    performance_data = {}
    
    for method in ['std', 'ens', 'cqr']:
        data_file_path = file_paths[method]
        
        print(f"\n=== {method_info[method]['name']} ({method_info[method]['full_name']}) ===")
        print(f"Base Model: {method_info[method]['base_model']}")
        print(f"Scoring Strategy: {method_info[method]['scoring']}")
        print(f"Loading from: {data_file_path}")
        
        try:
            # Load tensor data
            # Shape: [n_trials, n_classes, n_score_indices, n_metrics]
            control_data = torch.load(data_file_path, map_location='cpu', weights_only=False)
            
            print(f"Data shape: {control_data.shape}")
            print(f"  - {control_data.shape[0]} calibration trials")
            print(f"  - {control_data.shape[1]} object classes")
            print(f"  - {control_data.shape[2]} score indices")
            print(f"  - {control_data.shape[3]} metrics")
            
            # Extract metrics
            # Average over classes and score indices for each trial
            coverage_all = control_data[:, :, :, metric_indices['cov_box']].mean(dim=(1,2))
            mpiw_all = control_data[:, :, 0, metric_indices['mpiw']].mean(dim=1)  # Use first score index only
            
            # Convert to numpy
            coverage_data = coverage_all.cpu().numpy()
            mpiw_data = mpiw_all.cpu().numpy()
            
            # Calculate coverage-adjusted MPIW for fair comparison
            mpiw_adjusted = adjust_mpiw_to_target_coverage(coverage_data, mpiw_data, target_coverage=0.9)
            
            performance_data[method] = {
                'coverage': coverage_data,
                'mpiw': mpiw_data,
                'mpiw_adjusted': mpiw_adjusted,
                'info': method_info[method]
            }
            
            print(f"✓ Coverage: {coverage_data.mean():.3f} ± {coverage_data.std():.3f}")
            print(f"✓ MPIW (original): {mpiw_data.mean():.1f} ± {mpiw_data.std():.1f}")
            print(f"✓ MPIW (adjusted to 90%): {mpiw_adjusted.mean():.1f} ± {mpiw_adjusted.std():.1f}")
            
        except Exception as e:
            print(f"\n❌ ERROR: Failed to load {method.upper()} data from {data_file_path}")
            print(f"Error details: {e}")
            print(f"File exists: {os.path.exists(data_file_path)}")
            if os.path.exists(data_file_path):
                print(f"File size: {os.path.getsize(data_file_path)} bytes")
            print("\n🔧 TROUBLESHOOTING:")
            print("1. Check if the file was generated correctly")
            print("2. Verify the file is not corrupted")
            print("3. Re-run the conformal prediction experiment for this method")
            print("4. Check file permissions")
            raise Exception(f"Failed to load {method.upper()} result file: {e}")
    
    return performance_data

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_coverage_violin(performance_data, output_dir):
    """
    Create coverage violin plot comparing 3 methods.
    
    Args:
        performance_data (dict): Performance data from load_performance_data()
        output_dir (str): Directory to save the plot
    """
    print("\n" + "="*60)
    print("GENERATING COVERAGE VIOLIN PLOT (3 METHODS)")
    print("="*60)
    
    configure_matplotlib_no_latex()
    
    # Method order and colors
    methods = ['std', 'ens', 'cqr']
    colors = ["#E63946", "#219EBC", "#023047"]
    method_labels = [performance_data[m]['info']['name'] for m in methods]
    
    # Extract coverage data
    coverage_data = [performance_data[m]['coverage'] for m in methods]
    
    # Theoretical coverage distribution (for COCO validation set)
    n_calib = 930  # COCO validation set size (approximate)
    alpha = 0.1    # Miscoverage rate
    l = np.floor((n_calib + 1) * alpha)
    a_param = n_calib + 1 - l
    b_param = l
    rv = beta(a_param, b_param)
    liml, limh = rv.ppf(0.01), rv.ppf(0.99)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Add target coverage line and distribution
    ax.axhline(y=0.9, color="black", linestyle="--", linewidth=2, label='Target coverage (90%)')
    ax.axhspan(liml, limh, alpha=0.3, color="grey", label='Theoretical coverage distribution')
    
    # Create violin plot
    means = [d.mean() for d in coverage_data]
    violin = ax.violinplot(coverage_data, showextrema=False, widths=0.6, points=1000)
    
    # Style violins
    for i, body in enumerate(violin["bodies"]):
        body.set_facecolor(colors[i])
        body.set_edgecolor("black")
        body.set_alpha(0.8)
        body.set_linewidth(1)
        
        # Add horizontal mean lines
        try:
            path = body.get_paths()[0].to_polygons()[0]
            ax.plot([min(path[:,0])+0.01, max(path[:,0])-0.01], [means[i], means[i]], 
                   color="black", linestyle="-", linewidth=2)
        except (IndexError, ValueError):
            pass
    
    # Customize axes
    ax.set_ylabel("Coverage", fontsize=14)
    ax.set_ylim(0.84, 0.96)
    ax.set_yticks([0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96])
    
    # Set x-axis labels
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(method_labels, fontsize=12)
    ax.tick_params(axis="y", which="major", labelsize=11)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=11)
    
    # Add title with dynamic base model information
    unified_model = get_unified_model_name()
    ax.set_title(f"Coverage Comparison - COCO Validation Set\n"
                f"Base Model: {unified_model} (3 Methods)", fontsize=14, pad=20)
    
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(output_dir, "coco_val_coverage_violin")
    save_fig(output_file)
    print(f"✓ Coverage violin plot saved: {output_file}.png")
    
    # Print summary statistics
    print("\nCoverage Summary:")
    print("-" * 50)
    for i, method in enumerate(methods):
        data = coverage_data[i]
        print(f"{method_labels[i]:>10}: {data.mean():.3f} ± {data.std():.3f} "
              f"(min: {data.min():.3f}, max: {data.max():.3f})")

def plot_mpiw_violin(performance_data, output_dir):
    """
    Create MPIW violin plot comparing 3 methods with coverage adjustment.
    
    Args:
        performance_data (dict): Performance data from load_performance_data()
        output_dir (str): Directory to save the plot
    """
    print("\n" + "="*60)
    print("GENERATING COVERAGE-ADJUSTED MPIW VIOLIN PLOT (3 METHODS)")
    print("="*60)
    print("Note: MPIW values adjusted to 90% coverage for fair comparison")
    
    configure_matplotlib_no_latex()
    
    # Method order and colors
    methods = ['std', 'ens', 'cqr']
    colors = ["#E63946", "#219EBC", "#023047"]
    method_labels = [performance_data[m]['info']['name'] for m in methods]
    
    # Extract MPIW data (both original and adjusted)
    mpiw_original = [performance_data[m]['mpiw'] for m in methods]
    mpiw_adjusted = [performance_data[m]['mpiw_adjusted'] for m in methods]
    
    # Create plot with two subplots: original vs adjusted
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot 1: Original MPIW
    means_orig = [d.mean() for d in mpiw_original]
    violin1 = ax1.violinplot(mpiw_original, showextrema=False, widths=0.6, points=1000)
    
    for i, body in enumerate(violin1["bodies"]):
        body.set_facecolor(colors[i])
        body.set_edgecolor("black")
        body.set_alpha(0.8)
        body.set_linewidth(1)
        
        # Add horizontal mean lines
        try:
            path = body.get_paths()[0].to_polygons()[0]
            ax1.plot([min(path[:,0])+0.01, max(path[:,0])-0.01], [means_orig[i], means_orig[i]], 
                    color="black", linestyle="-", linewidth=2)
        except (IndexError, ValueError):
            pass
    
    ax1.set_ylabel("MPIW (Original)", fontsize=14)
    ax1.set_title("Original MPIW\n(Different Coverage Levels)", fontsize=12)
    ax1.set_xticks([1, 2, 3])
    ax1.set_xticklabels(method_labels, fontsize=10)
    ax1.tick_params(axis="y", which="major", labelsize=11)
    
    # Plot 2: Coverage-Adjusted MPIW
    means_adj = [d.mean() for d in mpiw_adjusted]
    violin2 = ax2.violinplot(mpiw_adjusted, showextrema=False, widths=0.6, points=1000)
    
    for i, body in enumerate(violin2["bodies"]):
        body.set_facecolor(colors[i])
        body.set_edgecolor("black")
        body.set_alpha(0.8)
        body.set_linewidth(1)
        
        # Add horizontal mean lines
        try:
            path = body.get_paths()[0].to_polygons()[0]
            ax2.plot([min(path[:,0])+0.01, max(path[:,0])-0.01], [means_adj[i], means_adj[i]], 
                    color="black", linestyle="-", linewidth=2)
        except (IndexError, ValueError):
            pass
    
    ax2.set_ylabel("MPIW (Adjusted to 90% Coverage)", fontsize=14)
    ax2.set_title("Coverage-Adjusted MPIW\n(Fair Comparison at 90%)", fontsize=12)
    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(method_labels, fontsize=10)
    ax2.tick_params(axis="y", which="major", labelsize=11)
    
    # Overall title with dynamic base model information
    unified_model = get_unified_model_name()
    fig.suptitle(f"MPIW Comparison - COCO Validation Set\n"
                f"Base Model: {unified_model} (Lower is Better)", fontsize=14)
    
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(output_dir, "coco_val_mpiw_adjusted_violin")
    save_fig(output_file)
    print(f"✓ Coverage-adjusted MPIW violin plot saved: {output_file}.png")
    
    # Print summary statistics
    print("\nMPIW Summary (Original vs Coverage-Adjusted):")
    print("-" * 70)
    print(f"{'Method':>10} {'Original MPIW':>15} {'Adjusted MPIW':>15} {'Coverage':>10}")
    print("-" * 70)
    for i, method in enumerate(methods):
        orig_data = mpiw_original[i]
        adj_data = mpiw_adjusted[i]
        cov_data = performance_data[method]['coverage']
        print(f"{method_labels[i]:>10}: {orig_data.mean():>7.1f} ± {orig_data.std():>4.1f} "
              f"→ {adj_data.mean():>7.1f} ± {adj_data.std():>4.1f} "
              f"(cov: {cov_data.mean():.3f})")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function to generate both performance plots"""
    print("="*80)
    print("CONFORMAL OBJECT DETECTION - PERFORMANCE PLOTTING (3 METHODS)")
    print("="*80)
    print("This script generates Coverage and Coverage-Adjusted MPIW violin plots")
    print("for 3 conformal prediction methods on COCO validation set.")
    print("(Learnable method removed for cleaner comparison)")
    print()
    print(f"Base Model Configuration: {get_base_model_summary()}")
    print("Dataset: COCO validation set")
    print("Methods: Standard, Ensemble, CQR")
    print("Coverage Adjustment: MPIW scaled to 90% coverage for fair comparison")
    print()
    print("💡 To switch base models, edit the BASE_MODEL_CONFIG at the top of this script")
    print("="*80)
    
    # Create output directory
    output_dir = "/ssd_4TB/divake/conformal-od/output/plots_perf"
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    print(f"Output directory: {output_dir}")
    
    # Load performance data
    performance_data = load_performance_data()
    
    # Generate plots
    plot_coverage_violin(performance_data, output_dir)
    plot_mpiw_violin(performance_data, output_dir)
    
    print("\n" + "="*80)
    print("✓ ALL PLOTS GENERATED SUCCESSFULLY!")
    print("="*80)
    print("Output files:")
    print(f"  - {output_dir}/coco_val_coverage_violin.png")
    print(f"  - {output_dir}/coco_val_mpiw_adjusted_violin.png")
    print()
    print("Key features:")
    print("  ✓ Easy base model switching via BASE_MODEL_CONFIG")
    print("  ✓ Dynamic file paths based on selected models")
    print("  ✓ Coverage adjustment for fair MPIW comparison")
    print("  ✓ Mixed model support (different models per method)")
    print("  ✓ All MPIW values normalized to 90% coverage level")
    print()
    print(f"Current configuration: {get_base_model_summary()}")
    print("="*80)

if __name__ == "__main__":
    main() 