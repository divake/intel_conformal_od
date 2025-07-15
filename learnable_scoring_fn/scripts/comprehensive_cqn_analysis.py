#!/usr/bin/env python
"""
Comprehensive analysis of CQN + Temperature Scaling results.
Produces detailed CSV files and publication-quality plots.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add paths
import sys
sys.path.append(str(Path(__file__).parent))

from learnable_scoring_fn.core_symmetric.models.cqn_model import ConditionalQuantileNetwork
from learnable_scoring_fn.calibrate_cqn import CalibratedCQN
from learnable_scoring_fn.core_symmetric.symmetric_adaptive import load_cached_data, prepare_splits

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12


def compute_iou(pred_boxes, gt_boxes):
    """
    Compute IoU between prediction and ground truth boxes.
    
    Args:
        pred_boxes: Predicted boxes, shape [N, 4] or [4] in [x1, y1, x2, y2] format
        gt_boxes: Ground truth boxes, shape [N, 4] or [4] in [x1, y1, x2, y2] format
        
    Returns:
        iou: IoU values, shape [N] or scalar
    """
    # Handle single box case
    if pred_boxes.dim() == 1:
        pred_boxes = pred_boxes.unsqueeze(0)
    if gt_boxes.dim() == 1:
        gt_boxes = gt_boxes.unsqueeze(0)
    
    # Compute intersection coordinates
    x1 = torch.max(pred_boxes[:, 0], gt_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], gt_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], gt_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], gt_boxes[:, 3])
    
    # Compute intersection area
    inter_width = torch.clamp(x2 - x1, min=0)
    inter_height = torch.clamp(y2 - y1, min=0)
    inter_area = inter_width * inter_height
    
    # Compute box areas
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    
    # Compute union area
    union_area = pred_area + gt_area - inter_area
    
    # Compute IoU
    iou = inter_area / (union_area + 1e-6)
    
    return iou.squeeze() if iou.shape[0] == 1 else iou


def match_predictions_to_gt(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Match predictions to ground truth boxes using Hungarian matching.
    Returns:
        matched_pred_idx: indices of matched predictions
        matched_gt_idx: indices of matched ground truths
        unmatched_pred_idx: indices of unmatched predictions (false positives)
        unmatched_gt_idx: indices of unmatched ground truths (false negatives)
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return [], [], list(range(len(pred_boxes))), list(range(len(gt_boxes)))
    
    # Compute IoU matrix
    iou_matrix = torch.zeros(len(pred_boxes), len(gt_boxes))
    for i, pred in enumerate(pred_boxes):
        for j, gt in enumerate(gt_boxes):
            iou_matrix[i, j] = compute_iou(pred, gt)
    
    # Simple greedy matching
    matched_pred_idx = []
    matched_gt_idx = []
    
    while iou_matrix.numel() > 0 and iou_matrix.max() > iou_threshold:
        max_iou = iou_matrix.max()
        max_idx = iou_matrix.argmax()
        pred_idx = max_idx // iou_matrix.shape[1]
        gt_idx = max_idx % iou_matrix.shape[1]
        
        matched_pred_idx.append(pred_idx.item())
        matched_gt_idx.append(gt_idx.item())
        
        # Remove matched boxes
        iou_matrix[pred_idx, :] = -1
        iou_matrix[:, gt_idx] = -1
    
    # Find unmatched
    all_pred_idx = set(range(len(pred_boxes)))
    all_gt_idx = set(range(len(gt_boxes)))
    unmatched_pred_idx = list(all_pred_idx - set(matched_pred_idx))
    unmatched_gt_idx = list(all_gt_idx - set(matched_gt_idx))
    
    return matched_pred_idx, matched_gt_idx, unmatched_pred_idx, unmatched_gt_idx


def categorize_by_size(areas):
    """Categorize objects by size."""
    small_mask = areas < 32**2
    medium_mask = (areas >= 32**2) & (areas < 96**2)
    large_mask = areas >= 96**2
    return small_mask, medium_mask, large_mask


def analyze_cqn_performance():
    """Perform comprehensive analysis of CQN + temperature scaling."""
    
    print("="*80)
    print("COMPREHENSIVE CQN + TEMPERATURE SCALING ANALYSIS")
    print("="*80)
    
    # Load model
    print("\nLoading CQN model and data...")
    checkpoint_path = Path("learnable_scoring_fn/experiment_tracking/checkpoints/cqn_calibrated_success.pt")
    if not checkpoint_path.exists():
        checkpoint_path = Path("learnable_scoring_fn/experiment_tracking/checkpoints/cqn_best.pt")
        temperature = 1.68  # Default from our analysis
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        temperature = checkpoint.get('temperature', 1.68)
    
    # Initialize model
    model = ConditionalQuantileNetwork(
        input_dim=17,
        hidden_dims=[256, 128, 64],
        dropout_rate=0.1,
        base_quantile=0.9
    )
    
    if checkpoint_path.name == 'cqn_best.pt':
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    # Create calibrated model
    calibrated_model = CalibratedCQN(model, target_coverage=0.9)
    calibrated_model.temperature = temperature
    
    # Load data
    cache_dir = "/ssd_4TB/divake/conformal-od/learnable_scoring_fn/cache_base_model"
    raw_data = load_cached_data(cache_dir)
    
    val_data_dict = {
        'features': raw_data['val_features'],
        'pred_coords': raw_data['val_data']['pred_coords'],
        'gt_coords': raw_data['val_data']['gt_coords'],
        'confidence': raw_data['val_data']['confidence']
    }
    
    # Use test split for analysis
    _, test_data = prepare_splits(val_data_dict, calib_fraction=0.5, seed=42)
    
    # Prepare for batch analysis
    test_dataset = TensorDataset(
        test_data['features'],
        test_data['pred_coords'],
        test_data['gt_coords'],
        test_data['confidence']
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # Batch size 1 for detailed analysis
    
    # Storage for results
    results = {
        'gt_count': 0,
        'pred_count': 0,
        'covered': {'small': [], 'medium': [], 'large': [], 'all': []},
        'uncovered': {'small': [], 'medium': [], 'large': [], 'all': []},
        'confidence_covered': [],
        'confidence_uncovered': [],
        'iou_covered': [],
        'iou_uncovered': [],
        'spatial_coverage': {'x': [], 'y': [], 'covered': []},
        'size_vs_mpiw': {'size': [], 'mpiw': [], 'covered': []},
        'mpiw_by_iou': {'low': [], 'medium': [], 'high': []},  # IoU < 0.7, 0.7-0.85, > 0.85
        'coverage_by_confidence': {}  # Will store coverage by confidence bins
    }
    
    print("\nAnalyzing predictions...")
    
    # Since CQN data has pred_coords and gt_coords already matched, we need different logic
    all_pred_boxes = test_data['pred_coords']
    all_gt_boxes = test_data['gt_coords'] 
    all_features = test_data['features']
    all_confidence = test_data['confidence']
    
    # Process in batches for efficiency
    batch_size = 256
    num_samples = len(all_pred_boxes)
    
    with torch.no_grad():
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            
            if start_idx % 5000 == 0:
                print(f"  Processing samples {start_idx}/{num_samples}...")
            
            # Get batch
            features = all_features[start_idx:end_idx].to(device)
            pred_boxes = all_pred_boxes[start_idx:end_idx].to(device)
            gt_boxes = all_gt_boxes[start_idx:end_idx].to(device)
            confidence = all_confidence[start_idx:end_idx].to(device)
            
            # Count boxes
            results['gt_count'] += len(gt_boxes)
            results['pred_count'] += len(pred_boxes)
            
            # Get predictions
            lower_bounds, upper_bounds = calibrated_model.predict_intervals(features, pred_boxes)
            
            # Compute MPIW for all predictions
            mpiw = (upper_bounds - lower_bounds).mean(dim=1)
            
            # Check coverage
            errors = torch.abs(gt_boxes - pred_boxes)
            widths = upper_bounds - pred_boxes
            covered = (errors <= widths).all(dim=1)
            
            # Compute IoU for all pairs
            ious = torch.zeros(len(pred_boxes), device=device)
            for i in range(len(pred_boxes)):
                ious[i] = compute_iou(pred_boxes[i:i+1], gt_boxes[i:i+1])
            
            # Compute areas
            gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
            
            # Process each sample
            for i in range(len(pred_boxes)):
                gt_area = gt_areas[i].item()
                box_mpiw = mpiw[i].item()
                box_conf = confidence[i].item()
                iou = ious[i].item()
                is_covered = covered[i].item()
                
                # Categorize by size
                size_category = 'small' if gt_area < 32**2 else \
                               'medium' if gt_area < 96**2 else 'large'
                
                # Store results
                if is_covered:
                    results['covered'][size_category].append(box_mpiw)
                    results['covered']['all'].append(box_mpiw)
                    results['confidence_covered'].append(box_conf)
                    results['iou_covered'].append(iou)
                else:
                    results['uncovered'][size_category].append(box_mpiw)
                    results['uncovered']['all'].append(box_mpiw)
                    results['confidence_uncovered'].append(box_conf)
                    results['iou_uncovered'].append(iou)
                
                # Spatial coverage
                center_x = (gt_boxes[i, 0] + gt_boxes[i, 2]) / 2
                center_y = (gt_boxes[i, 1] + gt_boxes[i, 3]) / 2
                results['spatial_coverage']['x'].append(center_x.item())
                results['spatial_coverage']['y'].append(center_y.item())
                results['spatial_coverage']['covered'].append(int(is_covered))
                
                # Size vs MPIW
                results['size_vs_mpiw']['size'].append(gt_area)
                results['size_vs_mpiw']['mpiw'].append(box_mpiw)
                results['size_vs_mpiw']['covered'].append(int(is_covered))
                
                # MPIW by IoU category
                if iou < 0.7:
                    results['mpiw_by_iou']['low'].append(box_mpiw)
                elif iou < 0.85:
                    results['mpiw_by_iou']['medium'].append(box_mpiw)
                else:
                    results['mpiw_by_iou']['high'].append(box_mpiw)
    
    # Generate comprehensive report
    print("\nGenerating analysis report...")
    
    # Create results directory
    results_dir = Path("learnable_scoring_fn/experiment_tracking/cqn_detailed_analysis")
    results_dir.mkdir(exist_ok=True)
    
    # 1. Summary statistics CSV
    summary_data = []
    
    # Overall statistics
    summary_data.append({
        'Category': 'Overall',
        'Metric': 'GT Count',
        'Value': results['gt_count']
    })
    summary_data.append({
        'Category': 'Overall',
        'Metric': 'Prediction Count',
        'Value': results['pred_count']
    })
    summary_data.append({
        'Category': 'Overall',
        'Metric': 'Pred/GT Ratio',
        'Value': f"{results['pred_count']/results['gt_count']:.2f}"
    })
    
    # Coverage and MPIW by category
    for category in ['small', 'medium', 'large', 'all']:
        covered = results['covered'][category]
        uncovered = results['uncovered'][category]
        
        total_matched = len(covered) + len(uncovered)
        coverage = len(covered) / total_matched * 100 if total_matched > 0 else 0
        
        summary_data.extend([
            {
                'Category': category.capitalize(),
                'Metric': 'Coverage (%)',
                'Value': f"{coverage:.1f}"
            },
            {
                'Category': category.capitalize(),
                'Metric': 'MPIW (Covered)',
                'Value': f"{np.mean(covered):.1f}" if covered else "N/A"
            },
            {
                'Category': category.capitalize(),
                'Metric': 'MPIW (Not Covered)',
                'Value': f"{np.mean(uncovered):.1f}" if uncovered else "N/A"
            },
            {
                'Category': category.capitalize(),
                'Metric': 'Count (Covered)',
                'Value': len(covered)
            },
            {
                'Category': category.capitalize(),
                'Metric': 'Count (Not Covered)',
                'Value': len(uncovered)
            },
            {
                'Category': category.capitalize(),
                'Metric': 'Total Count',
                'Value': total_matched
            }
        ])
    
    # Save summary CSV
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(results_dir / 'summary_statistics.csv', index=False)
    print(f"Saved: {results_dir / 'summary_statistics.csv'}")
    
    # 2. Detailed results CSV
    detailed_data = []
    
    # Covered predictions
    for size in ['small', 'medium', 'large']:
        for i, mpiw in enumerate(results['covered'][size]):
            detailed_data.append({
                'Type': 'Covered',
                'Size Category': size,
                'MPIW': mpiw,
                'Coverage': 1
            })
    
    # Uncovered predictions
    for size in ['small', 'medium', 'large']:
        for i, mpiw in enumerate(results['uncovered'][size]):
            detailed_data.append({
                'Type': 'Uncovered',
                'Size Category': size,
                'MPIW': mpiw,
                'Coverage': 0
            })
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_df.to_csv(results_dir / 'detailed_results.csv', index=False)
    print(f"Saved: {results_dir / 'detailed_results.csv'}")
    
    # 3. Create visualizations
    create_comprehensive_plots(results, results_dir, summary_df, temperature)
    
    # Print summary to console
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    print(f"\nGround Truth Objects: {results['gt_count']:,}")
    print(f"Predicted Objects: {results['pred_count']:,}")
    print(f"Prediction/GT Ratio: {results['pred_count']/results['gt_count']:.2f}")
    
    print("\nCoverage by Size Category:")
    for category in ['small', 'medium', 'large', 'all']:
        covered = results['covered'][category]
        uncovered = results['uncovered'][category]
        total = len(covered) + len(uncovered)
        if total > 0:
            coverage = len(covered) / total * 100
            print(f"  {category.capitalize()}: {coverage:.1f}% ({len(covered)}/{total})")
    
    print("\nMean Prediction Interval Width (MPIW):")
    print("  Category    | Covered | Not Covered | Difference")
    print("  " + "-"*50)
    for category in ['small', 'medium', 'large', 'all']:
        covered = results['covered'][category]
        uncovered = results['uncovered'][category]
        
        covered_mpiw = np.mean(covered) if covered else 0
        uncovered_mpiw = np.mean(uncovered) if uncovered else 0
        diff = uncovered_mpiw - covered_mpiw
        
        print(f"  {category.capitalize():10} | {covered_mpiw:7.1f} | {uncovered_mpiw:11.1f} | {diff:+10.1f}")
    
    # Add IoU-based analysis
    print("\nMPIW by IoU Category:")
    print("  IoU Range   | MPIW    | Count")
    print("  " + "-"*35)
    iou_categories = [('Low (<0.7)', results['mpiw_by_iou']['low']),
                      ('Med (0.7-0.85)', results['mpiw_by_iou']['medium']),
                      ('High (>0.85)', results['mpiw_by_iou']['high'])]
    for name, values in iou_categories:
        if values:
            print(f"  {name:12} | {np.mean(values):7.1f} | {len(values):6}")
    
    print("\n" + "="*80)
    print(f"All results saved to: {results_dir}")
    print("="*80)


def create_comprehensive_plots(results, results_dir, summary_df, temperature=1.68):
    """Create publication-quality plots."""
    
    print("\nCreating visualizations...")
    
    # Color palette
    colors = {
        'covered': '#2ecc71',
        'uncovered': '#e74c3c',
        'false_positive': '#f39c12',
        'small': '#3498db',
        'medium': '#9b59b6',
        'large': '#e67e22'
    }
    
    # 1. Coverage and MPIW comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Coverage by size
    categories = ['Small', 'Medium', 'Large', 'All']
    coverages = []
    for cat in categories:
        coverage_val = summary_df[(summary_df['Category'] == cat) & 
                                 (summary_df['Metric'] == 'Coverage (%)')]['Value'].values
        if len(coverage_val) > 0:
            coverages.append(float(coverage_val[0]))
        else:
            coverages.append(0)
    
    bars1 = ax1.bar(categories, coverages, color=[colors['small'], colors['medium'], 
                                                   colors['large'], '#34495e'])
    ax1.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='Target (90%)')
    ax1.set_ylabel('Coverage (%)')
    ax1.set_title('Coverage by Object Size')
    ax1.set_ylim(80, 100)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars1, coverages):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom')
    
    # MPIW comparison
    x = np.arange(len(categories))
    width = 0.35
    
    mpiw_covered = []
    mpiw_uncovered = []
    
    for cat in categories:
        # Get MPIW values
        covered_val = summary_df[(summary_df['Category'] == cat) & 
                                (summary_df['Metric'] == 'MPIW (Covered)')]['Value'].values
        uncovered_val = summary_df[(summary_df['Category'] == cat) & 
                                  (summary_df['Metric'] == 'MPIW (Not Covered)')]['Value'].values
        
        mpiw_covered.append(float(covered_val[0]) if covered_val[0] != 'N/A' else 0)
        mpiw_uncovered.append(float(uncovered_val[0]) if uncovered_val[0] != 'N/A' else 0)
    
    bars2 = ax2.bar(x - width/2, mpiw_covered, width, label='Covered', color=colors['covered'])
    bars3 = ax2.bar(x + width/2, mpiw_uncovered, width, label='Not Covered', color=colors['uncovered'])
    
    ax2.set_xlabel('Object Size Category')
    ax2.set_ylabel('Mean Prediction Interval Width (pixels)')
    ax2.set_title('MPIW by Coverage Status and Size')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'coverage_mpiw_comparison.png', bbox_inches='tight')
    plt.close()
    
    # 2. Distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # MPIW distributions
    ax = axes[0, 0]
    all_covered = results['covered']['all']
    all_uncovered = results['uncovered']['all']
    
    if all_covered:
        ax.hist(all_covered, bins=30, alpha=0.6, label=f'Covered (n={len(all_covered)})', 
                color=colors['covered'], density=True)
    if all_uncovered:
        ax.hist(all_uncovered, bins=30, alpha=0.6, label=f'Not Covered (n={len(all_uncovered)})', 
                color=colors['uncovered'], density=True)
    
    ax.set_xlabel('MPIW (pixels)')
    ax.set_ylabel('Density')
    ax.set_title('MPIW Distribution by Coverage Status')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Confidence score analysis
    ax = axes[0, 1]
    if results['confidence_covered']:
        ax.hist(results['confidence_covered'], bins=20, alpha=0.6, 
                label=f'Covered (n={len(results["confidence_covered"])})', 
                color=colors['covered'], density=True)
    if results['confidence_uncovered']:
        ax.hist(results['confidence_uncovered'], bins=20, alpha=0.6, 
                label=f'Not Covered (n={len(results["confidence_uncovered"])})', 
                color=colors['uncovered'], density=True)
    
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Density')
    ax.set_title('Confidence Score Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # IoU distribution for matched predictions
    ax = axes[1, 0]
    if results['iou_covered']:
        ax.hist(results['iou_covered'], bins=20, alpha=0.6, 
                label=f'Covered (n={len(results["iou_covered"])})', 
                color=colors['covered'], density=True)
    if results['iou_uncovered']:
        ax.hist(results['iou_uncovered'], bins=20, alpha=0.6, 
                label=f'Not Covered (n={len(results["iou_uncovered"])})', 
                color=colors['uncovered'], density=True)
    
    ax.set_xlabel('IoU with Ground Truth')
    ax.set_ylabel('Density')
    ax.set_title('IoU Distribution for Matched Predictions')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0.5, 1.0)
    
    # Size vs MPIW scatter plot
    ax = axes[1, 1]
    sizes = np.array(results['size_vs_mpiw']['size'])
    mpws = np.array(results['size_vs_mpiw']['mpiw'])
    covered_mask = np.array(results['size_vs_mpiw']['covered'], dtype=bool)
    
    if len(sizes) > 0:
        # Plot covered and uncovered separately
        ax.scatter(np.sqrt(sizes[covered_mask]), mpws[covered_mask], 
                  alpha=0.5, c=colors['covered'], label='Covered', s=20)
        ax.scatter(np.sqrt(sizes[~covered_mask]), mpws[~covered_mask], 
                  alpha=0.5, c=colors['uncovered'], label='Not Covered', s=20)
        
        # Add trend line
        z = np.polyfit(np.sqrt(sizes), mpws, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(0, np.sqrt(sizes).max(), 100)
        ax.plot(x_trend, p(x_trend), "k--", alpha=0.8, label=f'Trend: y={z[0]:.2f}x+{z[1]:.1f}')
    
    ax.set_xlabel('Object Size (√pixels)')
    ax.set_ylabel('MPIW (pixels)')
    ax.set_title('Size-Adaptive Behavior')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Add size category boundaries
    ax.axvline(x=32, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=96, color='gray', linestyle=':', alpha=0.5)
    ax.text(16, ax.get_ylim()[1]*0.9, 'Small', ha='center')
    ax.text(64, ax.get_ylim()[1]*0.9, 'Medium', ha='center')
    ax.text(120, ax.get_ylim()[1]*0.9, 'Large', ha='center')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'distribution_analysis.png', bbox_inches='tight')
    plt.close()
    
    # 3. Spatial coverage heatmap
    if len(results['spatial_coverage']['x']) > 100:  # Only if enough data
        fig, ax = plt.subplots(figsize=(10, 8))
        
        x = np.array(results['spatial_coverage']['x'])
        y = np.array(results['spatial_coverage']['y'])
        covered = np.array(results['spatial_coverage']['covered'])
        
        # Create 2D histogram for coverage rate
        H_covered, xedges, yedges = np.histogram2d(
            x[covered == 1], y[covered == 1], bins=20
        )
        H_total, _, _ = np.histogram2d(x, y, bins=20)
        
        # Coverage rate per bin
        with np.errstate(divide='ignore', invalid='ignore'):
            coverage_rate = H_covered / H_total
            coverage_rate[np.isnan(coverage_rate)] = 0
        
        im = ax.imshow(coverage_rate.T, origin='lower', aspect='auto', 
                      cmap='RdYlGn', vmin=0.7, vmax=1.0,
                      extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        
        ax.set_xlabel('X Position (pixels)')
        ax.set_ylabel('Y Position (pixels)')
        ax.set_title('Spatial Coverage Heatmap')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Coverage Rate')
        
        plt.savefig(results_dir / 'spatial_coverage_heatmap.png', bbox_inches='tight')
        plt.close()
    
    # 4. Performance summary visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create a nice summary table visualization
    table_data = []
    for cat in ['Small', 'Medium', 'Large', 'All']:
        row_data = []
        # Get values from summary_df
        coverage = summary_df[(summary_df['Category'] == cat) & 
                             (summary_df['Metric'] == 'Coverage (%)')]['Value'].values[0]
        mpiw_cov = summary_df[(summary_df['Category'] == cat) & 
                             (summary_df['Metric'] == 'MPIW (Covered)')]['Value'].values[0]
        mpiw_uncov = summary_df[(summary_df['Category'] == cat) & 
                               (summary_df['Metric'] == 'MPIW (Not Covered)')]['Value'].values[0]
        count_cov = summary_df[(summary_df['Category'] == cat) & 
                              (summary_df['Metric'] == 'Count (Covered)')]['Value'].values[0]
        count_uncov = summary_df[(summary_df['Category'] == cat) & 
                                (summary_df['Metric'] == 'Count (Not Covered)')]['Value'].values[0]
        total_count = summary_df[(summary_df['Category'] == cat) & 
                                (summary_df['Metric'] == 'Total Count')]['Value'].values[0]
        
        mpiw_diff = float(mpiw_uncov) - float(mpiw_cov) if mpiw_cov != 'N/A' and mpiw_uncov != 'N/A' else 'N/A'
        
        table_data.append([
            cat,
            f"{coverage}%",
            f"{mpiw_cov}" if mpiw_cov != 'N/A' else 'N/A',
            f"{mpiw_uncov}" if mpiw_uncov != 'N/A' else 'N/A',
            f"{mpiw_diff:+.1f}" if isinstance(mpiw_diff, float) else 'N/A',
            f"{count_cov}/{total_count}"
        ])
    
    # Create table
    col_labels = ['Size Category', 'Coverage', 'MPIW\n(Covered)', 'MPIW\n(Not Covered)', 'MPIW\nDifference', 'Count\n(Covered/Total)']
    
    # Remove axes
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=col_labels, 
                    cellLoc='center', loc='center',
                    colWidths=[0.15, 0.12, 0.15, 0.15, 0.13, 0.2])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Color header
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows alternately
    for i in range(1, len(table_data) + 1):
        for j in range(len(col_labels)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
            else:
                table[(i, j)].set_facecolor('white')
    
    # Highlight 'All' row
    for j in range(len(col_labels)):
        table[(len(table_data), j)].set_facecolor('#d5dbdb')
        table[(len(table_data), j)].set_text_props(weight='bold')
    
    ax.set_title('CQN + Temperature Scaling Performance Summary', 
                fontsize=18, weight='bold', pad=20)
    
    # Add temperature info
    ax.text(0.5, -0.1, f'Temperature: {temperature:.3f}', 
            transform=ax.transAxes, ha='center', fontsize=14)
    
    plt.savefig(results_dir / 'performance_summary_table.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Created 4 visualization plots in {results_dir}")


if __name__ == "__main__":
    analyze_cqn_performance()