#!/usr/bin/env python3
"""Compare and visualize results from all trained models."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import numpy as np
from typing import Dict, List, Optional
import argparse

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_all_results(results_dir: Path) -> List[Dict]:
    """Load results from all model directories."""
    all_results = []
    
    for model_dir in results_dir.iterdir():
        if model_dir.is_dir() and (model_dir / "results.json").exists():
            with open(model_dir / "results.json", 'r') as f:
                results = json.load(f)
                
                # Load training history if available
                history_file = model_dir / "training_history.json"
                if history_file.exists():
                    with open(history_file, 'r') as fh:
                        results['history'] = json.load(fh)
                
                all_results.append(results)
    
    return all_results


def create_comparison_plots(all_results: List[Dict], output_dir: Path):
    """Create comprehensive comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data for plotting
    models = []
    coverages = []
    mpIWs = []
    params = []
    train_times = []
    
    for result in all_results:
        models.append(result['model_type'])
        coverages.append(result['final_metrics']['coverage'])
        mpIWs.append(result['final_metrics']['avg_width'])
        params.append(result['model_params'])
        train_times.append(result['training_time'] / 60)  # Convert to minutes
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Coverage vs MPIW scatter plot
    ax1 = plt.subplot(3, 3, 1)
    scatter = ax1.scatter(mpIWs, coverages, s=200, alpha=0.7, c=range(len(models)), cmap='viridis')
    
    # Add model labels
    for i, model in enumerate(models):
        ax1.annotate(model, (mpIWs[i], coverages[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # Add target coverage line
    ax1.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='Target Coverage (0.9)')
    ax1.set_xlabel('Mean Prediction Interval Width (MPIW)', fontsize=12)
    ax1.set_ylabel('Coverage', fontsize=12)
    ax1.set_title('Coverage vs Efficiency Trade-off', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Model comparison bar chart
    ax2 = plt.subplot(3, 3, 2)
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, coverages, width, label='Coverage', alpha=0.8)
    
    # Create second y-axis for MPIW
    ax2_twin = ax2.twinx()
    bars2 = ax2_twin.bar(x + width/2, mpIWs, width, label='MPIW', color='orange', alpha=0.8)
    
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('Coverage', fontsize=12)
    ax2_twin.set_ylabel('MPIW', fontsize=12)
    ax2.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, val in zip(bars1, coverages):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    for bar, val in zip(bars2, mpIWs):
        height = bar.get_height()
        ax2_twin.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Parameter efficiency
    ax3 = plt.subplot(3, 3, 3)
    ax3.scatter(np.array(params)/1000, mpIWs, s=200, alpha=0.7, c=range(len(models)), cmap='viridis')
    
    for i, model in enumerate(models):
        ax3.annotate(model, (params[i]/1000, mpIWs[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax3.set_xlabel('Number of Parameters (K)', fontsize=12)
    ax3.set_ylabel('MPIW', fontsize=12)
    ax3.set_title('Parameter Efficiency', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Training time comparison
    ax4 = plt.subplot(3, 3, 4)
    bars = ax4.bar(models, train_times, alpha=0.8, color='skyblue')
    ax4.set_xlabel('Model', fontsize=12)
    ax4.set_ylabel('Training Time (minutes)', fontsize=12)
    ax4.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax4.set_xticklabels(models, rotation=45, ha='right')
    
    # Add value labels
    for bar, val in zip(bars, train_times):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax4.grid(True, alpha=0.3)
    
    # 5. Efficiency vs Coverage heatmap
    ax5 = plt.subplot(3, 3, 5)
    efficiency_matrix = np.array([[cov, mpiw] for cov, mpiw in zip(coverages, mpIWs)])
    
    im = ax5.imshow(efficiency_matrix.T, aspect='auto', cmap='RdYlGn_r')
    ax5.set_xticks(range(len(models)))
    ax5.set_xticklabels(models, rotation=45, ha='right')
    ax5.set_yticks([0, 1])
    ax5.set_yticklabels(['Coverage', 'MPIW'])
    ax5.set_title('Performance Heatmap', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(models)):
        ax5.text(i, 0, f'{coverages[i]:.3f}', ha='center', va='center', fontsize=9)
        ax5.text(i, 1, f'{mpIWs[i]:.1f}', ha='center', va='center', fontsize=9)
    
    plt.colorbar(im, ax=ax5)
    
    # 6. Ranking table
    ax6 = plt.subplot(3, 3, 6)
    ax6.axis('tight')
    ax6.axis('off')
    
    # Create ranking data
    ranking_data = []
    for i, (model, cov, mpiw, p, t) in enumerate(zip(models, coverages, mpIWs, params, train_times)):
        ranking_data.append([
            i+1,  # Rank by MPIW
            model,
            f'{cov:.3f}',
            f'{mpiw:.1f}',
            f'{p:,}',
            f'{t:.1f}'
        ])
    
    # Sort by MPIW
    ranking_data.sort(key=lambda x: float(x[3]))
    for i, row in enumerate(ranking_data):
        row[0] = i + 1  # Update rank
    
    table = ax6.table(cellText=ranking_data,
                     colLabels=['Rank', 'Model', 'Coverage', 'MPIW', 'Params', 'Time (min)'],
                     cellLoc='center', loc='center',
                     colWidths=[0.08, 0.25, 0.15, 0.15, 0.2, 0.17])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax6.set_title('Model Rankings (by MPIW)', fontsize=14, fontweight='bold', pad=20)
    
    # 7. Coverage deviation from target
    ax7 = plt.subplot(3, 3, 7)
    deviations = [abs(cov - 0.9) * 100 for cov in coverages]  # Percentage points
    bars = ax7.bar(models, deviations, alpha=0.8, color='coral')
    ax7.set_xlabel('Model', fontsize=12)
    ax7.set_ylabel('Deviation from Target (%)', fontsize=12)
    ax7.set_title('Coverage Deviation from Target (0.9)', fontsize=14, fontweight='bold')
    ax7.set_xticklabels(models, rotation=45, ha='right')
    
    # Add value labels
    for bar, val in zip(bars, deviations):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax7.grid(True, alpha=0.3)
    
    # 8. Training curves (if available)
    ax8 = plt.subplot(3, 3, 8)
    has_history = False
    
    for result in all_results:
        if 'history' in result and 'val_coverage' in result['history']:
            has_history = True
            epochs = range(1, len(result['history']['val_coverage']) + 1)
            ax8.plot(epochs, result['history']['val_coverage'], 
                    label=f"{result['model_type']}", linewidth=2, alpha=0.8)
    
    if has_history:
        ax8.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='Target')
        ax8.set_xlabel('Epoch', fontsize=12)
        ax8.set_ylabel('Validation Coverage', fontsize=12)
        ax8.set_title('Training Progress - Coverage', fontsize=14, fontweight='bold')
        ax8.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax8.grid(True, alpha=0.3)
    else:
        ax8.text(0.5, 0.5, 'Training history not available', 
                ha='center', va='center', transform=ax8.transAxes, fontsize=12)
        ax8.set_title('Training Progress', fontsize=14, fontweight='bold')
    
    # 9. MPIW training curves
    ax9 = plt.subplot(3, 3, 9)
    has_mpiw_history = False
    
    for result in all_results:
        if 'history' in result and 'val_avg_width' in result['history']:
            has_mpiw_history = True
            epochs = range(1, len(result['history']['val_avg_width']) + 1)
            ax9.plot(epochs, result['history']['val_avg_width'], 
                    label=f"{result['model_type']}", linewidth=2, alpha=0.8)
    
    if has_mpiw_history:
        ax9.set_xlabel('Epoch', fontsize=12)
        ax9.set_ylabel('Validation MPIW', fontsize=12)
        ax9.set_title('Training Progress - MPIW', fontsize=14, fontweight='bold')
        ax9.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax9.grid(True, alpha=0.3)
    else:
        ax9.text(0.5, 0.5, 'MPIW history not available', 
                ha='center', va='center', transform=ax9.transAxes, fontsize=12)
        ax9.set_title('Training Progress - MPIW', fontsize=14, fontweight='bold')
    
    # Overall title
    fig.suptitle('Comprehensive Model Comparison - Learnable Scoring Functions', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'model_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {output_dir}")


def create_summary_report(all_results: List[Dict], output_dir: Path):
    """Create a detailed summary report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'summary_report.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("LEARNABLE SCORING FUNCTION MODEL COMPARISON REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Sort results by MPIW
        sorted_results = sorted(all_results, key=lambda x: x['final_metrics']['avg_width'])
        
        # Best model summary
        best_model = sorted_results[0]
        f.write("BEST MODEL (by MPIW):\n")
        f.write("-"*40 + "\n")
        f.write(f"Model: {best_model['model_type']}\n")
        f.write(f"Coverage: {best_model['final_metrics']['coverage']:.3f}\n")
        f.write(f"MPIW: {best_model['final_metrics']['avg_width']:.2f}\n")
        f.write(f"Parameters: {best_model['model_params']:,}\n")
        f.write(f"Training Time: {best_model['training_time']/60:.1f} minutes\n")
        f.write(f"Configuration: {json.dumps(best_model['model_config'], indent=2)}\n")
        f.write("\n")
        
        # Detailed comparison
        f.write("DETAILED MODEL COMPARISON:\n")
        f.write("-"*40 + "\n\n")
        
        for i, result in enumerate(sorted_results):
            f.write(f"{i+1}. {result['model_type']}\n")
            f.write(f"   Coverage: {result['final_metrics']['coverage']:.3f} "
                   f"(deviation: {abs(result['final_metrics']['coverage'] - 0.9)*100:.1f}%)\n")
            f.write(f"   MPIW: {result['final_metrics']['avg_width']:.2f}\n")
            f.write(f"   Efficiency: {result['final_metrics'].get('efficiency', 'N/A')}\n")
            f.write(f"   Parameters: {result['model_params']:,}\n")
            f.write(f"   Training Time: {result['training_time']/60:.1f} minutes\n")
            f.write("\n")
        
        # Statistical summary
        f.write("\nSTATISTICAL SUMMARY:\n")
        f.write("-"*40 + "\n")
        
        coverages = [r['final_metrics']['coverage'] for r in all_results]
        mpIWs = [r['final_metrics']['avg_width'] for r in all_results]
        params_list = [r['model_params'] for r in all_results]
        times = [r['training_time']/60 for r in all_results]
        
        f.write(f"Coverage Range: [{min(coverages):.3f}, {max(coverages):.3f}]\n")
        f.write(f"Coverage Mean: {np.mean(coverages):.3f} ± {np.std(coverages):.3f}\n")
        f.write(f"MPIW Range: [{min(mpIWs):.1f}, {max(mpIWs):.1f}]\n")
        f.write(f"MPIW Mean: {np.mean(mpIWs):.1f} ± {np.std(mpIWs):.1f}\n")
        f.write(f"Parameter Range: [{min(params_list):,}, {max(params_list):,}]\n")
        f.write(f"Training Time Range: [{min(times):.1f}, {max(times):.1f}] minutes\n")
        
        # Recommendations
        f.write("\n\nRECOMMENDATIONS:\n")
        f.write("-"*40 + "\n")
        
        # Find models meeting coverage requirement
        valid_models = [r for r in sorted_results if r['final_metrics']['coverage'] >= 0.88]
        
        if valid_models:
            f.write(f"Models meeting coverage requirement (≥0.88): "
                   f"{', '.join([m['model_type'] for m in valid_models])}\n")
            f.write(f"Recommended model: {valid_models[0]['model_type']} "
                   f"(best MPIW among valid models)\n")
        else:
            f.write("WARNING: No models meet the coverage requirement of 0.88\n")
            f.write(f"Best coverage achieved: {max(coverages):.3f} by "
                   f"{[r['model_type'] for r in all_results if r['final_metrics']['coverage'] == max(coverages)][0]}\n")
    
    print(f"Summary report saved to {output_dir / 'summary_report.txt'}")


def main():
    parser = argparse.ArgumentParser(description="Compare model results")
    parser.add_argument('--results-dir', type=str, 
                       default='experiments/results',
                       help='Directory containing model results')
    parser.add_argument('--output-dir', type=str,
                       default='experiments/results/comparison',
                       help='Output directory for comparison results')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    if not results_dir.exists():
        print(f"Results directory {results_dir} does not exist!")
        return
    
    # Load all results
    all_results = load_all_results(results_dir)
    
    if not all_results:
        print("No results found to compare!")
        return
    
    print(f"Found {len(all_results)} model results")
    
    # Create comparison plots
    create_comparison_plots(all_results, output_dir)
    
    # Create summary report
    create_summary_report(all_results, output_dir)
    
    # Create CSV for easy analysis
    df_data = []
    for result in all_results:
        df_data.append({
            'Model': result['model_type'],
            'Coverage': result['final_metrics']['coverage'],
            'MPIW': result['final_metrics']['avg_width'],
            'Efficiency': result['final_metrics'].get('efficiency', 'N/A'),
            'Parameters': result['model_params'],
            'Training_Time_min': result['training_time'] / 60
        })
    
    df = pd.DataFrame(df_data)
    df = df.sort_values('MPIW')
    df.to_csv(output_dir / 'model_comparison.csv', index=False)
    
    print(f"\nComparison complete! Results saved to {output_dir}")
    print("\nTop 3 models by MPIW:")
    print(df[['Model', 'Coverage', 'MPIW']].head(3).to_string(index=False))


if __name__ == "__main__":
    main()