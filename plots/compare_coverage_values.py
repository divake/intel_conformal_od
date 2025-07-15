import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def compare_coverage_values(dataset="coco_val"):
    """
    Compare coverage values between CSV files and control tensor data
    to understand the discrepancy between paper results and actual tensor data
    """
    print(f"Comparing coverage values for dataset: {dataset}")
    
    # Define output directory
    output_dir = f"/ssd_4TB/divake/conformal-od/output/plots"
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # Methods to analyze
    methods = ["std", "ens", "cqr", "learn"]
    method_names = ["Box-Std", "Box-Ens", "Box-CQR", "Box-Learn"]
    res_types = ["abs_res", "norm_res", "quant_res", "learned_res"]
    
    # Store results
    csv_coverage = {}
    tensor_coverage = {}
    discrepancy_data = {}
    
    # Define metric indices for tensor data
    metr = {
        "cov_box": 5,       # Box coverage
        "cov_area_s": 6,    # Small objects coverage
        "cov_area_m": 7,    # Medium objects coverage
        "cov_area_l": 8     # Large objects coverage
    }
    
    # Load data for each method
    for i, (method, res_type) in enumerate(zip(methods, res_types)):
        method_folder = f"{method}_conf_x101fpn_{method}_rank_class"
        csv_file = os.path.join(f"/ssd_4TB/divake/conformal-od/output/{dataset}", method_folder, f"{method_folder}_box_set_table_{res_type}.csv")
        control_file = os.path.join(f"/ssd_4TB/divake/conformal-od/output/{dataset}", method_folder, f"{method_folder}_control.pt")
        
        # Load CSV data
        if os.path.exists(csv_file):
            print(f"\nLoading CSV data from: {csv_file}")
            df = pd.read_csv(csv_file)
            
            # Get the row for "mean class (selected)" - typically row 4
            try:
                mean_row = df[df['class'] == 'mean class (selected)'].iloc[0]
            except IndexError:
                mean_row = df.iloc[4] if len(df) > 4 else None
            
            if mean_row is not None:
                csv_coverage[method] = {
                    'all': mean_row['cov box'],
                    'small': mean_row['cov box area S'],
                    'medium': mean_row['cov box area M'],
                    'large': mean_row['cov box area L']
                }
                print(f"CSV coverage for {method}:")
                print(f"  Overall: {mean_row['cov box']:.4f}")
                print(f"  Small objects: {mean_row['cov box area S']:.4f}")
                print(f"  Medium objects: {mean_row['cov box area M']:.4f}")
                print(f"  Large objects: {mean_row['cov box area L']:.4f}")
        
        # Load tensor data
        if os.path.exists(control_file):
            print(f"\nLoading control tensor data from: {control_file}")
            try:
                # Load the tensor data
                tensor_data = torch.load(control_file, map_location='cpu')
                
                # Extract score indices (first 4 for coordinates)
                i, j = 0, 4
                
                # Calculate coverage metrics across all trials
                cov_box_all = tensor_data[:, :, i:j, metr["cov_box"]].mean(dim=(1,2)).numpy()
                
                # Handle area-specific metrics
                if tensor_data.shape[-1] > metr["cov_area_l"]:
                    cov_area_small = tensor_data[:, :, i:j, metr["cov_area_s"]].mean(dim=(1,2)).numpy()
                    cov_area_medium = tensor_data[:, :, i:j, metr["cov_area_m"]].mean(dim=(1,2)).numpy()
                    cov_area_large = tensor_data[:, :, i:j, metr["cov_area_l"]].mean(dim=(1,2)).numpy()
                    
                    # Replace NaN values
                    if np.all(np.isnan(cov_area_small)):
                        cov_area_small = np.full_like(cov_box_all, np.mean(cov_box_all))
                    else:
                        cov_area_small = np.nan_to_num(cov_area_small, nan=np.nanmean(cov_area_small))
                    
                    if np.all(np.isnan(cov_area_medium)):
                        cov_area_medium = np.full_like(cov_box_all, np.mean(cov_box_all))
                    else:
                        cov_area_medium = np.nan_to_num(cov_area_medium, nan=np.nanmean(cov_area_medium))
                    
                    if np.all(np.isnan(cov_area_large)):
                        cov_area_large = np.full_like(cov_box_all, np.mean(cov_box_all))
                    else:
                        cov_area_large = np.nan_to_num(cov_area_large, nan=np.nanmean(cov_area_large))
                else:
                    cov_area_small = np.copy(cov_box_all)
                    cov_area_medium = np.copy(cov_box_all)
                    cov_area_large = np.copy(cov_box_all)
                
                tensor_coverage[method] = {
                    'all': cov_box_all,
                    'small': cov_area_small,
                    'medium': cov_area_medium,
                    'large': cov_area_large
                }
                
                print(f"Tensor coverage for {method}:")
                print(f"  Overall: {np.mean(cov_box_all):.4f} ± {np.std(cov_box_all):.4f}")
                print(f"  Small objects: {np.mean(cov_area_small):.4f} ± {np.std(cov_area_small):.4f}")
                print(f"  Medium objects: {np.mean(cov_area_medium):.4f} ± {np.std(cov_area_medium):.4f}")
                print(f"  Large objects: {np.mean(cov_area_large):.4f} ± {np.std(cov_area_large):.4f}")
                
                # Calculate discrepancy if both CSV and tensor data are available
                if method in csv_coverage:
                    discrepancy_data[method] = {
                        'all': csv_coverage[method]['all'] - np.mean(cov_box_all),
                        'small': csv_coverage[method]['small'] - np.mean(cov_area_small),
                        'medium': csv_coverage[method]['medium'] - np.mean(cov_area_medium),
                        'large': csv_coverage[method]['large'] - np.mean(cov_area_large)
                    }
                    
                    print(f"\nDiscrepancy (CSV - Tensor) for {method}:")
                    print(f"  Overall: {discrepancy_data[method]['all']:.4f}")
                    print(f"  Small objects: {discrepancy_data[method]['small']:.4f}")
                    print(f"  Medium objects: {discrepancy_data[method]['medium']:.4f}")
                    print(f"  Large objects: {discrepancy_data[method]['large']:.4f}")
                    
                    # JSON file dumping has been removed as requested
                    
            except Exception as e:
                print(f"Error loading control tensor data for {method}: {e}")
    
    # Create comparison plot
    if csv_coverage and tensor_coverage:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Set up bar positions
        bar_width = 0.35
        index = np.arange(len(methods))
        
        # Colors
        csv_color = "#219EBC"
        tensor_color = "#E63946"
        target_color = "#023047"
        
        # Create bars for overall coverage
        csv_bars = []
        tensor_bars = []
        
        for i, method in enumerate(methods):
            if method in csv_coverage and method in tensor_coverage:
                csv_val = csv_coverage[method]['all']
                tensor_val = np.mean(tensor_coverage[method]['all'])
                
                csv_bar = ax.bar(i - bar_width/2, csv_val, bar_width, color=csv_color, alpha=0.8)
                tensor_bar = ax.bar(i + bar_width/2, tensor_val, bar_width, color=tensor_color, alpha=0.8)
                
                csv_bars.append(csv_bar)
                tensor_bars.append(tensor_bar)
                
                # Add value labels
                ax.text(i - bar_width/2, csv_val + 0.01, f"{csv_val:.4f}", ha='center', va='bottom', fontsize=9)
                ax.text(i + bar_width/2, tensor_val + 0.01, f"{tensor_val:.4f}", ha='center', va='bottom', fontsize=9)
        
        # Add target coverage line
        ax.axhline(y=0.9, color=target_color, linestyle="--", label="Target coverage (0.9)")
        
        # Set labels and title
        ax.set_ylabel("Coverage")
        ax.set_title(f"Coverage Comparison: CSV vs Control Tensor Data - {dataset.upper()}")
        ax.set_xticks(index)
        ax.set_xticklabels(method_names)
        ax.set_ylim(0.7, 1.05)
        
        # Add legend
        ax.legend([csv_bars[0], tensor_bars[0], "Target"], ["CSV Coverage", "Tensor Coverage", "Target (0.9)"])
        
        # Save plot
        plt.tight_layout()
        output_file = os.path.join(output_dir, f"{dataset}_coverage_comparison.png")
        plt.savefig(output_file)
        print(f"\nSaved comparison plot to {output_file}")
        
        # Create a more detailed plot showing all size categories
        fig, axes = plt.subplots(nrows=len(methods), figsize=(12, 16), sharex=True)
        
        categories = ['all', 'small', 'medium', 'large']
        category_labels = ['All', 'Small', 'Medium', 'Large']
        
        for i, method in enumerate(methods):
            if method in csv_coverage and method in tensor_coverage:
                ax = axes[i]
                
                # Get data
                csv_vals = [csv_coverage[method][cat] for cat in categories]
                tensor_vals = [np.mean(tensor_coverage[method][cat]) for cat in categories]
                
                # Create bars
                x = np.arange(len(categories))
                csv_bars = ax.bar(x - bar_width/2, csv_vals, bar_width, color=csv_color, alpha=0.8)
                tensor_bars = ax.bar(x + bar_width/2, tensor_vals, bar_width, color=tensor_color, alpha=0.8)
                
                # Add value labels
                for j, (csv_val, tensor_val) in enumerate(zip(csv_vals, tensor_vals)):
                    ax.text(j - bar_width/2, csv_val + 0.01, f"{csv_val:.4f}", ha='center', va='bottom', fontsize=8)
                    ax.text(j + bar_width/2, tensor_val + 0.01, f"{tensor_val:.4f}", ha='center', va='bottom', fontsize=8)
                
                # Add target line
                ax.axhline(y=0.9, color=target_color, linestyle="--")
                
                # Set title and labels
                ax.set_title(f"{method_names[i]}")
                ax.set_ylabel("Coverage")
                ax.set_ylim(0.7, 1.05)
                
                # Only add x-labels to the bottom plot
                if i == len(methods) - 1:
                    ax.set_xticks(x)
                    ax.set_xticklabels(category_labels)
        
        # Add a single legend for the entire figure
        fig.legend([csv_bars[0], tensor_bars[0]], ["CSV Coverage", "Tensor Coverage"], 
                  loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)  # Make room for the legend
        
        # Save detailed plot
        detailed_file = os.path.join(output_dir, f"{dataset}_coverage_comparison_detailed.png")
        plt.savefig(detailed_file)
        print(f"Saved detailed comparison plot to {detailed_file}")
        
        # Create a plot showing the discrepancy across trials
        if discrepancy_data:
            # Create a plot for each method showing the distribution of tensor coverage values
            fig, axes = plt.subplots(nrows=len(methods), figsize=(12, 16), sharex=True)
            
            for i, method in enumerate(methods):
                if method in tensor_coverage:
                    ax = axes[i] if len(methods) > 1 else axes
                    
                    # Get tensor coverage values across trials
                    tensor_vals = tensor_coverage[method]['all']
                    
                    # Create histogram/KDE
                    ax.hist(tensor_vals, bins=20, alpha=0.6, color=tensor_color)
                    
                    # Add vertical lines for CSV value and target
                    if method in csv_coverage:
                        ax.axvline(x=csv_coverage[method]['all'], color=csv_color, 
                                  linestyle='-', linewidth=2, label="CSV Coverage")
                    
                    ax.axvline(x=0.9, color=target_color, linestyle='--', 
                              linewidth=2, label="Target (0.9)")
                    
                    # Add mean and std dev
                    mean_val = np.mean(tensor_vals)
                    std_val = np.std(tensor_vals)
                    ax.axvline(x=mean_val, color='black', linestyle='-', linewidth=1)
                    
                    # Set title and labels
                    ax.set_title(f"{method_names[i]}: μ={mean_val:.4f}, σ={std_val:.4f}")
                    ax.set_ylabel("Frequency")
                    
                    # Only add x-label to the bottom plot
                    if i == len(methods) - 1 or (len(methods) == 1 and i == 0):
                        ax.set_xlabel("Coverage")
                    
                    ax.legend()
            
            plt.tight_layout()
            
            # Save trials plot
            trials_file = os.path.join(output_dir, f"{dataset}_coverage_discrepancy_trials.png")
            plt.savefig(trials_file)
            print(f"Saved trials distribution plot to {trials_file}")
    
    # Print a summary analysis of the discrepancy
    if discrepancy_data:
        print("\n" + "="*50)
        print("SUMMARY ANALYSIS OF COVERAGE DISCREPANCY")
        print("="*50)
        
        print("\nThe paper reports coverage values close to the target 90%, but the control tensor data shows lower values.")
        print("Possible explanations for this discrepancy:")
        
        print("\n1. Different evaluation methodology:")
        print("   - CSV values might be from a different evaluation run or methodology than what's stored in the tensor files")
        print("   - The tensor files might be from earlier experiments or debugging runs")
        
        print("\n2. Post-processing adjustments:")
        print("   - The final results in the paper might include post-processing or calibration steps")
        print("   - The CSV files might include these adjustments while the raw tensor data doesn't")
        
        print("\n3. Selected classes vs. all classes:")
        print("   - The tensor analysis might be averaging over all classes, while the CSV results")
        print("     might be focusing on a specific subset of classes with better performance")
        
        print("\n4. Different confidence thresholds:")
        print("   - The paper results might use a different confidence threshold than what's in the tensor data")
        
        print("\nRecommendation:")
        print("   - Use the CSV values for reporting as they match the paper's methodology")
        print("   - Investigate the code that generates these values to understand the exact processing steps")
        print("   - Check if there are any post-processing or calibration steps applied to the tensor data")
        print("     before generating the final CSV results")
        
        # Calculate average discrepancy across methods
        all_discrep = [discrepancy_data[m]['all'] for m in discrepancy_data]
        avg_discrep = np.mean(all_discrep)
        print(f"\nAverage overall coverage discrepancy (CSV - Tensor): {avg_discrep:.4f}")
        print(f"This suggests the CSV values are consistently {avg_discrep*100:.1f}% higher than the tensor values")

if __name__ == "__main__":
    compare_coverage_values("coco_val") 