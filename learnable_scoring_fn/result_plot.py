#!/usr/bin/env python
"""
Visualization script for Symmetric Size-Aware Adaptive Conformal Prediction

This script provides comprehensive visualization of:
1. Prediction boxes with symmetric intervals on actual images
2. Size-stratified performance metrics
3. Coverage-MPIW tradeoff analysis
4. Comparison of predicted intervals across object sizes
"""

import os
import sys
import json
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import seaborn as sns
from PIL import Image
import pickle

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import our modules
from learnable_scoring_fn.core_symmetric.models.symmetric_mlp import SymmetricAdaptiveMLP
from learnable_scoring_fn.core_symmetric.symmetric_adaptive import load_cached_data


class SymmetricAdaptiveVisualizer:
    """Visualizer for symmetric adaptive conformal prediction results."""
    
    def __init__(
        self,
        experiment_dir: str,
        cache_dir: str = "/ssd_4TB/divake/conformal-od/learnable_scoring_fn/cache_base_model",
        coco_images_dir: str = "/ssd_4TB/divake/conformal-od/data/coco/val2017"
    ):
        """
        Initialize the visualizer.
        
        Args:
            experiment_dir: Path to experiment directory containing model and results
            cache_dir: Directory with cached features/predictions
            coco_images_dir: Directory containing COCO validation images
        """
        self.experiment_dir = Path(experiment_dir)
        self.cache_dir = Path(cache_dir)
        self.coco_images_dir = Path(coco_images_dir)
        
        # Load model and configuration
        self.model, self.config, self.tau = self._load_model()
        
        # Load cached data
        self.data = load_cached_data(str(cache_dir))
        
        # COCO class names (subset of common classes)
        self.class_names = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
            4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
            8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
            13: 'stop sign', 14: 'parking meter', 15: 'bench',
            16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse',
            20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
            24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella'
        }
        
    def _load_model(self) -> Tuple[torch.nn.Module, Dict, float]:
        """Load the trained model, configuration, and final tau value."""
        model_dir = self.experiment_dir / "models"
        
        # Load best model
        model_path = model_dir / "best_model.pt"
        if not model_path.exists():
            # Try to find any model file
            model_files = list(model_dir.glob("best_model_*.pt"))
            if model_files:
                model_path = model_files[0]
            else:
                raise FileNotFoundError(f"No model found in {model_dir}")
        
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Load configuration
        config_path = self.experiment_dir / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = checkpoint.get('config', {})
        
        # Initialize model
        model_config = checkpoint.get('model_config', {})
        model = SymmetricAdaptiveMLP(
            input_dim=model_config.get('input_dim', 17),  # Default feature dim
            hidden_dims=model_config.get('hidden_dims', [128, 128]),
            activation=model_config.get('activation', 'elu'),
            dropout_rate=model_config.get('dropout_rate', 0.1),
            use_batch_norm=model_config.get('use_batch_norm', True)
        )
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Get tau value
        tau = checkpoint.get('tau', 1.0)
        
        return model, config, tau
    
    def get_predictions_for_image(self, img_name: str = None, img_idx: int = None) -> Dict:
        """Get predictions and intervals for a specific image.
        
        Args:
            img_name: COCO image name (not currently supported due to sequential IDs)
            img_idx: Sequential image index in the dataset
        """
        val_data = self.data['val_data']
        
        # Check if we have image IDs
        if 'img_ids' in val_data:
            if img_idx is not None:
                # Use sequential index
                mask = val_data['img_ids'] == img_idx
                print(f"Using sequential index {img_idx}, found {mask.sum().item()} predictions")
            else:
                # For now, use a default image with good predictions
                img_idx = 0  # First image has 12 predictions
                mask = val_data['img_ids'] == img_idx
                print(f"No image specified, using index {img_idx} with {mask.sum().item()} predictions")
            
            if mask.sum() == 0:
                print(f"No predictions found for image index {img_idx}")
                # Fallback to first 10 predictions
                sample_indices = range(min(10, len(val_data['features'])))
            else:
                sample_indices = torch.where(mask)[0].numpy()
        else:
            # No image IDs, use first 10 predictions
            print("Warning: No image IDs in cache, using first 10 predictions")
            sample_indices = range(min(10, len(val_data['features'])))
        
        features = val_data['features'][sample_indices]
        pred_boxes = val_data['pred_coords'][sample_indices]
        gt_boxes = val_data['gt_coords'][sample_indices]
        confidences = val_data['confidence'][sample_indices]
        
        # Get predicted widths from model
        with torch.no_grad():
            predicted_widths = self.model(features)
            scaled_widths = predicted_widths * self.tau
        
        # Create prediction intervals
        lower_bounds = pred_boxes - scaled_widths
        upper_bounds = pred_boxes + scaled_widths
        
        return {
            'pred_boxes': pred_boxes.numpy(),
            'gt_boxes': gt_boxes.numpy(),
            'lower_bounds': lower_bounds.numpy(),
            'upper_bounds': upper_bounds.numpy(),
            'widths': scaled_widths.numpy(),
            'confidences': confidences.numpy()
        }
    
    def plot_boxes_on_image(
        self,
        img_name: str = None,
        img_idx: int = None,
        save_path: Optional[str] = None,
        max_boxes: int = 10
    ):
        """
        Plot prediction boxes with intervals on an image.
        
        Args:
            img_name: Name of the image (without extension)
            img_idx: Sequential image index
            save_path: Path to save the plot
            max_boxes: Maximum number of boxes to display
        """
        # For now, use a placeholder image since we can't match to COCO images
        if img_idx is not None:
            # Create a blank canvas
            img_array = np.ones((600, 800, 3), dtype=np.uint8) * 255
            img_name = f"seq_idx_{img_idx}"
        else:
            # Load image
            img_path = self.coco_images_dir / f"{img_name}.jpg"
            if not img_path.exists():
                print(f"Image {img_path} not found")
                return
            
            img = Image.open(img_path)
            img_array = np.array(img)
        
        # Get predictions
        preds = self.get_predictions_for_image(img_name, img_idx)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(img_array)
        
        # Plot boxes
        n_boxes = min(len(preds['pred_boxes']), max_boxes)
        colors = plt.cm.Set3(np.linspace(0, 1, n_boxes))
        
        for i in range(n_boxes):
            # Prediction box
            pred_box = preds['pred_boxes'][i]
            x1, y1, x2, y2 = pred_box
            width = x2 - x1
            height = y2 - y1
            
            # Draw prediction box
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor=colors[i], facecolor='none',
                linestyle='-', label=f'Pred {i+1}'
            )
            ax.add_patch(rect)
            
            # Draw uncertainty interval
            lower = preds['lower_bounds'][i]
            upper = preds['upper_bounds'][i]
            
            # Create uncertainty region
            x1_lower, y1_lower, x2_lower, y2_lower = lower
            x1_upper, y1_upper, x2_upper, y2_upper = upper
            
            # Draw outer uncertainty box
            outer_rect = patches.Rectangle(
                (x1_lower, y1_lower),
                x2_upper - x1_lower,
                y2_upper - y1_lower,
                linewidth=1, edgecolor=colors[i], facecolor=colors[i],
                alpha=0.2, linestyle='--'
            )
            ax.add_patch(outer_rect)
            
            # Add confidence score
            conf = preds['confidences'][i]
            ax.text(x1, y1-5, f'{conf:.2f}', color=colors[i], fontsize=10,
                   weight='bold', bbox=dict(facecolor='white', alpha=0.7))
        
        # Add ground truth boxes if available
        for i in range(min(len(preds['gt_boxes']), max_boxes)):
            gt_box = preds['gt_boxes'][i]
            x1, y1, x2, y2 = gt_box
            width = x2 - x1
            height = y2 - y1
            
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor='green', facecolor='none',
                linestyle=':', label='GT' if i == 0 else ''
            )
            ax.add_patch(rect)
        
        ax.set_xlim(0, img_array.shape[1])
        ax.set_ylim(img_array.shape[0], 0)
        ax.set_title(f'Symmetric Adaptive Predictions - {img_name}')
        ax.axis('off')
        
        # Add legend
        handles = [
            patches.Patch(color='green', label='Ground Truth'),
            patches.Patch(color='blue', label='Prediction'),
            patches.Patch(color='blue', alpha=0.3, label='Uncertainty Interval')
        ]
        ax.legend(handles=handles, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_size_stratified_performance(self, save_path: Optional[str] = None):
        """Plot size-stratified coverage and MPIW metrics."""
        # Load size-stratified results from CSV
        # Look in both experiment dir and logs dir
        experiment_name = self.experiment_dir.name
        log_dir = Path("/ssd_4TB/divake/conformal-od/learnable_scoring_fn/logs/symmetric") / experiment_name
        
        csv_paths = list(self.experiment_dir.glob("**/csv/*_size_stratified.csv"))
        if not csv_paths and log_dir.exists():
            csv_paths = list(log_dir.glob("**/csv/*_size_stratified.csv"))
        
        if not csv_paths:
            print("No size-stratified results found")
            return
        
        csv_path = csv_paths[0]
        
        import pandas as pd
        df = pd.read_csv(csv_path)
        
        # Get final epoch results
        final_epoch = df['epoch'].max()
        final_df = df[(df['epoch'] == final_epoch) & (df['phase'] == 'val')]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Coverage by size
        sizes = ['small', 'medium', 'large']
        coverages = [final_df[final_df['size_category'] == s]['coverage'].values[0] for s in sizes]
        counts = [final_df[final_df['size_category'] == s]['count'].values[0] for s in sizes]
        
        bars1 = ax1.bar(sizes, coverages, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax1.axhline(y=0.9, color='r', linestyle='--', label='Target (90%)')
        ax1.set_xlabel('Object Size Category')
        ax1.set_ylabel('Coverage Rate')
        ax1.set_title('Coverage by Object Size')
        ax1.set_ylim(0.7, 1.0)
        
        # Add value labels
        for bar, val, cnt in zip(bars1, coverages, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.1%}\n(n={cnt})', ha='center', va='bottom')
        
        # Plot 2: MPIW by size
        mpiws = [final_df[final_df['size_category'] == s]['avg_mpiw'].values[0] for s in sizes]
        
        bars2 = ax2.bar(sizes, mpiws, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax2.set_xlabel('Object Size Category')
        ax2.set_ylabel('Average MPIW (pixels)')
        ax2.set_title('Mean Prediction Interval Width by Object Size')
        
        # Add value labels
        for bar, val in zip(bars2, mpiws):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}', ha='center', va='bottom')
        
        plt.suptitle('Size-Aware Symmetric Adaptive Performance', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_training_evolution(self, save_path: Optional[str] = None):
        """Plot training evolution metrics."""
        # Load metrics from CSV
        # Look in both experiment dir and logs dir
        experiment_name = self.experiment_dir.name
        log_dir = Path("/ssd_4TB/divake/conformal-od/learnable_scoring_fn/logs/symmetric") / experiment_name
        
        csv_paths = list(self.experiment_dir.glob("**/csv/*_metrics.csv"))
        if not csv_paths and log_dir.exists():
            csv_paths = list(log_dir.glob("**/csv/*_metrics.csv"))
        
        if not csv_paths:
            print("No metrics CSV found")
            return
        
        csv_path = csv_paths[0]
        
        import pandas as pd
        df = pd.read_csv(csv_path)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Coverage evolution
        ax = axes[0, 0]
        val_df = df[df['phase'] == 'val']
        ax.plot(val_df['epoch'], val_df['coverage_rate'], 'b-', linewidth=2)
        ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='Target')
        ax.fill_between(val_df['epoch'], 0.88, 0.905, alpha=0.2, color='green', label='Target Range')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Coverage Rate')
        ax.set_title('Coverage Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. MPIW evolution
        ax = axes[0, 1]
        ax.plot(val_df['epoch'], val_df['avg_mpiw'], 'm-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Average MPIW (pixels)')
        ax.set_title('Mean Prediction Interval Width')
        ax.grid(True, alpha=0.3)
        
        # 3. Loss evolution
        ax = axes[1, 0]
        train_df = df[df['phase'] == 'train']
        if len(train_df) > 0:
            ax.plot(train_df['epoch'], train_df['loss'], 'b-', label='Train', alpha=0.7)
        ax.plot(val_df['epoch'], val_df['loss'], 'r-', label='Validation', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        ax.set_title('Loss Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Tau evolution
        ax = axes[1, 1]
        tau_csv = list(self.experiment_dir.glob("**/csv/*_tau.csv"))
        if not tau_csv and log_dir.exists():
            tau_csv = list(log_dir.glob("**/csv/*_tau.csv"))
        if tau_csv:
            tau_df = pd.read_csv(tau_csv[0])
            ax.plot(tau_df['epoch'], tau_df['new_tau'], 'c-', linewidth=2)
            ax.axhline(y=1.0, color='k', linestyle=':', alpha=0.5)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Tau')
            ax.set_title('Calibration Factor Evolution')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Training Evolution - Symmetric Size-Aware Adaptive', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_width_distribution(self, save_path: Optional[str] = None):
        """Plot distribution of predicted widths."""
        # Get validation data
        val_features = self.data['val_data']['features']
        val_gt = self.data['val_data']['gt_coords']
        
        # Get predictions
        with torch.no_grad():
            predicted_widths = self.model(val_features)
            scaled_widths = predicted_widths * self.tau
        
        # Calculate object sizes
        box_widths = val_gt[:, 2] - val_gt[:, 0]
        box_heights = val_gt[:, 3] - val_gt[:, 1]
        object_sizes = torch.sqrt(box_widths * box_heights)
        
        # Categorize by size
        small_mask = object_sizes < 32
        medium_mask = (object_sizes >= 32) & (object_sizes < 96)
        large_mask = object_sizes >= 96
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot width distributions for each coordinate
        coord_names = ['x1', 'y1', 'x2', 'y2']
        for i, (ax, coord) in enumerate(zip(axes.flat, coord_names)):
            # Get widths for this coordinate
            widths = scaled_widths[:, i].cpu().numpy()
            
            # Plot overall distribution
            ax.hist(widths, bins=50, alpha=0.4, color='gray', label='All', density=True)
            
            # Plot by size category
            if small_mask.any():
                ax.hist(widths[small_mask], bins=30, alpha=0.6, color='blue', 
                       label='Small', density=True, histtype='step', linewidth=2)
            if medium_mask.any():
                ax.hist(widths[medium_mask], bins=30, alpha=0.6, color='orange',
                       label='Medium', density=True, histtype='step', linewidth=2)
            if large_mask.any():
                ax.hist(widths[large_mask], bins=30, alpha=0.6, color='green',
                       label='Large', density=True, histtype='step', linewidth=2)
            
            ax.set_xlabel(f'Width for {coord} (pixels)')
            ax.set_ylabel('Density')
            ax.set_title(f'Width Distribution - {coord}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Predicted Width Distributions by Object Size', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def create_summary_report(self, output_dir: str, img_idx: int = None):
        """Create a comprehensive summary report with all visualizations."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("Creating summary report...")
        
        # 1. Plot boxes on sample image
        print("1. Plotting prediction boxes on image...")
        if img_idx is not None:
            self.plot_boxes_on_image(
                img_idx=img_idx,
                save_path=output_path / "sample_predictions.png"
            )
        else:
            self.plot_boxes_on_image(
                "000000054593",
                save_path=output_path / "sample_predictions.png"
            )
        
        # 2. Size-stratified performance
        print("2. Creating size-stratified performance plots...")
        self.plot_size_stratified_performance(
            save_path=output_path / "size_stratified_performance.png"
        )
        
        # 3. Training evolution
        print("3. Plotting training evolution...")
        self.plot_training_evolution(
            save_path=output_path / "training_evolution.png"
        )
        
        # 4. Width distributions
        print("4. Analyzing width distributions...")
        self.plot_width_distribution(
            save_path=output_path / "width_distributions.png"
        )
        
        # 5. Create summary statistics
        print("5. Computing summary statistics...")
        self._create_summary_stats(output_path)
        
        print(f"\nSummary report saved to: {output_path}")
    
    def _create_summary_stats(self, output_path: Path):
        """Create summary statistics file."""
        # Load final results
        final_results_path = self.experiment_dir / "final_results.json"
        if final_results_path.exists():
            with open(final_results_path, 'r') as f:
                final_results = json.load(f)
        else:
            final_results = {}
        
        # Add model information
        summary = {
            'experiment_name': self.experiment_dir.name,
            'model_info': {
                'architecture': self.model.model_name,
                'parameters': sum(p.numel() for p in self.model.parameters()),
                'final_tau': float(self.tau)
            },
            'performance': {
                'final_coverage': final_results.get('final_coverage', 'N/A'),
                'final_mpiw': final_results.get('final_mpiw', 'N/A'),
                'size_metrics': final_results.get('size_metrics', {})
            },
            'config': self.config
        }
        
        # Save summary
        with open(output_path / "summary_stats.json", 'w') as f:
            json.dump(summary, f, indent=2)


def main():
    """Main function to run visualizations."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize Symmetric Size-Aware Results')
    parser.add_argument('--experiment_dir', type=str, required=True,
                       help='Path to experiment directory')
    parser.add_argument('--output_dir', type=str, default='./visualization_results',
                       help='Directory to save visualizations')
    parser.add_argument('--image_name', type=str, default='000000054593',
                       help='COCO image name for box visualization')
    parser.add_argument('--image_idx', type=int, default=None,
                       help='Sequential image index in dataset (use this for now)')
    parser.add_argument('--cache_dir', type=str, 
                       default='/ssd_4TB/divake/conformal-od/learnable_scoring_fn/cache_base_model',
                       help='Cache directory with features')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = SymmetricAdaptiveVisualizer(
        experiment_dir=args.experiment_dir,
        cache_dir=args.cache_dir
    )
    
    # Create comprehensive report
    visualizer.create_summary_report(args.output_dir, img_idx=args.image_idx)
    
    # Also create individual plot for the specified image
    if args.image_idx is not None:
        print(f"\nCreating detailed visualization for image index: {args.image_idx}")
        visualizer.plot_boxes_on_image(
            img_idx=args.image_idx,
            save_path=Path(args.output_dir) / f"detailed_idx_{args.image_idx}.png"
        )
    elif args.image_name:
        print(f"\nCreating detailed visualization for image: {args.image_name}")
        visualizer.plot_boxes_on_image(
            args.image_name,
            save_path=Path(args.output_dir) / f"detailed_{args.image_name}.png"
        )


if __name__ == "__main__":
    main()