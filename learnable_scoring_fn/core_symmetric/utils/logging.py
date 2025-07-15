"""Comprehensive logging utilities for symmetric adaptive training."""

import os
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


class AdaptiveConformalLogger:
    """
    Comprehensive logger for symmetric adaptive conformal prediction training.
    
    Logs every aspect of training including:
    - Epoch-level metrics
    - Phase transitions (train/calibrate/validate)
    - Tau evolution
    - Coverage and MPIW statistics
    - Size-stratified metrics
    """
    
    def __init__(self, log_dir: str, experiment_name: Optional[str] = None):
        """
        Initialize the logger.
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Optional experiment name (auto-generated if None)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = f"symmetric_adaptive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_name = experiment_name
        
        # Create experiment-specific subdirectories
        self.experiment_dir = self.log_dir / experiment_name
        self.experiment_dir.mkdir(exist_ok=True)
        
        self.csv_dir = self.experiment_dir / "csv"
        self.json_dir = self.experiment_dir / "json"
        self.plot_dir = self.experiment_dir / "plots"
        
        for dir in [self.csv_dir, self.json_dir, self.plot_dir]:
            dir.mkdir(exist_ok=True)
        
        # Initialize CSV files
        self._init_csv_files()
        
        # In-memory storage for plotting
        self.history = defaultdict(list)
        
        # Current epoch state
        self.current_epoch = 0
        self.current_phase = None
        
    def _init_csv_files(self):
        """Initialize CSV files with headers."""
        # Main metrics CSV
        self.metrics_csv = self.csv_dir / f"{self.experiment_name}_metrics.csv"
        with open(self.metrics_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'phase', 'timestamp', 'tau', 'loss', 'coverage_loss', 
                'efficiency_loss', 'coverage_rate', 'avg_mpiw', 'normalized_mpiw'
            ])
        
        # Tau evolution CSV
        self.tau_csv = self.csv_dir / f"{self.experiment_name}_tau.csv"
        with open(self.tau_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'old_tau', 'new_tau', 'coverage_rate'])
        
        # Size-stratified CSV
        self.size_csv = self.csv_dir / f"{self.experiment_name}_size_stratified.csv"
        with open(self.size_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'phase', 'size_category', 'coverage', 'avg_mpiw', 'count'
            ])
    
    def log_epoch_start(self, epoch: int, tau: float):
        """Log the start of an epoch."""
        self.current_epoch = epoch
        
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch} STARTED")
        print(f"Current tau: {tau:.4f}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        
        # Store in history
        self.history['epoch'].append(epoch)
        self.history['tau'].append(tau)
    
    def log_training_phase(self, epoch: int, batch_idx: int, batch_metrics: Dict[str, float]):
        """Log training phase metrics."""
        self.current_phase = 'train'
        
        # Log every 100 batches
        if batch_idx % 100 == 0:
            print(f"[Train] Epoch {epoch}, Batch {batch_idx}:")
            print(f"  Loss: {batch_metrics['total']:.4f} "
                  f"(coverage: {batch_metrics['coverage']:.4f}, "
                  f"efficiency: {batch_metrics['efficiency']:.4f})")
            print(f"  Coverage rate: {batch_metrics['coverage_rate']:.3f}")
            print(f"  Avg MPIW: {batch_metrics['avg_mpiw']:.2f} pixels")
            print(f"  Normalized MPIW: {batch_metrics['normalized_mpiw']:.3f}")
            
            if 'avg_widths' in batch_metrics:
                avg_widths = batch_metrics['avg_widths']
                print(f"  Width stats: {avg_widths.detach().cpu().numpy()}")
    
    def log_epoch_metrics(self, epoch: int, phase: str, metrics: Dict[str, Any]):
        """Log metrics for a complete phase."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Write to CSV
        with open(self.metrics_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, phase, timestamp, metrics.get('tau', self.history['tau'][-1]),
                metrics.get('total', 0), metrics.get('coverage', 0),
                metrics.get('efficiency', 0), metrics.get('coverage_rate', 0),
                metrics.get('avg_mpiw', 0), metrics.get('normalized_mpiw', 0)
            ])
        
        # Store in history
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.history[f"{phase}_{key}"].append(value)
        
        # Print summary
        print(f"\n[{phase.upper()}] Epoch {epoch} Summary:")
        print(f"  Average loss: {metrics.get('total', 0):.4f}")
        print(f"  Coverage rate: {metrics.get('coverage_rate', 0):.3f}")
        print(f"  Average MPIW: {metrics.get('avg_mpiw', 0):.2f} pixels")
        print(f"  Normalized MPIW: {metrics.get('normalized_mpiw', 0):.3f}")
    
    def log_calibration_phase(
        self, 
        epoch: int, 
        old_tau: float, 
        new_tau: float,
        calibration_stats: Dict[str, float]
    ):
        """Log calibration phase results."""
        self.current_phase = 'calibration'
        
        print(f"\n[CALIBRATION] Epoch {epoch}:")
        print(f"  Old tau: {old_tau:.4f}")
        print(f"  New tau: {new_tau:.4f} (change: {(new_tau/old_tau - 1)*100:+.1f}%)")
        print(f"  Calibration coverage: {calibration_stats['actual_coverage']:.3f}")
        print(f"  Calibration MPIW: {calibration_stats['avg_mpiw']:.2f}")
        
        # Write to tau CSV
        with open(self.tau_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, old_tau, new_tau, calibration_stats['actual_coverage']
            ])
        
        # Store tau evolution
        self.history['tau_evolution'].append(new_tau)
    
    def log_validation_phase(
        self,
        epoch: int,
        metrics: Dict[str, Any],
        size_stratified_metrics: Optional[Dict[str, Dict]] = None
    ):
        """Log validation phase with detailed metrics."""
        self.current_phase = 'validation'
        
        # Log overall metrics
        self.log_epoch_metrics(epoch, 'val', metrics)
        
        # Log size-stratified metrics if provided
        if size_stratified_metrics:
            print("\n  Size-stratified results:")
            for size_cat, size_metrics in size_stratified_metrics.items():
                coverage = size_metrics.get('coverage', 0)
                mpiw = size_metrics.get('mpiw', 0)
                count = size_metrics.get('count', 0)
                
                print(f"    {size_cat}: coverage={coverage:.3f}, "
                      f"MPIW={mpiw:.1f}, n={count}")
                
                # Write to CSV
                with open(self.size_csv, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch, 'val', size_cat, coverage, mpiw, count
                    ])
    
    def log_best_model(self, epoch: int, reason: str, metrics: Dict[str, float]):
        """Log when a best model is saved."""
        print(f"\n[CHECKPOINT] Saving best model at epoch {epoch}")
        print(f"  Reason: {reason}")
        print(f"  Coverage: {metrics.get('coverage_rate', 0):.3f}")
        print(f"  MPIW: {metrics.get('avg_mpiw', 0):.2f}")
    
    def create_visualization(self, epoch: int, update_frequency: int = 1):
        """Create and save visualization plots.
        
        Args:
            epoch: Current epoch number
            update_frequency: How often to update the plot (default: every epoch)
        """
        if epoch % update_frequency != 0 and epoch != self.history['epoch'][-1]:
            return  # Skip plotting based on frequency
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Tau evolution
        ax = axes[0, 0]
        if 'tau_evolution' in self.history:
            ax.plot(self.history['tau_evolution'], 'b-', linewidth=2)
            ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
            ax.set_xlabel('Calibration Step')
            ax.set_ylabel('Tau')
            ax.set_title('Tau Evolution Over Training')
            ax.grid(True, alpha=0.3)
        
        # 2. Coverage vs MPIW tradeoff
        ax = axes[0, 1]
        if 'val_coverage_rate' in self.history and 'val_avg_mpiw' in self.history:
            coverages = self.history['val_coverage_rate']
            mpiws = self.history['val_avg_mpiw']
            
            # Color by epoch
            scatter = ax.scatter(coverages, mpiws, c=range(len(coverages)), 
                               cmap='viridis', s=50, alpha=0.7)
            ax.axvline(x=0.9, color='r', linestyle='--', alpha=0.5, label='Target coverage')
            ax.set_xlabel('Coverage Rate')
            ax.set_ylabel('Average MPIW (pixels)')
            ax.set_title('Coverage vs Efficiency Tradeoff')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Epoch')
        
        # 3. Loss curves
        ax = axes[1, 0]
        if 'train_total' in self.history:
            ax.plot(self.history.get('train_total', []), label='Train', alpha=0.8)
            ax.plot(self.history.get('val_total', []), label='Validation', alpha=0.8)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Total Loss')
            ax.set_title('Training and Validation Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 4. Width distribution (if available)
        ax = axes[1, 1]
        ax.text(0.5, 0.5, 'Width Distribution\n(Updated during training)', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Width Distribution by Object Size')
        
        plt.tight_layout()
        # Save to a single file that gets updated each time
        plt.savefig(self.plot_dir / f"{self.experiment_name}_training_progress.png",
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_final_summary(self):
        """Save final training summary."""
        summary = {
            'experiment_name': self.experiment_name,
            'total_epochs': len(self.history['epoch']),
            'final_tau': self.history['tau_evolution'][-1] if self.history['tau_evolution'] else 1.0,
            'final_coverage': self.history['val_coverage_rate'][-1] if self.history['val_coverage_rate'] else 0,
            'final_mpiw': self.history['val_avg_mpiw'][-1] if self.history['val_avg_mpiw'] else 0,
            'tau_range': [min(self.history['tau_evolution']), max(self.history['tau_evolution'])] 
                         if self.history['tau_evolution'] else [1.0, 1.0],
            'coverage_range': [min(self.history['val_coverage_rate']), max(self.history['val_coverage_rate'])]
                             if self.history['val_coverage_rate'] else [0, 0],
            'training_completed': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save as JSON
        with open(self.json_dir / f"{self.experiment_name}_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETED")
        print(f"Final tau: {summary['final_tau']:.4f}")
        print(f"Final coverage: {summary['final_coverage']:.3f}")
        print(f"Final MPIW: {summary['final_mpiw']:.2f}")
        print(f"{'='*60}")
    
    def log_error(self, message: str, exception: Optional[Exception] = None):
        """Log error messages."""
        error_msg = f"[ERROR] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {message}"
        if exception:
            error_msg += f"\nException: {str(exception)}"
        
        print(error_msg)
        
        # Also save to error log
        error_file = self.log_dir / "errors.log"
        with open(error_file, 'a') as f:
            f.write(error_msg + "\n\n")