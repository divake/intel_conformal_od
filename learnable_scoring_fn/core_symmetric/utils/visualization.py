"""Visualization utilities for symmetric adaptive training."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def plot_training_results(
    history: Dict[str, List],
    save_path: Optional[Path] = None,
    show: bool = True
):
    """
    Create comprehensive training result plots.
    
    Args:
        history: Dictionary with training history
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    
    # 1. Loss curves
    ax = axes[0, 0]
    if 'train_loss' in history and 'val_loss' in history:
        epochs = range(1, len(history['train_loss']) + 1)
        ax.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
        ax.plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 2. Coverage evolution
    ax = axes[0, 1]
    if 'val_coverage' in history:
        epochs = range(1, len(history['val_coverage']) + 1)
        ax.plot(epochs, history['val_coverage'], 'g-', linewidth=2)
        ax.axhline(y=0.9, color='r', linestyle='--', label='Target (90%)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Coverage Rate')
        ax.set_title('Validation Coverage Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.7, 1.0)
    
    # 3. MPIW evolution
    ax = axes[1, 0]
    if 'val_mpiw' in history:
        epochs = range(1, len(history['val_mpiw']) + 1)
        ax.plot(epochs, history['val_mpiw'], 'm-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MPIW (pixels)')
        ax.set_title('Mean Prediction Interval Width')
        ax.grid(True, alpha=0.3)
    
    # 4. Tau evolution
    ax = axes[1, 1]
    if 'tau' in history:
        epochs = range(1, len(history['tau']) + 1)
        ax.plot(epochs, history['tau'], 'c-', linewidth=2)
        ax.axhline(y=1.0, color='k', linestyle=':', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Tau')
        ax.set_title('Calibration Factor (Tau) Evolution')
        ax.grid(True, alpha=0.3)
    
    # 5. Coverage vs MPIW scatter
    ax = axes[2, 0]
    if 'val_coverage' in history and 'val_mpiw' in history:
        coverages = history['val_coverage']
        mpiws = history['val_mpiw']
        
        # Color by epoch
        scatter = ax.scatter(coverages, mpiws, c=range(len(coverages)), 
                           cmap='viridis', s=100, alpha=0.7, edgecolors='black')
        
        # Mark start and end
        ax.scatter(coverages[0], mpiws[0], color='green', s=200, 
                  marker='s', label='Start', edgecolors='black', linewidth=2)
        ax.scatter(coverages[-1], mpiws[-1], color='red', s=200, 
                  marker='*', label='End', edgecolors='black', linewidth=2)
        
        ax.axvline(x=0.9, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Coverage Rate')
        ax.set_ylabel('MPIW (pixels)')
        ax.set_title('Coverage-Efficiency Tradeoff Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Epoch')
    
    # 6. Learning rate schedule
    ax = axes[2, 1]
    if 'learning_rate' in history:
        epochs = range(1, len(history['learning_rate']) + 1)
        ax.plot(epochs, history['learning_rate'], 'orange', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_tau_evolution(
    tau_history: List[float],
    coverage_history: Optional[List[float]] = None,
    save_path: Optional[Path] = None,
    show: bool = True
):
    """
    Plot detailed tau evolution.
    
    Args:
        tau_history: List of tau values
        coverage_history: Optional list of coverage values
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    steps = range(len(tau_history))
    
    # Plot tau
    ax.plot(steps, tau_history, 'b-', linewidth=2, label='Tau')
    ax.axhline(y=1.0, color='k', linestyle=':', alpha=0.5, label='Tau=1.0')
    
    # Highlight regions
    tau_array = np.array(tau_history)
    ax.fill_between(steps, 1.0, tau_array, where=(tau_array > 1.0),
                    alpha=0.2, color='red', label='Over-correction')
    ax.fill_between(steps, tau_array, 1.0, where=(tau_array < 1.0),
                    alpha=0.2, color='blue', label='Under-correction')
    
    ax.set_xlabel('Calibration Step')
    ax.set_ylabel('Tau Value')
    ax.set_title('Tau Evolution During Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add coverage on secondary axis if provided
    if coverage_history and len(coverage_history) == len(tau_history):
        ax2 = ax.twinx()
        ax2.plot(steps, coverage_history, 'g--', linewidth=1.5, alpha=0.7, label='Coverage')
        ax2.axhline(y=0.9, color='g', linestyle=':', alpha=0.5)
        ax2.set_ylabel('Coverage Rate', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        ax2.set_ylim(0.7, 1.0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_width_distribution(
    widths: torch.Tensor,
    object_sizes: Optional[torch.Tensor] = None,
    save_path: Optional[Path] = None,
    show: bool = True
):
    """
    Plot distribution of predicted widths.
    
    Args:
        widths: Predicted widths [N, 4]
        object_sizes: Optional object sizes for stratification
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    coord_names = ['x0', 'y0', 'x1', 'y1']
    
    for i, (ax, name) in enumerate(zip(axes.flat, coord_names)):
        width_data = widths[:, i].cpu().numpy()
        
        # Histogram
        ax.hist(width_data, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(x=width_data.mean(), color='red', linestyle='--', 
                  linewidth=2, label=f'Mean: {width_data.mean():.1f}')
        ax.set_xlabel(f'Width for {name} (pixels)')
        ax.set_ylabel('Count')
        ax.set_title(f'Distribution of Widths - {name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_size_stratified_results(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[Path] = None,
    show: bool = True
):
    """
    Plot size-stratified coverage and MPIW results.
    
    Args:
        results: Dictionary with size categories and their metrics
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    categories = list(results.keys())
    coverages = [results[cat]['coverage'] for cat in categories]
    mpiws = [results[cat]['mpiw'] for cat in categories]
    
    # Coverage by size
    bars1 = ax1.bar(categories, coverages, color='skyblue', edgecolor='black')
    ax1.axhline(y=0.9, color='r', linestyle='--', label='Target (90%)')
    ax1.set_xlabel('Object Size Category')
    ax1.set_ylabel('Coverage Rate')
    ax1.set_title('Coverage by Object Size')
    ax1.legend()
    ax1.set_ylim(0.7, 1.0)
    
    # Add value labels on bars
    for bar, val in zip(bars1, coverages):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom')
    
    # MPIW by size
    bars2 = ax2.bar(categories, mpiws, color='lightcoral', edgecolor='black')
    ax2.set_xlabel('Object Size Category')
    ax2.set_ylabel('Average MPIW (pixels)')
    ax2.set_title('MPIW by Object Size')
    
    # Add value labels on bars
    for bar, val in zip(bars2, mpiws):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()