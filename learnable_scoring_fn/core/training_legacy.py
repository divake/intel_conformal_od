"""Core training functionality for learnable scoring functions."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import json
import logging
import time
from typing import Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt

from .loss import RegressionCoverageLoss, AdaptiveCoverageLoss


def train_model(
    model: nn.Module,
    train_data: Dict[str, torch.Tensor],
    val_data: Dict[str, torch.Tensor],
    config: Dict[str, Any],
    output_dir: Path,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """Train a scoring function model.
    
    Args:
        model: Model to train
        train_data: Training data dictionary with keys:
            - features: [n_train, input_dim]
            - gt_coords: [n_train, 4]
            - pred_coords: [n_train, 4]
            - confidence: [n_train]
        val_data: Validation data (same structure as train_data)
        config: Training configuration
        output_dir: Output directory for saving results
        logger: Optional logger
        
    Returns:
        Dictionary with training results
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Setup device
    device = next(model.parameters()).device
    
    # Create dataloaders
    train_dataset = TensorDataset(
        train_data['features'],
        train_data['pred_coords'],
        train_data['gt_coords']
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['TRAINING']['BATCH_SIZE'],
        shuffle=True
    )
    
    # Split validation data into calibration and test if needed
    if 'calib_indices' in val_data:
        calib_idx = val_data['calib_indices']
        test_idx = val_data['test_indices']
        
        calib_features = val_data['features'][calib_idx]
        calib_pred = val_data['pred_coords'][calib_idx]
        calib_gt = val_data['gt_coords'][calib_idx]
        
        test_features = val_data['features'][test_idx]
        test_pred = val_data['pred_coords'][test_idx]
        test_gt = val_data['gt_coords'][test_idx]
    else:
        # Use all validation data as test
        test_features = val_data['features']
        test_pred = val_data['pred_coords']
        test_gt = val_data['gt_coords']
        
        # No calibration data
        calib_features = test_features[:100]  # Small subset
        calib_pred = test_pred[:100]
        calib_gt = test_gt[:100]
    
    # Setup optimizer
    lr = float(config['TRAINING']['LEARNING_RATE'])
    weight_decay = float(config['TRAINING']['WEIGHT_DECAY'])
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Setup scheduler
    scheduler_config = config['TRAINING'].get('LR_SCHEDULER', {})
    if scheduler_config.get('TYPE') == 'cosine':
        T_max = int(config['TRAINING']['NUM_EPOCHS'])
        eta_min = float(scheduler_config.get('ETA_MIN', 1e-6))
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min
        )
    elif scheduler_config.get('TYPE') == 'reduce_on_plateau':
        # Ensure all scheduler parameters are numeric
        factor = float(scheduler_config.get('FACTOR', 0.5))
        patience = int(scheduler_config.get('PATIENCE', 10))
        min_lr = float(scheduler_config.get('MIN_LR', 1e-6))
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=factor,
            patience=patience,
            min_lr=min_lr
        )
    else:
        scheduler = None
    
    # Loss function - use adaptive loss if specified
    use_adaptive_loss = config.get('USE_ADAPTIVE_LOSS', False)
    
    if use_adaptive_loss:
        criterion = AdaptiveCoverageLoss(
            target_coverage=config['LOSS']['TARGET_COVERAGE'],
            efficiency_weight=config['LOSS']['EFFICIENCY_WEIGHT'],
            ranking_weight=config['LOSS'].get('RANKING_WEIGHT', 0.05),
            variance_weight=config['LOSS'].get('VARIANCE_WEIGHT', 0.02),
            smoothness_weight=config['LOSS'].get('SMOOTHNESS_WEIGHT', 0.01)
        )
        logger.info("Using AdaptiveCoverageLoss for training")
    else:
        criterion = RegressionCoverageLoss(
            target_coverage=config['LOSS']['TARGET_COVERAGE'],
            efficiency_weight=config['LOSS']['EFFICIENCY_WEIGHT'],
            calibration_weight=config['LOSS']['CALIBRATION_WEIGHT']
        )
        logger.info("Using RegressionCoverageLoss for training")
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_coverage': [],
        'val_avg_width': [],
        'learning_rate': []
    }
    
    best_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    early_stopping_patience = config['TRAINING'].get('EARLY_STOPPING_PATIENCE', 20)
    
    # Fixed tau for regression
    tau = torch.tensor(1.0, device=device)
    
    logger.info(f"Starting training for {config['TRAINING']['NUM_EPOCHS']} epochs")
    
    for epoch in range(config['TRAINING']['NUM_EPOCHS']):
        # Training phase
        model.train()
        train_losses = []
        
        for batch_features, batch_pred, batch_gt in train_loader:
            # Forward pass
            scores = model(batch_features)
            
            # Call loss function with appropriate arguments
            if use_adaptive_loss:
                # AdaptiveCoverageLoss expects features for smoothness
                losses = criterion(scores, batch_gt, batch_pred, batch_features)
            else:
                # RegressionCoverageLoss expects tau
                losses = criterion(scores, batch_gt, batch_pred, tau)
            
            # Backward pass
            optimizer.zero_grad()
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['TRAINING']['GRAD_CLIP_NORM']
            )
            
            optimizer.step()
            
            train_losses.append(losses['total'].item())
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            # Test set evaluation
            test_scores = model(test_features)
            
            # Call loss function with appropriate arguments
            if use_adaptive_loss:
                test_losses = criterion(test_scores, test_gt, test_pred, test_features)
            else:
                test_losses = criterion(test_scores, test_gt, test_pred, tau)
            
            # Calculate coverage
            # For adaptive loss, scores are nonconformity scores, not widths
            if use_adaptive_loss:
                # Use scores directly as intervals for coverage calculation
                # This is a simplification - in practice you'd calibrate first
                test_avg_width = test_scores.mean().item()
                test_coverage = test_losses['actual_coverage'].item()
            else:
                # Legacy behavior - scores are widths
                interval_half_widths = test_scores * tau
                lower_bounds = test_pred - interval_half_widths.expand(-1, 4)
                upper_bounds = test_pred + interval_half_widths.expand(-1, 4)
                
                test_covered = ((test_gt >= lower_bounds) & (test_gt <= upper_bounds)).all(dim=1).float()
                test_coverage = test_covered.mean().item()
                test_avg_width = test_scores.mean().item()
        
        # Update history
        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(test_losses['total'].item())
        history['val_coverage'].append(test_coverage)
        history['val_avg_width'].append(test_avg_width)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Logging
        if epoch % 10 == 0:
            logger.info(
                f"Epoch {epoch}/{config['TRAINING']['NUM_EPOCHS']} - "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {test_losses['total'].item():.4f}, "
                f"Coverage: {test_coverage:.3f}, "
                f"Avg Width: {test_avg_width:.2f}"
            )
        
        # Learning rate scheduling
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_losses['total'].item())
            else:
                scheduler.step()
        
        # Model checkpointing
        val_loss = test_losses['total'].item()
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'coverage': test_coverage,
                'avg_width': test_avg_width
            }, output_dir / 'best_model.pt')
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
        
        # Save checkpoint periodically
        if epoch % config['OUTPUT'].get('SAVE_FREQUENCY', 10) == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, output_dir / f'checkpoint_epoch_{epoch}.pt')
    
    # Load best model for final evaluation
    checkpoint = torch.load(output_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_scores = model(test_features)
        
        # Call loss with appropriate arguments  
        if use_adaptive_loss:
            test_losses = criterion(test_scores, test_gt, test_pred, test_features)
        else:
            test_losses = criterion(test_scores, test_gt, test_pred, tau)
        
        # Calculate final metrics
        if use_adaptive_loss:
            # For adaptive loss, scores are nonconformity scores
            final_avg_width = test_scores.mean().item()
            final_coverage = test_losses['actual_coverage'].item()
        else:
            # Legacy behavior - scores are widths
            interval_half_widths = test_scores * tau
            lower_bounds = test_pred - interval_half_widths.expand(-1, 4)
            upper_bounds = test_pred + interval_half_widths.expand(-1, 4)
            
            test_covered = ((test_gt >= lower_bounds) & (test_gt <= upper_bounds)).all(dim=1).float()
            final_coverage = test_covered.mean().item()
            final_avg_width = test_scores.mean().item()
        
        final_efficiency = 1.0 / (final_avg_width + 1e-6)
    
    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    if config['OUTPUT'].get('PLOT_RESULTS', True):
        plot_training_curves(history, output_dir)
    
    # Final results
    final_metrics = {
        'test_coverage': final_coverage,
        'test_avg_width': final_avg_width,
        'test_efficiency': final_efficiency,
        'best_epoch': best_epoch,
        'total_epochs': epoch + 1
    }
    
    logger.info(f"Training completed. Best epoch: {best_epoch}")
    logger.info(f"Final coverage: {final_coverage:.3f}, Final MPIW: {final_avg_width:.2f}")
    
    return final_metrics


def plot_training_curves(history: Dict[str, list], output_dir: Path):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Coverage
    axes[0, 1].plot(history['val_coverage'])
    axes[0, 1].axhline(y=0.9, color='r', linestyle='--', label='Target (0.9)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Coverage')
    axes[0, 1].set_title('Validation Coverage')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Average width
    axes[1, 0].plot(history['val_avg_width'])
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Average Width')
    axes[1, 0].set_title('Mean Prediction Interval Width')
    axes[1, 0].grid(True)
    
    # Learning rate
    axes[1, 1].plot(history['learning_rate'])
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()