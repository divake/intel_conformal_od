#!/usr/bin/env python
"""
Training script for Conditional Quantile Networks (CQN).

This implements Solution 2 from the experiment plan.
Key features:
- Direct quantile regression with pinball loss
- Simpler architecture than decomposed models
- Natural coverage through quantile optimization
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset

# Add paths
sys.path.append(str(Path(__file__).parent.parent))

# Import components
from learnable_scoring_fn.core_symmetric.models.cqn_model import (
    ConditionalQuantileNetwork, QuantileLoss
)
from learnable_scoring_fn.core_symmetric.symmetric_adaptive import load_cached_data, prepare_splits
from learnable_scoring_fn.experiment_tracking.tracker import ExperimentTracker


def train_cqn(
    epochs=100,
    batch_size=256,
    learning_rate=0.001,
    target_coverage=0.9,
    device='cuda'
):
    """
    Train Conditional Quantile Network.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        target_coverage: Target coverage level
        device: Device to use
        
    Returns:
        model: Trained model
        best_metrics: Best validation metrics
    """
    print("\n" + "="*60)
    print("Training Conditional Quantile Network (CQN)")
    print("="*60)
    print(f"Target coverage: {target_coverage:.0%}")
    print(f"Architecture: Simple feedforward with quantile heads")
    print(f"Loss: Pinball loss for direct quantile regression")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    cache_dir = "/ssd_4TB/divake/conformal-od/learnable_scoring_fn/cache_base_model"
    raw_data = load_cached_data(cache_dir)
    
    # Prepare data
    train_features = raw_data['train_features']
    val_features = raw_data['val_features']
    
    # Prepare splits
    val_data_dict = {
        'features': val_features,
        'pred_coords': raw_data['val_data']['pred_coords'],
        'gt_coords': raw_data['val_data']['gt_coords'],
        'confidence': raw_data['val_data']['confidence']
    }
    
    cal_data, test_data = prepare_splits(val_data_dict, calib_fraction=0.5, seed=42)
    
    # Format data
    train_data = {
        'features': train_features,
        'pred_boxes': raw_data['train_data']['pred_coords'],
        'gt_boxes': raw_data['train_data']['gt_coords']
    }
    
    val_data = {
        'features': cal_data['features'],
        'pred_boxes': cal_data['pred_coords'],
        'gt_boxes': cal_data['gt_coords']
    }
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(train_data['features']),
        torch.FloatTensor(train_data['pred_boxes']),
        torch.FloatTensor(train_data['gt_boxes'])
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    val_dataset = TensorDataset(
        torch.FloatTensor(val_data['features']),
        torch.FloatTensor(val_data['pred_boxes']),
        torch.FloatTensor(val_data['gt_boxes'])
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = ConditionalQuantileNetwork(
        input_dim=17,
        hidden_dims=[256, 128, 64],
        dropout_rate=0.1,
        base_quantile=target_coverage
    ).to(device)
    
    print(f"\nModel initialized: {model.get_config()['model_type']}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize loss and optimizer
    criterion = QuantileLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # Initialize tracker
    tracker = ExperimentTracker()
    tracker.log_experiment_start(
        "Conditional Quantile Networks",
        "Direct quantile regression to learn coverage-aware prediction intervals"
    )
    
    # Training variables
    best_coverage_error = float('inf')
    best_metrics = {}
    no_improve_count = 0
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 
               'coverage': [], 'mpiw': [], 'width_std': []}
    
    # Training loop
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_losses = []
        
        for batch_idx, (features, pred_boxes, gt_boxes) in enumerate(train_loader):
            features = features.to(device)
            pred_boxes = pred_boxes.to(device)
            gt_boxes = gt_boxes.to(device)
            
            # Forward pass
            outputs = model(features)
            
            # Compute loss
            alpha = 1.0 - target_coverage
            loss_dict = criterion.compute_interval_loss(
                pred_boxes, gt_boxes,
                outputs['lower_quantiles'],
                outputs['upper_quantiles'],
                alpha=alpha
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss_dict['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss_dict['total'].item())
            
            # Log progress
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}: Loss={loss_dict['total']:.4f}, "
                      f"Coverage={loss_dict['coverage_rate']:.1%}, "
                      f"Avg Width={loss_dict['avg_width']:.1f}")
        
        # Validation phase
        model.eval()
        val_metrics = evaluate_model(model, val_loader, target_coverage, device)
        
        # Update learning rate
        scheduler.step()
        
        # Log epoch results
        avg_train_loss = np.mean(train_losses)
        print(f"\nEpoch {epoch}/{epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val Coverage: {val_metrics['coverage']:.1%}")
        print(f"  Val MPIW: {val_metrics['mpiw']:.1f}")
        print(f"  Val Width STD: {val_metrics['width_std']:.2f}")
        
        # Update history
        history['epoch'].append(epoch)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['coverage'].append(val_metrics['coverage'])
        history['mpiw'].append(val_metrics['mpiw'])
        history['width_std'].append(val_metrics['width_std'])
        
        # Check for improvement
        coverage_error = abs(val_metrics['coverage'] - target_coverage)
        
        # Save if best model
        if coverage_error < best_coverage_error or \
           (coverage_error == best_coverage_error and val_metrics['mpiw'] < best_metrics.get('mpiw', float('inf'))):
            best_coverage_error = coverage_error
            best_metrics = val_metrics.copy()
            best_metrics['epoch'] = epoch
            
            # Save checkpoint
            checkpoint_path = Path("learnable_scoring_fn/experiment_tracking/checkpoints/cqn_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': best_metrics,
                'config': model.get_config()
            }, checkpoint_path)
            
            print(f"  ✓ New best model saved (coverage error: {coverage_error:.3f})")
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Log to tracker every 10 epochs
        if epoch % 10 == 0:
            tracker.log_training_progress(epoch, val_metrics)
            # Save checkpoint
            checkpoint_path = Path(f"learnable_scoring_fn/experiment_tracking/checkpoints/cqn_epoch{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics': val_metrics
            }, checkpoint_path)
        
        # Early stopping check
        if val_metrics['coverage'] >= 0.88 and val_metrics['coverage'] <= 0.92:
            # In target range - check if stable
            if len(history['coverage']) >= 5:
                recent_coverages = history['coverage'][-5:]
                if all(0.88 <= c <= 0.92 for c in recent_coverages):
                    print(f"\nEarly stopping: Coverage stable in target range")
                    print(f"Recent coverages: {recent_coverages}")
                    break
        
        # Hard early stopping
        if no_improve_count >= 20:
            print(f"\nEarly stopping: No improvement for 20 epochs")
            break
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best metrics (epoch {best_metrics['epoch']}):")
    print(f"  Coverage: {best_metrics['coverage']:.1%}")
    print(f"  MPIW: {best_metrics['mpiw']:.1f}")
    print(f"  Width STD: {best_metrics['width_std']:.2f}")
    print("="*60)
    
    return model, best_metrics, history


def evaluate_model(model, data_loader, target_coverage, device):
    """Evaluate model on validation data."""
    model.eval()
    
    all_coverages = []
    all_widths = []
    all_losses = []
    
    criterion = QuantileLoss()
    alpha = 1.0 - target_coverage
    
    with torch.no_grad():
        for features, pred_boxes, gt_boxes in data_loader:
            features = features.to(device)
            pred_boxes = pred_boxes.to(device)
            gt_boxes = gt_boxes.to(device)
            
            # Get predictions
            outputs = model(features)
            
            # Compute loss
            loss_dict = criterion.compute_interval_loss(
                pred_boxes, gt_boxes,
                outputs['lower_quantiles'],
                outputs['upper_quantiles'],
                alpha=alpha
            )
            
            all_losses.append(loss_dict['total'].item())
            
            # Check coverage
            widths = outputs['widths']
            errors = torch.abs(gt_boxes - pred_boxes)
            covered = (errors <= widths).all(dim=1)
            all_coverages.extend(covered.cpu().numpy())
            
            # Track widths
            mpiw = (2 * widths).mean(dim=1)  # Mean prediction interval width
            all_widths.extend(mpiw.cpu().numpy())
    
    # Compute metrics
    coverage = np.mean(all_coverages)
    avg_mpiw = np.mean(all_widths)
    width_std = np.std(all_widths)
    avg_loss = np.mean(all_losses)
    
    # Check for collapse
    collapsed = width_std < 1.0 or (avg_mpiw < 1.0 and coverage < 0.5)
    
    return {
        'loss': avg_loss,
        'coverage': coverage,
        'mpiw': avg_mpiw,
        'width_std': width_std,
        'collapsed': collapsed
    }


def main():
    """Main training function."""
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Train model
    model, best_metrics, history = train_cqn(
        epochs=100,
        batch_size=256,
        learning_rate=0.001,
        target_coverage=0.9,
        device=device
    )
    
    # Final evaluation on test set
    print("\nFinal evaluation on test set...")
    cache_dir = "/ssd_4TB/divake/conformal-od/learnable_scoring_fn/cache_base_model"
    raw_data = load_cached_data(cache_dir)
    
    val_data_dict = {
        'features': raw_data['val_features'],
        'pred_coords': raw_data['val_data']['pred_coords'],
        'gt_coords': raw_data['val_data']['gt_coords'],
        'confidence': raw_data['val_data']['confidence']
    }
    
    _, test_data = prepare_splits(val_data_dict, calib_fraction=0.5, seed=42)
    
    test_dataset = TensorDataset(
        torch.FloatTensor(test_data['features']),
        torch.FloatTensor(test_data['pred_coords']),
        torch.FloatTensor(test_data['gt_coords'])
    )
    test_loader = DataLoader(test_dataset, batch_size=256)
    
    test_metrics = evaluate_model(model, test_loader, 0.9, device)
    
    print("\nTest Set Results:")
    print(f"  Coverage: {test_metrics['coverage']:.1%}")
    print(f"  MPIW: {test_metrics['mpiw']:.1f}")
    print(f"  Width STD: {test_metrics['width_std']:.2f}")
    
    # Check success criteria
    if 0.88 <= test_metrics['coverage'] <= 0.92 and test_metrics['width_std'] > 5.0:
        print("\n🎉 SUCCESS! Target metrics achieved! 🎉")
        
        # Save final model
        final_path = Path("learnable_scoring_fn/experiment_tracking/checkpoints/cqn_final_success.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'test_metrics': test_metrics,
            'best_val_metrics': best_metrics,
            'config': model.get_config(),
            'history': history
        }, final_path)
        
        # Update experiment log
        tracker = ExperimentTracker()
        tracker.log_results(test_metrics, "SUCCESS", "CQN achieved target coverage with good width diversity")
    else:
        print("\nTarget not achieved. Moving to next solution if needed.")


if __name__ == "__main__":
    main()