"""Main training script for symmetric adaptive conformal prediction."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import json
import pickle
from datetime import datetime

# Import our modules
from .models.symmetric_mlp import SymmetricAdaptiveMLP
from .losses.symmetric_loss import SymmetricAdaptiveLoss
from .calibration.tau_calibration import TauCalibrator
from .utils.logging import AdaptiveConformalLogger
from .utils.visualization import plot_training_results, plot_tau_evolution

# Import feature extraction from parent
import sys
sys.path.append(str(Path(__file__).parent.parent))
from feature_utils import FeatureExtractor


def load_cached_data(cache_dir: str) -> Dict[str, Any]:
    """Load cached features and predictions."""
    cache_path = Path(cache_dir)
    
    print(f"Loading cached data from {cache_path}")
    
    # Load features (these are already dictionaries)
    train_data = torch.load(cache_path / "features_train.pt")
    val_data = torch.load(cache_path / "features_val.pt")
    
    # Load predictions
    with open(cache_path / "predictions_train.pkl", 'rb') as f:
        train_preds = pickle.load(f)
    
    with open(cache_path / "predictions_val.pkl", 'rb') as f:
        val_preds = pickle.load(f)
    
    # Extract features from the dictionaries
    train_features = train_data['features']
    val_features = val_data['features']
    
    print(f"Loaded train features: {train_features.shape}")
    print(f"Loaded val features: {val_features.shape}")
    
    return {
        'train_features': train_features,
        'train_data': train_data,  # Full data dict
        'val_features': val_features,
        'val_data': val_data,      # Full data dict
        'train_predictions': train_preds,
        'val_predictions': val_preds
    }


def prepare_splits(
    val_data: Dict[str, torch.Tensor],
    calib_fraction: float = 0.5,
    seed: int = 42
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Split validation data into calibration and test sets."""
    torch.manual_seed(seed)
    
    # Extract data from validation dictionary
    val_features = val_data['features']
    pred_coords = val_data['pred_coords']
    gt_coords = val_data['gt_coords']
    confidence = val_data['confidence']
    
    print(f"Total validation samples: {len(val_features)}")
    
    # Split indices
    n_val = len(val_features)
    n_calib = int(n_val * calib_fraction)
    
    indices = torch.randperm(n_val)
    calib_idx = indices[:n_calib]
    test_idx = indices[n_calib:]
    
    # Create splits
    calib_data = {
        'features': val_features[calib_idx],
        'pred_coords': pred_coords[calib_idx],
        'gt_coords': gt_coords[calib_idx],
        'confidence': confidence[calib_idx]
    }
    
    test_data = {
        'features': val_features[test_idx],
        'pred_coords': pred_coords[test_idx],
        'gt_coords': gt_coords[test_idx],
        'confidence': confidence[test_idx]
    }
    
    print(f"Calibration samples: {len(calib_idx)}")
    print(f"Test samples: {len(test_idx)}")
    
    return calib_data, test_data


def compute_size_stratified_metrics(
    model: nn.Module,
    data_loader: DataLoader,
    tau: float,
    device: torch.device
) -> Dict[str, Dict[str, float]]:
    """Compute metrics stratified by object size."""
    model.eval()
    
    # Size bins (matching COCO standards)
    size_bins = {
        'small': (0, 32**2),
        'medium': (32**2, 96**2),
        'large': (96**2, float('inf'))
    }
    
    # Initialize collectors
    results = {cat: {'covered': [], 'mpiw': []} for cat in size_bins}
    
    with torch.no_grad():
        for batch in data_loader:
            features = batch[0].to(device)
            pred_coords = batch[1].to(device)
            gt_coords = batch[2].to(device)
            
            # Get predictions
            widths = model(features)
            scaled_widths = widths * tau
            
            # Check coverage
            lower = pred_coords - scaled_widths
            upper = pred_coords + scaled_widths
            covered = ((gt_coords >= lower) & (gt_coords <= upper)).all(dim=1)
            
            # Compute MPIW
            mpiw = (2 * scaled_widths).mean(dim=1)
            
            # Compute object sizes
            box_widths = gt_coords[:, 2] - gt_coords[:, 0]
            box_heights = gt_coords[:, 3] - gt_coords[:, 1]
            areas = box_widths * box_heights
            
            # Stratify
            for i in range(len(areas)):
                area = areas[i].item()
                for cat, (min_size, max_size) in size_bins.items():
                    if min_size <= area < max_size:
                        results[cat]['covered'].append(covered[i].item())
                        results[cat]['mpiw'].append(mpiw[i].item())
                        break
    
    # Aggregate results
    final_results = {}
    for cat, data in results.items():
        if data['covered']:
            final_results[cat] = {
                'coverage': np.mean(data['covered']),
                'mpiw': np.mean(data['mpiw']),
                'count': len(data['covered'])
            }
        else:
            final_results[cat] = {
                'coverage': 0.0,
                'mpiw': 0.0,
                'count': 0
            }
    
    return final_results


def train_symmetric_adaptive(
    config: Dict,
    cache_dir: str = "/ssd_4TB/divake/conformal-od/learnable_scoring_fn/cache_base_model",
    output_dir: str = "/ssd_4TB/divake/conformal-od/learnable_scoring_fn/saved_models/symmetric",
    log_dir: str = "/ssd_4TB/divake/conformal-od/learnable_scoring_fn/logs/symmetric"
):
    """
    Main training function for symmetric adaptive conformal prediction.
    
    Args:
        config: Training configuration
        cache_dir: Directory with cached features/predictions
        output_dir: Directory to save models
        log_dir: Directory for logs
    """
    # Generate experiment name with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"symmetric_adaptive_{timestamp}"
    
    # Create organized directory structure for this run
    experiment_dir = Path(output_dir) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    model_dir = experiment_dir / "models"
    model_dir.mkdir(exist_ok=True)
    plot_dir = experiment_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    
    # Initialize logger with experiment-specific directory
    logger = AdaptiveConformalLogger(log_dir, experiment_name)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load cached data
    data = load_cached_data(cache_dir)
    
    # Use the pre-processed data from cache
    train_features = data['train_features']
    train_pred_coords = data['train_data']['pred_coords']
    train_gt_coords = data['train_data']['gt_coords']
    
    # Prepare calibration and test splits
    calib_data, test_data = prepare_splits(
        data['val_data'],
        calib_fraction=0.5
    )
    
    # Create data loaders
    train_dataset = TensorDataset(
        train_features, train_pred_coords, train_gt_coords
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    calib_dataset = TensorDataset(
        calib_data['features'],
        calib_data['pred_coords'],
        calib_data['gt_coords']
    )
    calib_loader = DataLoader(
        calib_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )
    
    test_dataset = TensorDataset(
        test_data['features'],
        test_data['pred_coords'],
        test_data['gt_coords']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )
    
    # Initialize model
    model = SymmetricAdaptiveMLP(
        input_dim=17,
        hidden_dims=config.get('hidden_dims', [128, 128]),
        dropout_rate=config.get('dropout_rate', 0.1),
        activation=config.get('activation', 'relu'),
        use_batch_norm=config.get('use_batch_norm', True)
    ).to(device)
    
    print(f"Model: {model.model_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Initialize loss function
    if config.get('use_size_aware_loss', False):
        from .losses.size_aware_loss import SizeAwareSymmetricLoss
        criterion = SizeAwareSymmetricLoss(
            small_target_coverage=config.get('small_target_coverage', 0.90),
            medium_target_coverage=config.get('medium_target_coverage', 0.89),
            large_target_coverage=config.get('large_target_coverage', 0.85),
            lambda_efficiency=config['lambda_efficiency'],
            coverage_loss_type=config.get('coverage_loss_type', 'smooth_l1'),
            size_normalization=config.get('size_normalization', True),
            small_threshold=config.get('small_threshold', 32.0),
            large_threshold=config.get('large_threshold', 96.0)
        )
        print("Using SizeAwareSymmetricLoss with targets:")
        print(f"  Small objects (<{config.get('small_threshold', 32.0)}): {config.get('small_target_coverage', 0.90):.0%}")
        print(f"  Medium objects: {config.get('medium_target_coverage', 0.89):.0%}")
        print(f"  Large objects (>{config.get('large_threshold', 96.0)}): {config.get('large_target_coverage', 0.85):.0%}")
    else:
        criterion = SymmetricAdaptiveLoss(
            target_coverage=config['target_coverage'],
            lambda_efficiency=config['lambda_efficiency'],
            coverage_loss_type=config.get('coverage_loss_type', 'smooth_l1'),
            size_normalization=config.get('size_normalization', True)
        )
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    # Initialize scheduler
    if config.get('lr_scheduler') == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['epochs'],
            eta_min=config.get('min_lr', 1e-6)
        )
    else:
        scheduler = None
    
    # Initialize tau calibrator
    tau_calibrator = TauCalibrator(
        target_coverage=config['target_coverage'],
        smoothing_factor=config.get('tau_smoothing', 0.7)
    )
    
    # Training state
    current_tau = 1.0
    best_coverage_error = float('inf')
    best_mpiw = float('inf')
    history = {}
    
    # Training loop
    for epoch in range(1, config['epochs'] + 1):
        logger.log_epoch_start(epoch, current_tau)
        
        # Phase 1: Training
        model.train()
        train_losses = []
        train_metrics = {}
        
        for batch_idx, batch in enumerate(train_loader):
            features = batch[0].to(device)
            pred_coords = batch[1].to(device)
            gt_coords = batch[2].to(device)
            
            # Forward pass
            widths = model(features)
            
            # Compute loss
            loss_dict = criterion(
                pred_coords, gt_coords, widths, current_tau
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss_dict['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config.get('grad_clip_norm', 1.0)
            )
            
            optimizer.step()
            
            # Log batch metrics
            train_losses.append(loss_dict['total'].item())
            
            if batch_idx % 100 == 0:
                logger.log_training_phase(epoch, batch_idx, loss_dict)
        
        # Aggregate training metrics
        train_metrics['total'] = np.mean(train_losses)
        logger.log_epoch_metrics(epoch, 'train', train_metrics)
        
        # Phase 2: Calibration (skip for epoch 1)
        if epoch > 1:
            old_tau = current_tau
            current_tau, calib_stats = tau_calibrator.calibrate(
                model, calib_data, config['batch_size'], device
            )
            logger.log_calibration_phase(epoch, old_tau, current_tau, calib_stats)
        
        # Phase 3: Validation
        model.eval()
        val_losses = []
        val_coverages = []
        val_mpiws = []
        
        with torch.no_grad():
            for batch in test_loader:
                features = batch[0].to(device)
                pred_coords = batch[1].to(device)
                gt_coords = batch[2].to(device)
                
                # Get predictions
                widths = model(features)
                
                # Compute metrics
                loss_dict = criterion(
                    pred_coords, gt_coords, widths, current_tau
                )
                
                val_losses.append(loss_dict['total'].item())
                val_coverages.append(loss_dict['coverage_rate'].item())
                val_mpiws.append(loss_dict['avg_mpiw'].item())
        
        # Aggregate validation metrics
        val_metrics = {
            'total': np.mean(val_losses),
            'coverage_rate': np.mean(val_coverages),
            'avg_mpiw': np.mean(val_mpiws),
            'tau': current_tau
        }
        
        # Compute size-stratified metrics
        size_metrics = compute_size_stratified_metrics(
            model, test_loader, current_tau, device
        )
        
        logger.log_validation_phase(epoch, val_metrics, size_metrics)
        
        # Smart model checkpointing
        coverage_error = abs(val_metrics['coverage_rate'] - config['target_coverage'])
        min_target_coverage = config.get('min_coverage', 0.88)
        max_target_coverage = config.get('max_coverage', 0.905)
        
        # Save best model logic - prioritize coverage in target range with lowest MPIW
        save_model = False
        save_reason = ""
        
        if min_target_coverage <= val_metrics['coverage_rate'] <= max_target_coverage:
            # In target range - prioritize by MPIW
            if val_metrics['avg_mpiw'] < best_mpiw:
                save_model = True
                save_reason = f"Target coverage ({val_metrics['coverage_rate']:.3f}) with better MPIW ({val_metrics['avg_mpiw']:.1f})"
                best_mpiw = val_metrics['avg_mpiw']
                best_coverage_error = coverage_error
            elif abs(val_metrics['avg_mpiw'] - best_mpiw) < 0.1 and coverage_error < best_coverage_error:
                save_model = True
                save_reason = f"Similar MPIW, closer to {config['target_coverage']:.0%} coverage"
                best_coverage_error = coverage_error
        
        if save_model:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'model_config': model.get_config(),
                'tau': current_tau,
                'metrics': val_metrics,
                'size_metrics': size_metrics,
                'config': config
            }
            # Save with descriptive filename
            model_filename = f'best_model_cov{val_metrics["coverage_rate"]:.3f}_mpiw{val_metrics["avg_mpiw"]:.1f}.pt'
            torch.save(checkpoint, model_dir / model_filename)
            # Also save as 'best_model.pt' for easy access
            torch.save(checkpoint, model_dir / 'best_model.pt')
            logger.log_best_model(epoch, save_reason, val_metrics)
        
        # Only save checkpoints when significant progress is made
        # No need to save every 10 epochs - just save when we have a good model
        # This reduces clutter and focuses on meaningful checkpoints
        
        # Update history
        for key, value in val_metrics.items():
            if key not in history:
                history[key] = []
            history[key].append(value)
        
        # Learning rate scheduling
        if scheduler:
            scheduler.step()
            history.setdefault('learning_rate', []).append(
                optimizer.param_groups[0]['lr']
            )
        
        # Visualization
        logger.create_visualization(epoch)
        
        # Early stopping check based on stable coverage
        if epoch > config.get('warmup_epochs', 5):
            # Check if we have enough history
            if len(history.get('coverage_rate', [])) >= 10:
                recent_coverages = history['coverage_rate'][-10:]
                # Check if coverage is stable in target range (88-90.5%)
                if all(0.88 <= c <= 0.905 for c in recent_coverages):
                    recent_mpiws = history['avg_mpiw'][-10:]
                    avg_coverage = np.mean(recent_coverages)
                    std_coverage = np.std(recent_coverages)
                    print(f"\nEarly stopping: Coverage stable at {avg_coverage:.3f} (±{std_coverage:.3f})")
                    print(f"Average MPIW over last 10 epochs: {np.mean(recent_mpiws):.2f}")
                    break
    
    # Final summary
    logger.save_final_summary()
    
    # Save final plots to experiment directory
    plot_training_results(
        history,
        save_path=plot_dir / "final_training_results.png",
        show=False
    )
    
    plot_tau_evolution(
        tau_calibrator.get_tau_evolution(),
        coverage_history=history.get('coverage_rate'),
        save_path=plot_dir / "tau_evolution.png",
        show=False
    )
    
    # Save configuration for reproducibility
    import yaml
    with open(experiment_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Save final results summary
    final_results = {
        'experiment_name': experiment_name,
        'final_tau': current_tau,
        'final_coverage': history['coverage_rate'][-1] if 'coverage_rate' in history else 0,
        'final_mpiw': history['avg_mpiw'][-1] if 'avg_mpiw' in history else 0,
        'best_model_saved': (model_dir / 'best_model.pt').exists(),
        'total_epochs': epoch,
        'size_metrics': size_metrics if 'size_metrics' in locals() else None
    }
    
    with open(experiment_dir / "final_results.json", 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Final tau: {current_tau:.4f}")
    print(f"Final coverage: {val_metrics['coverage_rate']:.3f}")
    print(f"Final MPIW: {val_metrics['avg_mpiw']:.2f}")
    
    return model, current_tau, history


if __name__ == "__main__":
    # Default configuration
    config = {
        "learning_rate": 1e-3,
        "epochs": 100,
        "batch_size": 256,
        "target_coverage": 0.9,
        "lambda_efficiency": 0.1,
        "tau_smoothing": 0.7,
        "lr_scheduler": "cosine",
        "warmup_epochs": 5,
        "hidden_dims": [128, 128],
        "dropout_rate": 0.1,
        "activation": "relu",
        "use_batch_norm": True,
        "coverage_loss_type": "smooth_l1",
        "size_normalization": True,
        "grad_clip_norm": 1.0,
        "weight_decay": 1e-4,
        "min_lr": 1e-6
    }
    
    # Run training
    train_symmetric_adaptive(config)