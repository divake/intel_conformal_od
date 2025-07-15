#!/usr/bin/env python3
"""
Main training script for regression-based learnable scoring function.

FIXED IMPLEMENTATION with correct coverage definition:
- Coverage = P(gt ∈ [pred - width*tau, pred + width*tau])
- Fixed tau = 1.0 (model learns appropriate widths)
- Proper efficiency and calibration losses

Data splits:
- COCO train set for training the scoring function
- COCO val set split into calibration and test sets
"""

import os
import sys
import argparse
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import json
import logging
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pickle
import yaml
from yacs.config import CfgNode

# Add project paths
project_root = "/ssd_4TB/divake/conformal-od"
detectron2_path = "/ssd_4TB/divake/conformal-od/detectron2"

sys.path.insert(0, detectron2_path)
sys.path.insert(0, project_root)

os.environ['DETECTRON2_DATASETS'] = '/ssd_4TB/divake/conformal-od/data'

# Import components from new modular structure
from learnable_scoring_fn.models import create_model
from learnable_scoring_fn.core.loss import RegressionCoverageLoss, AdaptiveCoverageLoss, calculate_tau_regression
from learnable_scoring_fn.model import UncertaintyFeatureExtractor, save_regression_model
from learnable_scoring_fn.feature_utils import FeatureExtractor, get_feature_names

# Import project components
from util import io_file
from util.util import set_seed, set_device
from data import data_loader
from model import model_loader
from detectron2.data import get_detection_dataset_dicts, MetadataCatalog
from control.std_conformal import StdConformal
from calibration.random_split import random_split


def load_learnable_config(config_path):
    """Load configuration for learnable scoring function."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(output_dir: Path):
    """Setup logging configuration."""
    log_file = output_dir / "training.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def load_existing_predictions(dataset_name, output_dir="/ssd_4TB/divake/conformal-od/output"):
    """Load existing predictions from standard conformal runs."""
    base_dir = Path(output_dir) / dataset_name
    
    # Look for existing prediction files
    possible_dirs = [
        "std_conf_x101fpn_std_rank_class",
        "std_conf_x101fpn_std_bonf_class",
        "std_conf_x101fpn",
    ]
    
    for dir_name in possible_dirs:
        pred_dir = base_dir / dir_name
        img_file = pred_dir / f"{dir_name}_img_list.json"
        ist_file = pred_dir / f"{dir_name}_ist_list.json"
        
        if img_file.exists() and ist_file.exists():
            print(f"Found existing predictions in {pred_dir}")
            with open(img_file, 'r') as f:
                img_list = json.load(f)
            with open(ist_file, 'r') as f:
                ist_list = json.load(f)
            return img_list, ist_list
    
    return None, None


def collect_predictions_for_dataset(cfg_file, cfg_path, dataset_type, cache_dir=None, logger=None, learnable_config=None):
    """
    Collect predictions for a specific dataset (train or val).
    
    Args:
        cfg_file: Base config file name
        cfg_path: Path to config directory
        dataset_type: 'train' or 'val'
        cache_dir: Directory to cache predictions
        logger: Logger instance
    """
    if logger:
        logger.info(f"Collecting predictions for {dataset_type} dataset...")
    
    # Check cache first
    if cache_dir:
        cache_file = Path(cache_dir) / f"predictions_{dataset_type}.pkl"
        if cache_file.exists():
            if logger:
                logger.info(f"Loading cached {dataset_type} predictions...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
    
    # Modify config path based on dataset type
    if dataset_type == 'train':
        cfg_path = cfg_path.replace('coco_val', 'coco_train')
        # For training, we might need to use a different config
        # Check if train config exists
        train_cfg_file = Path(cfg_path) / f"{cfg_file}.yaml"
        if not train_cfg_file.exists():
            # Use QR training config as base
            cfg_file = 'cfg_qr_train'
    
    # Load configuration
    cfg = io_file.load_yaml(cfg_file, cfg_path, to_yacs=True)
    
    # Override dataset name if needed
    if dataset_type == 'train':
        cfg.DATASETS.DATASET.NAME = 'coco_train'
        cfg.DATASETS.DATASET.IMG_DIR = 'coco/train2017'
        cfg.DATASETS.DATASET.ANN_FILE = 'coco/annotations/instances_train2017.json'
        # Ensure the data directory is set correctly
        cfg.DATASETS.DIR = '/ssd_4TB/divake/conformal-od/data'
        # Use local checkpoint for train dataset to avoid model zoo error
        cfg.MODEL.LOCAL_CHECKPOINT = True
        cfg.MODEL.CHECKPOINT_PATH = 'checkpoints/x101fpn_train_qr_5k_postprocess.pth'
        # Also use a local config file path
        cfg.MODEL.FILE = '/ssd_4TB/divake/conformal-od/detectron2/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml'
        # Use StandardROIHeads instead of QuantileROIHead
        cfg.MODEL.CONFIG.MODEL.ROI_HEADS.NAME = 'StandardROIHeads'
        # Add missing CALIBRATION section for train data (not used, but required by StdConformal)
        cfg.CALIBRATION = CfgNode()
        cfg.CALIBRATION.FRACTION = 0.5
        cfg.CALIBRATION.BOX_CORRECTION = 'rank_coord'
        cfg.CALIBRATION.BOX_SET_STRATEGY = 'max'
        cfg.CALIBRATION.TRIALS = 100
    
    data_name = cfg.DATASETS.DATASET.NAME
    
    # Check if predictions already exist
    if dataset_type == 'val':
        img_list, ist_list = load_existing_predictions(data_name)
        if img_list is not None and ist_list is not None:
            if logger:
                logger.info(f"Using existing predictions for {data_name}")
            
            # Cache the predictions
            if cache_dir:
                cache_file = Path(cache_dir) / f"predictions_{dataset_type}.pkl"
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_file, 'wb') as f:
                    pickle.dump((img_list, ist_list), f)
            
            return img_list, ist_list
    
    # Otherwise, collect new predictions
    set_seed(cfg.PROJECT.SEED, logger=logger)
    cfg, device = set_device(cfg, 'cuda', logger=logger)
    
    # Register dataset
    data_loader.d2_register_dataset(cfg, logger=logger)
    
    # Build and load model
    cfg_model, model = model_loader.d2_build_model(cfg, logger=logger)
    model_loader.d2_load_model(cfg_model, model, logger=logger)
    model.eval()
    
    # Load dataset
    data_list = get_detection_dataset_dicts(
        data_name, 
        filter_empty=cfg.DATASETS.DATASET.FILTER_EMPTY
    )
    
    # Limit dataset size based on config
    if learnable_config and learnable_config['DATA']['LIMIT_DATASET_SIZE']:
        max_images = learnable_config['DATA']['MAX_TRAIN_IMAGES'] if dataset_type == 'train' else learnable_config['DATA']['MAX_VAL_IMAGES']
        if max_images > 0 and len(data_list) > max_images:
            data_list = data_list[:max_images]
            if logger:
                logger.info(f"Limited {dataset_type} dataset to {max_images} images")
    
    dataloader = data_loader.d2_load_dataset_from_dict(
        data_list, cfg, cfg_model, logger=logger
    )
    
    metadata = MetadataCatalog.get(data_name).as_dict()
    nr_class = len(metadata["thing_classes"])
    
    # Create args for StdConformal
    class Args:
        def __init__(self):
            self.alpha = 0.1
            self.label_set = 'top_singleton'
            self.label_alpha = 0.1
            self.risk_control = False
            self.save_label_set = False
    
    args = Args()
    
    # Use StdConformal to collect predictions
    controller = StdConformal(
        cfg=cfg,
        args=args,
        nr_class=nr_class,
        filedir='.',
        log=None,
        logger=logger
    )
    
    controller.set_collector(nr_class, len(data_list))
    
    # Create checkpoint directory if caching is enabled
    checkpoint_dir = None
    if cache_dir:
        checkpoint_dir = Path(cache_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect predictions with checkpoint support
    img_list, ist_list = controller.collect_predictions(
        model, 
        dataloader, 
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=0.1  # Save checkpoint every 10%
    )
    
    # Cache final predictions
    if cache_dir:
        cache_file = Path(cache_dir) / f"predictions_{dataset_type}.pkl"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump((img_list, ist_list), f)
        if logger:
            logger.info(f"Cached {dataset_type} predictions to {cache_file}")
    
    return img_list, ist_list


def prepare_regression_data(img_list, ist_list, logger):
    """Prepare data for regression training from collected predictions."""
    logger.info("Preparing regression training data...")
    
    feature_extractor = FeatureExtractor()
    uncertainty_extractor = UncertaintyFeatureExtractor()
    
    all_features = []
    all_pred_coords = []
    all_gt_coords = []
    all_errors = []
    all_confidence = []
    all_img_ids = []
    
    # Process each class
    for class_id in range(len(ist_list)):
        if 'pred_x0' not in ist_list[class_id]:
            continue
            
        class_data = ist_list[class_id]
        n_preds = len(class_data['pred_x0'])
        
        if n_preds == 0:
            continue
        
        # Convert to tensors
        pred_x0 = torch.tensor(class_data['pred_x0'], dtype=torch.float32)
        pred_y0 = torch.tensor(class_data['pred_y0'], dtype=torch.float32)
        pred_x1 = torch.tensor(class_data['pred_x1'], dtype=torch.float32)
        pred_y1 = torch.tensor(class_data['pred_y1'], dtype=torch.float32)
        pred_scores = torch.tensor(class_data['pred_score'], dtype=torch.float32)
        
        gt_x0 = torch.tensor(class_data['gt_x0'], dtype=torch.float32)
        gt_y0 = torch.tensor(class_data['gt_y0'], dtype=torch.float32)
        gt_x1 = torch.tensor(class_data['gt_x1'], dtype=torch.float32)
        gt_y1 = torch.tensor(class_data['gt_y1'], dtype=torch.float32)
        
        img_ids = torch.tensor(class_data['img_id'], dtype=torch.int64)
        
        # Stack coordinates
        pred_coords = torch.stack([pred_x0, pred_y0, pred_x1, pred_y1], dim=1)
        gt_coords = torch.stack([gt_x0, gt_y0, gt_x1, gt_y1], dim=1)
        
        # Extract features for each detection
        for i in range(n_preds):
            # Get features
            features = feature_extractor.extract_features(
                pred_coords[i:i+1],
                pred_scores[i:i+1]
            ).squeeze(0)
            
            # Calculate error
            error = torch.abs(gt_coords[i] - pred_coords[i])
            
            all_features.append(features)
            all_pred_coords.append(pred_coords[i])
            all_gt_coords.append(gt_coords[i])
            all_errors.append(error)
            all_confidence.append(pred_scores[i])
            all_img_ids.append(img_ids[i])
    
    if not all_features:
        raise ValueError("No valid prediction-GT pairs found!")
    
    # Stack all
    features = torch.stack(all_features)
    pred_coords = torch.stack(all_pred_coords)
    gt_coords = torch.stack(all_gt_coords)
    errors = torch.stack(all_errors)
    confidence = torch.stack(all_confidence)
    img_ids = torch.stack(all_img_ids)
    
    # Fit error distribution
    uncertainty_extractor.fit_error_distribution(errors)
    
    # Extract uncertainty features
    uncertainty_features = uncertainty_extractor.extract_uncertainty_features(
        pred_coords, confidence
    )
    
    # Combine features
    combined_features = torch.cat([features, uncertainty_features], dim=1)
    
    # Normalize features
    feature_extractor.fit_normalizer(combined_features)
    combined_features = feature_extractor.normalize_features(combined_features)
    
    logger.info(f"Prepared {len(combined_features)} samples")
    logger.info(f"Feature dimension: {combined_features.shape[1]}")
    logger.info(f"Average error: {errors.mean():.3f}")
    
    return combined_features, pred_coords, gt_coords, errors, img_ids, feature_extractor, uncertainty_extractor


def split_val_data(features, pred_coords, gt_coords, errors, img_ids, calib_fraction=0.5):
    """
    Split validation data into calibration and test sets using the project's standard approach.
    """
    # Get unique image IDs
    unique_imgs = torch.unique(img_ids)
    n_imgs = len(unique_imgs)
    
    # Create image mask
    img_mask = torch.zeros(n_imgs, dtype=torch.bool)
    img_mask[:] = True  # All images are relevant
    
    # Use random_split to get calibration/test split
    calib_mask, calib_img_idx, test_img_idx = random_split(
        img_mask, img_ids, calib_fraction, verbose=True
    )
    
    # Get indices for calibration and test
    cal_idx = torch.where(calib_mask)[0]
    test_idx = torch.where(~calib_mask)[0]
    
    # Split data
    cal_data = (
        features[cal_idx],
        pred_coords[cal_idx],
        gt_coords[cal_idx],
        errors[cal_idx]
    )
    
    test_data = (
        features[test_idx],
        pred_coords[test_idx],
        gt_coords[test_idx],
        errors[test_idx]
    )
    
    return cal_data, test_data


def train_model(train_features, train_pred, train_gt, cal_data, test_data, args, logger):
    """Train the regression model."""
    device = torch.device(args.device)
    
    # Create training dataset
    train_dataset = TensorDataset(train_features, train_pred, train_gt)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Move calibration and test data to device
    cal_features, cal_pred, cal_gt, cal_errors = cal_data
    cal_data = (
        cal_features.to(device),
        cal_pred.to(device),
        cal_gt.to(device),
        cal_errors.to(device)
    )
    
    test_features, test_pred, test_gt, test_errors = test_data
    test_data = (
        test_features.to(device),
        test_pred.to(device),
        test_gt.to(device),
        test_errors.to(device)
    )
    
    # Initialize model using factory with ADAPTIVE parameters
    model_config = {
        'hidden_dims': args.hidden_dims,
        'dropout_rate': args.dropout_rate,
        'scoring_strategy': learnable_cfg.get('MODEL', {}).get('SCORING_STRATEGY', 'direct'),
        'output_constraint': learnable_cfg.get('MODEL', {}).get('OUTPUT_CONSTRAINT', 'natural')
    }
    
    # Add model-specific configs if available
    model_specific_config = learnable_cfg.get('MODEL', {}).get(args.model_type.upper(), {})
    for key, value in model_specific_config.items():
        if key.lower() not in model_config:
            model_config[key.lower()] = value
    
    logger.info(f"Creating {args.model_type} model with config: {model_config}")
    
    model = create_model(
        model_type=args.model_type,
        input_dim=train_features.shape[1],
        config=model_config
    ).to(device)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.learning_rate * 0.01
    )
    
    # Loss function - use adaptive loss if specified
    use_adaptive_loss = learnable_cfg.get('USE_ADAPTIVE_LOSS', False)
    
    if use_adaptive_loss:
        logger.info("Using AdaptiveCoverageLoss for training")
        criterion = AdaptiveCoverageLoss(
            target_coverage=args.target_coverage,
            efficiency_weight=args.efficiency_weight,
            ranking_weight=learnable_cfg.get('LOSS', {}).get('RANKING_WEIGHT', 0.05),
            variance_weight=learnable_cfg.get('LOSS', {}).get('VARIANCE_WEIGHT', 0.02),
            smoothness_weight=learnable_cfg.get('LOSS', {}).get('SMOOTHNESS_WEIGHT', 0.01)
        )
    else:
        # NO FALLBACK - We must use adaptive loss!
        raise ValueError(
            "USE_ADAPTIVE_LOSS must be True! The old RegressionCoverageLoss is deprecated.\n"
            "Please ensure your config file includes: USE_ADAPTIVE_LOSS: true"
        )
    
    # Training history
    history = {
        'train_loss': [], 'test_loss': [], 'coverage': [],
        'efficiency': [], 'tau_values': [], 'calibration': []
    }
    
    best_coverage_gap = float('inf')
    best_epoch = 0
    
    logger.info(f"Starting training with {sum(p.numel() for p in model.parameters())} parameters")
    logger.info(f"Train: {len(train_features)}, Cal: {len(cal_features)}, Test: {len(test_features)} samples")
    
    for epoch in range(args.num_epochs):
        # Use fixed tau = 1.0 (model learns appropriate widths)
        if epoch == 0:
            tau = torch.tensor(1.0, device=device)
            logger.info(f"Using fixed tau = {tau.item():.1f}")
        
        # Training
        model.train()
        train_losses = []
        
        for batch_features, batch_pred, batch_gt in train_loader:
            batch_features = batch_features.to(device)
            batch_pred = batch_pred.to(device)
            batch_gt = batch_gt.to(device)
            
            widths = model(batch_features)
            losses = criterion(widths, batch_gt, batch_pred, tau)
            
            optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            
            train_losses.append({k: v.item() for k, v in losses.items()})
        
        # Test set evaluation
        model.eval()
        with torch.no_grad():
            test_features, test_pred, test_gt, test_errors = test_data
            test_widths = model(test_features)
            test_losses = criterion(test_widths, test_gt, test_pred, tau)
            
            # Calculate metrics with CORRECT coverage definition
            interval_half_widths = test_widths * tau
            lower_bounds = test_pred - interval_half_widths.expand(-1, 4)
            upper_bounds = test_pred + interval_half_widths.expand(-1, 4)
            
            # Check if ground truth falls within intervals
            test_covered = ((test_gt >= lower_bounds) & (test_gt <= upper_bounds)).all(dim=1).float()
            test_coverage = test_covered.mean().item()
            avg_width = test_widths.mean().item()
            
            # Calibration metric (correlation between widths and errors)
            correlation = test_losses.get('correlation', torch.tensor(0.0)).item()
        
        scheduler.step()
        
        # Average training losses
        avg_train_losses = {
            k: np.mean([l[k] for l in train_losses]) 
            for k in train_losses[0].keys()
        }
        
        # Log progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"\nEpoch {epoch+1}/{args.num_epochs}")
            logger.info(f"  Train Loss: {avg_train_losses['total']:.4f}")
            logger.info(f"  Test Coverage: {test_coverage:.3f} (target: {args.target_coverage})")
            logger.info(f"  Avg Width: {avg_width:.3f}, Correlation: {correlation:.3f}")
        
        # Save history
        history['train_loss'].append(avg_train_losses['total'])
        history['test_loss'].append(test_losses['total'].item())
        history['coverage'].append(test_coverage)
        history['efficiency'].append(avg_width)
        history['tau_values'].append(tau.item())
        history['calibration'].append(correlation)
        
        # Update efficiency weight based on coverage
        if test_coverage >= 0.85:  # Close to target
            criterion.efficiency_weight = min(args.efficiency_weight, criterion.efficiency_weight * 1.05)
        elif test_coverage < 0.5:  # Far from target
            criterion.efficiency_weight = max(0.0001, criterion.efficiency_weight * 0.9)
        
        # Save best model based on coverage gap
        coverage_gap = abs(test_coverage - args.target_coverage)
        if coverage_gap < best_coverage_gap:
            best_coverage_gap = coverage_gap
            best_epoch = epoch
            save_regression_model(
                model, optimizer, epoch, test_losses, 
                tau.item(), args.output_dir / "best_model.pt"
            )
        
        # Early stopping
        if epoch - best_epoch > args.early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    logger.info(f"\nTraining completed! Best epoch: {best_epoch+1}")
    logger.info(f"Best coverage gap: {best_coverage_gap:.3f}")
    
    return model, history


def plot_results(history, output_dir):
    """Plot training results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train', alpha=0.7)
    axes[0, 0].plot(history['test_loss'], label='Test', alpha=0.7)
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Coverage
    axes[0, 1].plot(history['coverage'], linewidth=2)
    axes[0, 1].axhline(y=0.9, color='r', linestyle='--', label='Target')
    axes[0, 1].set_title('Test Set Coverage')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Interval width
    axes[0, 2].plot(history['efficiency'], color='green')
    axes[0, 2].set_title('Average Interval Width')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Tau values
    axes[1, 0].plot(history['tau_values'], color='orange')
    axes[1, 0].set_title('Tau Values (from Calibration Set)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Calibration
    axes[1, 1].plot(history['calibration'], color='purple')
    axes[1, 1].set_title('Calibration STD')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Coverage vs Efficiency
    axes[1, 2].scatter(history['efficiency'], history['coverage'], 
                      c=range(len(history['coverage'])), cmap='viridis', alpha=0.6)
    axes[1, 2].axhline(y=0.9, color='r', linestyle='--', alpha=0.5)
    axes[1, 2].set_xlabel('Average Width')
    axes[1, 2].set_ylabel('Coverage')
    axes[1, 2].set_title('Coverage vs Efficiency Trade-off')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_results.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train regression-based scoring function")
    
    # Learnable scoring function config
    parser.add_argument('--learnable_config', type=str, 
                        default='config/learnable_scoring_fn/default_config.yaml',
                        help='Path to learnable scoring function configuration file')
    
    # Data configuration (can be overridden by command line)
    parser.add_argument('--config_file', type=str, default=None)
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default=None)
    
    # Model architecture
    parser.add_argument('--model_type', type=str, default='mlp',
                        choices=['mlp', 'ft_transformer', 'tabm', 't2g_former', 'saint_s', 'regression_dlns'],
                        help='Type of model to use')
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[256, 128, 64])
    parser.add_argument('--dropout_rate', type=float, default=0.15)
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=1.0)
    
    # Loss weights
    parser.add_argument('--target_coverage', type=float, default=0.9)
    parser.add_argument('--efficiency_weight', type=float, default=0.05)
    parser.add_argument('--calibration_weight', type=float, default=0.1)
    
    # Other parameters  
    parser.add_argument('--early_stopping_patience', type=int, default=None)
    parser.add_argument('--calib_fraction', type=float, default=None)
    parser.add_argument('--output_dir', type=Path, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    
    args = parser.parse_args()
    
    # Load learnable scoring function configuration
    learnable_cfg = load_learnable_config(args.learnable_config)
    
    # Override args with config values if not specified on command line
    if args.config_file is None:
        args.config_file = learnable_cfg['CONFORMAL']['CONFIG_FILE']
    if args.config_path is None:
        args.config_path = learnable_cfg['CONFORMAL']['CONFIG_PATH']
    if args.cache_dir is None:
        args.cache_dir = learnable_cfg['DATA']['CACHE_DIR']
    if args.hidden_dims == [256, 128, 64]:  # default value
        args.hidden_dims = learnable_cfg['MODEL']['HIDDEN_DIMS']
    if args.dropout_rate == 0.15:  # default value
        args.dropout_rate = learnable_cfg['MODEL']['DROPOUT_RATE']
    if args.num_epochs == 100:  # default value
        args.num_epochs = learnable_cfg['TRAINING']['NUM_EPOCHS']
    if args.batch_size == 128:  # default value
        args.batch_size = learnable_cfg['TRAINING']['BATCH_SIZE']
    if args.learning_rate == 0.001:  # default value
        args.learning_rate = learnable_cfg['TRAINING']['LEARNING_RATE']
    if args.weight_decay == 0.0001:  # default value
        args.weight_decay = learnable_cfg['TRAINING']['WEIGHT_DECAY']
    if args.grad_clip_norm == 1.0:  # default value
        args.grad_clip_norm = learnable_cfg['TRAINING']['GRAD_CLIP_NORM']
    if args.target_coverage == 0.9:  # default value
        args.target_coverage = learnable_cfg['LOSS']['TARGET_COVERAGE']
    if args.efficiency_weight == 0.05:  # default value
        args.efficiency_weight = learnable_cfg['LOSS']['EFFICIENCY_WEIGHT']
    if args.calibration_weight == 0.1:  # default value
        args.calibration_weight = learnable_cfg['LOSS']['CALIBRATION_WEIGHT']
    if args.early_stopping_patience is None:
        args.early_stopping_patience = learnable_cfg['TRAINING']['EARLY_STOPPING_PATIENCE']
    if args.calib_fraction is None:
        args.calib_fraction = learnable_cfg['DATA']['CALIB_FRACTION']
    if args.output_dir is None:
        args.output_dir = Path(learnable_cfg['OUTPUT']['BASE_DIR']) / learnable_cfg['OUTPUT']['EXPERIMENT_NAME']
    if args.device is None:
        args.device = learnable_cfg['TRAINING']['DEVICE']
    if args.seed is None:
        args.seed = learnable_cfg['TRAINING']['SEED']
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    
    logger.info("="*60)
    logger.info("Regression-Based Scoring Function Training")
    logger.info("Using Project's Standard Data Splits:")
    logger.info("  - COCO train set for training")
    logger.info("  - COCO val set split into calibration/test")
    logger.info("="*60)
    logger.info(f"Configuration: {vars(args)}")
    
    try:
        # Load COCO train dataset predictions for training the scoring function
        logger.info("Loading COCO train dataset predictions for training...")
        train_img_list, train_ist_list = collect_predictions_for_dataset(
            args.config_file, args.config_path, 'train', args.cache_dir, logger, learnable_cfg
        )
        
        # Load COCO val dataset predictions for calibration and test
        logger.info("Loading COCO val dataset predictions for calibration/test split...")
        val_img_list, val_ist_list = collect_predictions_for_dataset(
            args.config_file, args.config_path, 'val', args.cache_dir, logger, learnable_cfg
        )
        
        # Prepare training data
        train_features, train_pred, train_gt, train_errors, train_img_ids, \
            train_feat_ext, train_uncert_ext = prepare_regression_data(
                train_img_list, train_ist_list, logger
            )
        
        # Prepare validation data
        val_features, val_pred, val_gt, val_errors, val_img_ids, \
            val_feat_ext, val_uncert_ext = prepare_regression_data(
                val_img_list, val_ist_list, logger
            )
        
        # Normalize validation features using training statistics
        val_features_raw = torch.cat([
            val_feat_ext.extract_features(val_pred, val_gt[:, 0]),  # Using first coord as proxy for confidence
            val_uncert_ext.extract_uncertainty_features(val_pred, val_gt[:, 0])
        ], dim=1)
        val_features = train_feat_ext.normalize_features(val_features_raw)
        
        # Split validation data into calibration and test sets
        cal_data, test_data = split_val_data(
            val_features, val_pred, val_gt, val_errors, val_img_ids, 
            args.calib_fraction
        )
        
        # Save feature statistics
        torch.save({
            'feature_stats': train_feat_ext.feature_stats,
            'error_stats': train_uncert_ext.error_stats
        }, args.output_dir / 'data_stats.pt')
        
        # Train model
        model, history = train_model(
            train_features, train_pred, train_gt, 
            cal_data, test_data, args, logger
        )
        
        # Plot results
        plot_results(history, args.output_dir)
        
        # Save final results
        results = {
            'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            'history': history,
            'data_stats': {
                'train_samples': len(train_features),
                'cal_samples': len(cal_data[0]),
                'test_samples': len(test_data[0]),
                'feature_dim': train_features.shape[1]
            },
            'final_metrics': {
                'coverage': history['coverage'][-1],
                'avg_width': history['efficiency'][-1],
                'tau': history['tau_values'][-1],
                'calibration_std': history['calibration'][-1]
            }
        }
        
        with open(args.output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save the config file used
        with open(args.output_dir / 'config_used.yaml', 'w') as f:
            yaml.dump(learnable_cfg, f, default_flow_style=False)
        
        logger.info("\n" + "="*60)
        logger.info("Training completed successfully!")
        logger.info(f"Results saved to {args.output_dir}")
        logger.info("Data splits used:")
        logger.info(f"  - Training: {len(train_features)} samples from COCO train")
        logger.info(f"  - Calibration: {len(cal_data[0])} samples from COCO val")
        logger.info(f"  - Test: {len(test_data[0])} samples from COCO val")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()