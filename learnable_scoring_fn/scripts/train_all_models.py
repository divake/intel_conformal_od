#!/usr/bin/env python3
"""
Unified training script for all scoring function models.
Allows selective training and comparison of models.
"""

import os
import sys
import yaml
import torch
import argparse
import json
import time
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.factory import create_model, list_available_models
from core.loss import RegressionCoverageLoss
from core.data_utils import (
    load_cached_predictions,
    extract_features_from_predictions,
    prepare_data_splits
)


# Configuration
MODELS_TO_TRAIN = {
    "mlp": True,
    "ft_transformer": True,
    "tabm": True,
    "t2g_former": True,
    "saint_s": True,
    "regression_dlns": True
}


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging configuration."""
    log_file = output_dir / "training.log"
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger('train_all_models')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def load_config(model_type: str, config_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration for a specific model."""
    if config_dir is None:
        config_dir = project_root / "configs"
    
    # Ensure config_dir is a Path object
    if isinstance(config_dir, str):
        config_dir = Path(config_dir)
    
    base_config_path = config_dir / "base_config.yaml"
    model_config_path = config_dir / f"{model_type}_config.yaml"
    
    # Load base config
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load and merge model-specific config
    if model_config_path.exists():
        with open(model_config_path, 'r') as f:
            model_config = yaml.safe_load(f)
            
            # Handle __BASE__ directive if present
            if '__BASE__' in model_config:
                base_file = model_config.pop('__BASE__')
                base_path = config_dir / base_file
                if base_path.exists():
                    with open(base_path, 'r') as bf:
                        base_config = yaml.safe_load(bf)
                        # First merge base config
                        deep_merge(config, base_config)
            
            # Then merge model-specific config
            deep_merge(config, model_config)
    else:
        # If no model-specific config, set model type
        if 'MODEL' not in config:
            config['MODEL'] = {}
        config['MODEL']['TYPE'] = model_type
    
    return config


def deep_merge(base_dict: dict, update_dict: dict):
    """Deep merge update_dict into base_dict."""
    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            deep_merge(base_dict[key], value)
        else:
            base_dict[key] = value


def prepare_features_if_needed(config: Dict[str, Any], logger: logging.Logger) -> tuple:
    """Load or prepare feature data from cached predictions."""
    cache_dir = Path(config['DATA']['CACHE_DIR'])
    
    # Check if features already exist
    features_train_path = cache_dir / "features_train.pt"
    features_val_path = cache_dir / "features_val.pt"
    
    if features_train_path.exists() and features_val_path.exists():
        logger.info("Loading existing feature data...")
        train_data = torch.load(features_train_path)
        val_data = torch.load(features_val_path)
        return train_data, val_data
    
    logger.info("Feature data not found. Extracting features from predictions...")
    
    # Load cached predictions
    train_predictions, val_predictions = load_cached_predictions(
        cache_dir=str(cache_dir),
        limit_dataset_size=config['DATA'].get('LIMIT_DATASET_SIZE', False)
    )
    
    # Extract features
    logger.info("Extracting features from training predictions...")
    import time
    start_time = time.time()
    train_features, train_gt, train_preds, train_conf, train_img_ids = extract_features_from_predictions(
        train_predictions
    )
    logger.info(f"Training features extracted in {time.time() - start_time:.2f} seconds")
    
    logger.info("Extracting features from validation predictions...")
    start_time = time.time()
    # Use safe parameters to avoid hanging on validation data
    val_features, val_gt, val_preds, val_conf, val_img_ids = extract_features_from_predictions(
        val_predictions,
        num_workers=8,  # Reduced from 36 to avoid hanging
        timeout_per_class=60,  # Increased timeout
        use_sequential_fallback=True  # Enable fallback
    )
    logger.info(f"Validation features extracted in {time.time() - start_time:.2f} seconds")
    
    # Prepare data splits
    data_splits = prepare_data_splits(
        train_features, train_gt, train_preds, train_conf,
        val_features, val_gt, val_preds, val_conf,
        calib_fraction=config['DATA'].get('CALIB_FRACTION', 0.5),
        seed=config['TRAINING'].get('SEED', 42),
        train_img_ids=train_img_ids,
        val_img_ids=val_img_ids
    )
    
    # Save features for future use
    train_data = {
        'features': data_splits['train_features'],
        'gt_coords': data_splits['train_gt_coords'],
        'pred_coords': data_splits['train_pred_coords'],
        'confidence': data_splits['train_confidence']
    }
    
    # Add train image IDs if available
    if 'train_img_ids' in data_splits:
        train_data['img_ids'] = data_splits['train_img_ids']
    
    val_data = {
        'features': torch.cat([data_splits['calib_features'], data_splits['test_features']]),
        'gt_coords': torch.cat([data_splits['calib_gt_coords'], data_splits['test_gt_coords']]),
        'pred_coords': torch.cat([data_splits['calib_pred_coords'], data_splits['test_pred_coords']]),
        'confidence': torch.cat([data_splits['calib_confidence'], data_splits['test_confidence']])
    }
    
    # Add validation image IDs if available
    if 'calib_img_ids' in data_splits and 'test_img_ids' in data_splits:
        val_data['img_ids'] = torch.cat([data_splits['calib_img_ids'], data_splits['test_img_ids']])
    
    # Also save calibration indices for later use
    n_calib = len(data_splits['calib_features'])
    val_data['calib_indices'] = torch.arange(n_calib)
    val_data['test_indices'] = torch.arange(n_calib, len(val_data['features']))
    
    # Save for future use
    torch.save(train_data, features_train_path)
    torch.save(val_data, features_val_path)
    logger.info(f"Saved features to {features_train_path} and {features_val_path}")
    
    return train_data, val_data


def train_single_model(
    model_type: str, 
    config: Dict[str, Any],
    train_data: dict,
    val_data: dict,
    force_retrain: bool = False,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """Train a single model."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_type.upper()}")
    logger.info(f"{'='*60}")
    
    # Setup output directory
    output_dir = Path(config['OUTPUT']['BASE_DIR']) / model_type
    
    # Check if already trained
    if output_dir.exists() and not force_retrain:
        results_file = output_dir / "results.json"
        if results_file.exists():
            logger.info(f"Model {model_type} already trained. Loading results...")
            with open(results_file, 'r') as f:
                return json.load(f)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config used
    with open(output_dir / "config_used.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Create model
    model_config = config.get('MODEL', {}).get(model_type.upper(), {})
    
    # Add general MODEL settings (like SCORING_STRATEGY, OUTPUT_CONSTRAINT)
    general_model_config = config.get('MODEL', {})
    for key in ['SCORING_STRATEGY', 'OUTPUT_CONSTRAINT']:
        if key in general_model_config and key not in model_config:
            model_config[key] = general_model_config[key]
    
    # Convert only the top-level config keys from uppercase to lowercase for compatibility
    if model_config:
        converted_config = {}
        for key, value in model_config.items():
            # Only convert simple keys, not nested dictionaries
            if isinstance(value, dict):
                converted_config[key.lower()] = value
            else:
                converted_config[key.lower()] = value
        model_config = converted_config
    
    model = create_model(
        model_type=model_type,
        input_dim=train_data['features'].shape[1],
        config=model_config
    )
    
    logger.info(f"Model architecture: {model}")
    logger.info(f"Model config: {model.get_config()}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Setup device
    device = torch.device(config['TRAINING']['DEVICE'] if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Move data to device
    train_data_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in train_data.items()}
    val_data_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                      for k, v in val_data.items()}
    
    # Create loss function
    loss_fn = RegressionCoverageLoss(
        target_coverage=config['LOSS']['TARGET_COVERAGE'],
        efficiency_weight=config['LOSS']['EFFICIENCY_WEIGHT'],
        calibration_weight=config['LOSS']['CALIBRATION_WEIGHT']
    )
    
    # Train model
    start_time = time.time()
    
    # Use the new training function
    from core.training import train_model
    
    final_metrics = train_model(
        model=model,
        train_data=train_data_device,
        val_data=val_data_device,
        config=config,
        output_dir=output_dir,
        logger=logger
    )
    
    training_time = time.time() - start_time
    
    # Save results
    results = {
        'model_type': model_type,
        'model_config': model.get_config(),
        'final_metrics': {
            'coverage': float(final_metrics.get('test_coverage', 0)),
            'avg_width': float(final_metrics.get('test_avg_width', 0)),
            'efficiency': float(final_metrics.get('test_efficiency', 0))
        },
        'model_params': total_params,
        'trainable_params': trainable_params,
        'training_time': training_time,
        'training_time_str': f"{training_time/60:.2f} minutes",
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save final model with input_dim included
    model_config = model.get_config()
    model_config['input_dim'] = train_data['features'].shape[1]  # Add input_dim to config
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'model_type': model_type,
        'results': results
    }, output_dir / "final_model.pt")
    
    logger.info(f"Training completed for {model_type}")
    logger.info(f"Final coverage: {results['final_metrics']['coverage']:.3f}")
    logger.info(f"Final MPIW: {results['final_metrics']['avg_width']:.2f}")
    logger.info(f"Training time: {results['training_time_str']}")
    
    return results


def compare_all_results(results_dir: Path, output_dir: Path, logger: logging.Logger):
    """Compare results from all trained models."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect results
    all_results = []
    
    for model_dir in results_dir.iterdir():
        if model_dir.is_dir() and (model_dir / "results.json").exists():
            with open(model_dir / "results.json", 'r') as f:
                results = json.load(f)
                all_results.append({
                    'Model': results['model_type'],
                    'Coverage': results['final_metrics']['coverage'],
                    'MPIW': results['final_metrics']['avg_width'],
                    'Efficiency': results['final_metrics'].get('efficiency', 'N/A'),
                    'Parameters': f"{results['model_params']:,}",
                    'Training Time': results['training_time_str']
                })
    
    if not all_results:
        logger.warning("No results found to compare!")
        return pd.DataFrame()
    
    # Create comparison DataFrame
    df = pd.DataFrame(all_results)
    df = df.sort_values('MPIW')
    
    # Save comparison
    df.to_csv(output_dir / "model_comparison.csv", index=False)
    
    # Create a nice formatted table
    with open(output_dir / "model_comparison.txt", 'w') as f:
        f.write("Model Performance Comparison\n")
        f.write("=" * 80 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        
        # Add best model summary
        best_model = df.iloc[0]
        f.write(f"Best Model (by MPIW): {best_model['Model']}\n")
        f.write(f"  - Coverage: {best_model['Coverage']:.3f}\n")
        f.write(f"  - MPIW: {best_model['MPIW']:.2f}\n")
        f.write(f"  - Parameters: {best_model['Parameters']}\n")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Train scoring function models")
    parser.add_argument('--models', nargs='+', 
                        choices=list_available_models() + ['all'],
                        default=['all'], help='Models to train')
    parser.add_argument('--force-retrain', action='store_true',
                        help='Force retraining even if results exist')
    parser.add_argument('--compare-only', action='store_true',
                        help='Only run comparison, skip training')
    parser.add_argument('--config-dir', type=str, default=None,
                        help='Path to configuration directory')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Base output directory')
    
    args = parser.parse_args()
    
    # Setup base output directory
    if args.output_dir:
        base_output_dir = Path(args.output_dir)
    else:
        base_output_dir = project_root / "experiments" / "results"
    
    # Setup logging
    base_output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(base_output_dir)
    
    logger.info(f"Starting model training pipeline")
    logger.info(f"Available models: {list_available_models()}")
    
    # Determine which models to train
    if 'all' in args.models:
        models_to_train = [m for m, enabled in MODELS_TO_TRAIN.items() if enabled]
    else:
        models_to_train = args.models
    
    logger.info(f"Models to process: {models_to_train}")
    
    # Training phase
    if not args.compare_only:
        # Load base config to get data settings
        base_config = load_config('mlp', args.config_dir)
        
        # Fix cache directory path to be absolute
        if 'DATA' in base_config and 'CACHE_DIR' in base_config['DATA']:
            cache_dir = base_config['DATA']['CACHE_DIR']
            if not os.path.isabs(cache_dir):
                # Make it absolute relative to learnable_scoring_fn directory
                base_config['DATA']['CACHE_DIR'] = os.path.join(
                    '/ssd_4TB/divake/conformal-od/learnable_scoring_fn',
                    cache_dir
                )
        
        # Prepare features once for all models
        train_data, val_data = prepare_features_if_needed(base_config, logger)
        
        # Train each model
        all_results = []
        for model_type in models_to_train:
            if MODELS_TO_TRAIN.get(model_type, False):
                try:
                    # Load model-specific config
                    config = load_config(model_type, args.config_dir)
                    
                    # Override output directory if specified
                    if args.output_dir:
                        config['OUTPUT']['BASE_DIR'] = str(base_output_dir)
                    
                    # Train model
                    results = train_single_model(
                        model_type=model_type,
                        config=config,
                        train_data=train_data,
                        val_data=val_data,
                        force_retrain=args.force_retrain,
                        logger=logger
                    )
                    all_results.append(results)
                    
                except Exception as e:
                    logger.error(f"Error training {model_type}: {str(e)}", exc_info=True)
                    continue
    
    # Comparison phase
    logger.info("\n" + "="*60)
    logger.info("COMPARING ALL RESULTS")
    logger.info("="*60)
    
    results_df = compare_all_results(
        results_dir=base_output_dir,
        output_dir=base_output_dir / "comparison",
        logger=logger
    )
    
    if not results_df.empty:
        logger.info("\n" + "="*60)
        logger.info("FINAL RESULTS SUMMARY")
        logger.info("="*60)
        logger.info("\n" + results_df.to_string(index=False))
        logger.info(f"\nResults saved to: {base_output_dir / 'comparison' / 'model_comparison.csv'}")
    
    logger.info("\nTraining pipeline completed!")


if __name__ == "__main__":
    main()