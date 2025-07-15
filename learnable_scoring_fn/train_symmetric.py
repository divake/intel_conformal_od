#!/usr/bin/env python
"""
Train symmetric adaptive conformal prediction model.

Usage:
    python train_symmetric.py [--config CONFIG_FILE]
"""

import argparse
import yaml
import json
from pathlib import Path
from datetime import datetime

from core_symmetric import train_symmetric_adaptive


def load_config(config_path: str = None) -> dict:
    """Load configuration from file or use defaults."""
    
    # Default configuration
    default_config = {
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
        "min_lr": 1e-6,
        "early_stopping_patience": 15
    }
    
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                user_config = yaml.safe_load(f)
            else:
                user_config = json.load(f)
        
        # Merge with defaults
        default_config.update(user_config)
    
    return default_config


def main():
    parser = argparse.ArgumentParser(
        description="Train symmetric adaptive conformal prediction model"
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default=None,
        help='Path to configuration file (YAML or JSON)'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default="/ssd_4TB/divake/conformal-od/learnable_scoring_fn/cache_base_model",
        help='Directory with cached features and predictions'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory to save trained models'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default=None,
        help='Directory for logs and visualizations'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default=None,
        help='Name for this experiment'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up directories
    base_output_dir = Path("/ssd_4TB/divake/conformal-od/learnable_scoring_fn/saved_models/symmetric")
    base_log_dir = Path("/ssd_4TB/divake/conformal-od/learnable_scoring_fn/logs/symmetric")
    
    # Create experiment-specific directories
    if args.experiment_name:
        exp_name = args.experiment_name
    else:
        exp_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    output_dir = args.output_dir or str(base_output_dir / exp_name)
    log_dir = args.log_dir or str(base_log_dir / exp_name)
    
    # Save configuration
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    config_save_path = Path(output_dir) / "config.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Experiment: {exp_name}")
    print(f"Output directory: {output_dir}")
    print(f"Log directory: {log_dir}")
    print(f"Configuration saved to: {config_save_path}")
    print(f"\nConfiguration:")
    print(yaml.dump(config, default_flow_style=False))
    
    # Run training
    model, tau, history = train_symmetric_adaptive(
        config=config,
        cache_dir=args.cache_dir,
        output_dir=output_dir,
        log_dir=log_dir
    )
    
    # Save final results summary
    final_results = {
        "experiment_name": exp_name,
        "final_tau": float(tau),
        "final_coverage": float(history['coverage_rate'][-1]) if 'coverage_rate' in history else None,
        "final_mpiw": float(history['avg_mpiw'][-1]) if 'avg_mpiw' in history else None,
        "total_epochs": len(history.get('coverage_rate', [])),
        "config": config
    }
    
    results_path = Path(output_dir) / "final_results.json"
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nFinal results saved to: {results_path}")


if __name__ == "__main__":
    main()