#!/usr/bin/env python3
"""
Run adaptive scoring function training using the CORRECT script.
This ensures all models use proper adaptive parameters.
"""

import subprocess
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Train adaptive scoring functions')
    parser.add_argument('--models', nargs='+', 
                       default=['mlp', 'tabm', 't2g_former', 'regression_dlns', 'saint_s', 'ft_transformer'],
                       help='Models to train')
    parser.add_argument('--config-dir', type=str,
                       default='learnable_scoring_fn/configs',
                       help='Configuration directory')
    parser.add_argument('--force-retrain', action='store_true',
                       help='Force retraining even if results exist')
    args = parser.parse_args()
    
    print("🚀 Starting Adaptive Scoring Function Training")
    print("=" * 70)
    print("Using train_all_models.py for CORRECT adaptive training")
    print(f"Models to train: {', '.join(args.models)}")
    print("=" * 70)
    
    # Use the CORRECT training script that passes all parameters
    cmd = [
        "/home/divake/miniconda3/envs/env_cu121/bin/python",
        "/ssd_4TB/divake/conformal-od/learnable_scoring_fn/scripts/train_all_models.py",
        "--models"] + args.models + [
        "--config-dir", args.config_dir
    ]
    
    if args.force_retrain:
        cmd.append("--force-retrain")
    
    print("📊 This will use the adaptive configuration with:")
    print("   - OUTPUT_CONSTRAINT: natural")
    print("   - SCORING_STRATEGY: direct")
    print("   - USE_ADAPTIVE_LOSS: true")
    print("=" * 70)
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Training completed successfully!")
        print("📂 Results saved to: experiments/results_adaptive/")
        print("\n⚠️  IMPORTANT: Never use the old train.py directly!")
        print("   Always use this script or train_all_models.py")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with error code {e.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    main()