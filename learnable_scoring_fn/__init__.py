"""
Learnable Scoring Function Package

This package implements learnable scoring functions for conformal prediction 
in object detection, with support for both legacy and symmetric adaptive approaches.

Main Components:
- core_symmetric/: New symmetric adaptive implementation
- model.py: Legacy regression neural network
- feature_utils.py: Feature extraction and normalization
- data_utils.py: Data loading and preprocessing utilities  

Usage:
1. Train symmetric adaptive model (recommended):
   ```bash
   cd /ssd_4TB/divake/conformal-od/learnable_scoring_fn
   python train_symmetric.py --config configs/symmetric_default.yaml
   ```

2. Use trained symmetric model:
   ```python
   from learnable_scoring_fn.core_symmetric import SymmetricAdaptiveMLP
   model = SymmetricAdaptiveMLP.from_config(checkpoint['model_config'])
   lower, upper = model.predict_intervals(features, predictions, tau)
   ```

3. Legacy model (for comparison):
   ```python
   from learnable_scoring_fn import RegressionScoringFunction, load_regression_model
   model, checkpoint = load_regression_model('path/to/model.pt')
   widths = model(features)
   intervals = predictions ± (widths * tau)
   ```
"""

__version__ = "3.0.0"
__author__ = "Conformal Object Detection Team"

# Import main components
from .model import (
    RegressionScoringFunction,
    RegressionCoverageLoss,
    calculate_tau_regression,
    UncertaintyFeatureExtractor,
    save_regression_model,
    load_regression_model
)
from .feature_utils import FeatureExtractor, get_feature_names
from .data_utils import (
    prepare_training_data, 
    split_data, 
    COCOClassMapper,
    get_coco_class_frequencies,
    get_top_classes
)

# Import symmetric components (lazy import to avoid circular dependencies)
def get_symmetric_components():
    """Lazy import of symmetric components."""
    from .core_symmetric import (
        SymmetricAdaptiveMLP,
        SymmetricAdaptiveLoss,
        calibrate_tau,
        train_symmetric_adaptive
    )
    return {
        'SymmetricAdaptiveMLP': SymmetricAdaptiveMLP,
        'SymmetricAdaptiveLoss': SymmetricAdaptiveLoss,
        'calibrate_tau': calibrate_tau,
        'train_symmetric_adaptive': train_symmetric_adaptive
    }

# Define what gets imported with "from learnable_scoring_fn import *"
__all__ = [
    # Legacy model components
    'RegressionScoringFunction',
    'RegressionCoverageLoss',
    'calculate_tau_regression',
    'UncertaintyFeatureExtractor',
    'save_regression_model',
    'load_regression_model',
    
    # Feature utilities
    'FeatureExtractor',
    'get_feature_names',
    
    # Data utilities
    'prepare_training_data',
    'split_data',
    'COCOClassMapper',
    'get_coco_class_frequencies',
    'get_top_classes',
    
    # Symmetric components (access via get_symmetric_components())
    'get_symmetric_components',
] 