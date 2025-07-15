"""Factory for creating scoring function models."""

from typing import Dict, Any, List
from .base import BaseScoringFunction
from .mlp import MLPScoringFunction
from .ft_transformer import FTTransformerScoringFunction
from .tabm import TabMScoringFunction
from .t2g_former import T2GFormerScoringFunction
from .saint_s import SAINTSScoringFunction
from .regression_dlns import RegressionDLNScoringFunction

# Model registry with all available models
MODEL_REGISTRY = {
    "mlp": MLPScoringFunction,
    "ft_transformer": FTTransformerScoringFunction,
    "tabm": TabMScoringFunction,
    "t2g_former": T2GFormerScoringFunction,
    "saint_s": SAINTSScoringFunction,
    "regression_dlns": RegressionDLNScoringFunction
}


def create_model(model_type: str, input_dim: int, config: Dict[str, Any] = None) -> BaseScoringFunction:
    """Factory function to create models.
    
    Args:
        model_type: Type of model to create (e.g., 'mlp', 'ft_transformer')
        input_dim: Input dimension for the model
        config: Model-specific configuration dictionary
        
    Returns:
        model: Instantiated scoring function model
        
    Raises:
        ValueError: If model_type is not recognized
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )
    
    if config is None:
        config = {}
    
    model_class = MODEL_REGISTRY[model_type]
    return model_class(input_dim=input_dim, **config)


def list_available_models() -> List[str]:
    """Return list of available model types.
    
    Returns:
        models: List of model type strings
    """
    return list(MODEL_REGISTRY.keys())


def register_model(model_type: str, model_class: type):
    """Register a new model class.
    
    Args:
        model_type: String identifier for the model
        model_class: Model class (must inherit from BaseScoringFunction)
    """
    if not issubclass(model_class, BaseScoringFunction):
        raise TypeError(f"{model_class} must inherit from BaseScoringFunction")
    
    MODEL_REGISTRY[model_type] = model_class