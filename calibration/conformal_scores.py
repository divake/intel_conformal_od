import torch


def abs_res(gt: torch.Tensor, pred: torch.Tensor):
    # Fixed-width absolute residual scores
    return torch.abs(gt - pred)


def norm_res(gt: torch.Tensor, pred: torch.Tensor, unc: torch.Tensor):
    # Scalable normalized residual scores
    return torch.abs(gt - pred) / unc


def quant_res(gt: torch.Tensor, pred_lower: torch.Tensor, pred_upper: torch.Tensor):
    # Scalable CQR scores, see Eq. 6 in the paper
    return torch.max(pred_lower - gt, gt - pred_upper)


def one_sided_res(gt: torch.Tensor, pred: torch.Tensor, min: bool):
    # Fixed-width one-sided scores from Andeol et al. (2023), see Eq. 6 in the paper
    return (pred - gt) if min else (gt - pred)


def one_sided_mult_res(gt: torch.Tensor, pred: torch.Tensor, mult: torch.Tensor, min: bool):
    # Scalable one-sided scores from Andeol et al. (2023), see Eq. 7 in the paper
    return (pred - gt) / mult if min else (gt - pred) / mult


# ===== LEARNABLE SCORING FUNCTION =====

# Global cache for trained model
_trained_scoring_model = None
_trained_feature_extractor = None
_trained_uncertainty_extractor = None
_model_path = None


def clear_model_cache():
    """Clear the cached model to free memory."""
    global _trained_scoring_model, _trained_feature_extractor, _trained_uncertainty_extractor, _model_path
    _trained_scoring_model = None
    _trained_feature_extractor = None
    _trained_uncertainty_extractor = None
    _model_path = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_trained_scoring_model(model_path: str = None, force_cpu: bool = False):
    """
    Load the trained scoring function model and feature extractor.
    
    Args:
        model_path: Path to trained model checkpoint
        force_cpu: Force loading model on CPU to save GPU memory
        
    Returns:
        model: Loaded ScoringMLP model
        feature_extractor: Loaded FeatureExtractor
        uncertainty_extractor: Loaded UncertaintyFeatureExtractor
    """
    global _trained_scoring_model, _trained_feature_extractor, _trained_uncertainty_extractor, _model_path
    
    # Use default path if not specified
    if model_path is None:
        model_path = "/ssd_4TB/divake/conformal-od/learnable_scoring_fn/experiments/real_data_v1/best_model.pt"
    
    # Load model if not cached or path changed
    if (_trained_scoring_model is None or 
        _trained_feature_extractor is None or 
        _trained_uncertainty_extractor is None or 
        _model_path != model_path):
        
        try:
            # Import here to avoid circular dependencies
            from learnable_scoring_fn.model import load_regression_model, UncertaintyFeatureExtractor
            from learnable_scoring_fn.feature_utils import FeatureExtractor
            
            # First try to load on GPU if not forced to CPU
            if not force_cpu and torch.cuda.is_available():
                try:
                    # Check available GPU memory before loading
                    gpu_mem = torch.cuda.get_device_properties(0).total_memory
                    gpu_used = torch.cuda.memory_allocated(0)
                    gpu_free = gpu_mem - gpu_used
                    
                    # If less than 1GB free, use CPU
                    if gpu_free < 1e9:  # 1GB threshold
                        print(f"Low GPU memory ({gpu_free/1e9:.2f}GB free), loading scoring model on CPU")
                        device = torch.device("cpu")
                    else:
                        device = torch.device("cuda")
                except Exception as e:
                    print(f"Error checking GPU memory: {e}, loading on CPU")
                    device = torch.device("cpu")
            else:
                device = torch.device("cpu")
            
            # Load model checkpoint
            model, checkpoint = load_regression_model(model_path, device)
            
            # Create and load feature extractor
            feature_extractor = FeatureExtractor()
            
            # Create and load uncertainty feature extractor
            uncertainty_extractor = UncertaintyFeatureExtractor()
            
            if 'feature_stats' in checkpoint and checkpoint['feature_stats'] is not None:
                feature_extractor.feature_stats = checkpoint['feature_stats']
            else:
                # Try to load from separate data stats file (contains feature_stats)
                import os
                data_stats_path = os.path.join(os.path.dirname(model_path), 'data_stats.pt')
                feature_stats_path = os.path.join(os.path.dirname(model_path), 'feature_stats.pt')
                
                if os.path.exists(data_stats_path):
                    data_stats = torch.load(data_stats_path, map_location=device)
                    if 'feature_stats' in data_stats:
                        feature_extractor.feature_stats = data_stats['feature_stats']
                    else:
                        raise FileNotFoundError(f"Feature stats not found in data_stats.pt")
                    
                    # Load error stats for uncertainty extractor
                    if 'error_stats' in data_stats:
                        uncertainty_extractor.error_stats = data_stats['error_stats']
                        
                elif os.path.exists(feature_stats_path):
                    feature_extractor.load_stats(feature_stats_path)
                else:
                    raise FileNotFoundError(f"Feature stats not found in checkpoint, data_stats.pt, or feature_stats.pt")
            
            # Cache the loaded model and extractors
            _trained_scoring_model = model
            _trained_feature_extractor = feature_extractor
            _trained_uncertainty_extractor = uncertainty_extractor
            _model_path = model_path
            
            print(f"Loaded trained scoring model from {model_path}")
            
        except Exception as e:
            print(f"Error loading trained scoring model: {e}")
            print("Falling back to absolute residual scoring...")
            return None, None, None
    
    return _trained_scoring_model, _trained_feature_extractor, _trained_uncertainty_extractor


def learned_score(gt: torch.Tensor, pred: torch.Tensor, pred_score: torch.Tensor = None, 
                 model_path: str = None, fallback_to_abs: bool = False, adaptive_mode: bool = False):
    """
    Learnable scoring function - uses trained model to compute nonconformity scores.
    
    Supports two modes:
    1. Legacy mode (adaptive_mode=False): Outputs width predictions that multiply errors
    2. Adaptive mode (adaptive_mode=True): Outputs pure nonconformity scores
    
    Args:
        gt: Ground truth coordinates [N] or [N, 4] or single scalar (used for feature extraction during training)
        pred: Predicted coordinates [N] or [N, 4] or single scalar
        pred_score: Prediction confidence scores [N] or single scalar (required for learned scoring)
        model_path: Path to trained model (optional, uses default if None)
        fallback_to_abs: Deprecated - no longer used
        adaptive_mode: If True, return pure nonconformity scores. If False, use legacy width prediction.
        
    Returns:
        scores: Learned nonconformity scores (same shape as input coordinates)
    """
    # Handle missing prediction scores
    if pred_score is None:
        raise ValueError("pred_score is required for learned scoring function")
    
    # Try loading with GPU memory check first
    try:
        # Load trained model - will automatically check GPU memory
        model, feature_extractor, uncertainty_extractor = load_trained_scoring_model(model_path, force_cpu=False)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("GPU out of memory, retrying with CPU...")
            # Clear GPU cache and retry on CPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model, feature_extractor, uncertainty_extractor = load_trained_scoring_model(model_path, force_cpu=True)
        else:
            raise
    
    if model is None or feature_extractor is None or uncertainty_extractor is None:
        raise RuntimeError("Failed to load trained scoring model")
    
    # Set model to evaluation mode to fix batch normalization issue
    model.eval()
    
    device = next(model.parameters()).device
    
    # Convert inputs to tensors and ensure consistent shapes
    gt_tensor = torch.tensor(gt, dtype=torch.float32) if not isinstance(gt, torch.Tensor) else gt.clone()
    pred_tensor = torch.tensor(pred, dtype=torch.float32) if not isinstance(pred, torch.Tensor) else pred.clone()
    score_tensor = torch.tensor(pred_score, dtype=torch.float32) if not isinstance(pred_score, torch.Tensor) else pred_score.clone()
    
    # Handle scalar inputs (single coordinate case)
    if gt_tensor.dim() == 0:
        gt_tensor = gt_tensor.unsqueeze(0)  # [1]
    if pred_tensor.dim() == 0:
        pred_tensor = pred_tensor.unsqueeze(0)  # [1]
    if score_tensor.dim() == 0:
        score_tensor = score_tensor.unsqueeze(0)  # [1]
    
    # For single coordinate evaluation, we need to create a dummy 4-coordinate vector
    if gt_tensor.dim() == 1 and len(gt_tensor) == 1:
        # Single coordinate case - create [1, 4] with actual coordinate at position 0
        pred_coords = torch.stack([
            pred_tensor[0],     # actual coordinate
            pred_tensor[0],     # use same value for other coords (reasonable default)
            pred_tensor[0] + 50, # x1 = x0 + width (reasonable default)
            pred_tensor[0] + 50  # y1 = y0 + height (reasonable default)
        ]).unsqueeze(0)  # [1, 4]
        pred_scores = score_tensor[:1]  # [1]
        
        # Move to device
        pred_coords = pred_coords.to(device)
        pred_scores = pred_scores.to(device)
        
        # Extract geometric features (13 dimensions)
        geometric_features = feature_extractor.extract_features(pred_coords, pred_scores)
        
        # Extract uncertainty features (4 dimensions)
        uncertainty_features = uncertainty_extractor.extract_uncertainty_features(pred_coords, pred_scores)
        
        # Combine features to get 17 dimensions total
        combined_features = torch.cat([geometric_features, uncertainty_features], dim=1)
        
        # Normalize combined features
        normalized_features = feature_extractor.normalize_features(combined_features)
        
        with torch.no_grad():
            model_output = model(normalized_features).squeeze()  # scalar or [1]
        
        if adaptive_mode:
            # Adaptive mode: return pure nonconformity score
            if model_output.dim() == 0:
                return model_output.item()
            else:
                return model_output[0].item()
        else:
            # Legacy mode: model output represents a width that scales with error
            # In single coordinate case, we don't have actual error, so return the width directly
            if model_output.dim() == 0:
                learned_score_value = model_output.item()
            else:
                learned_score_value = model_output[0].item()
            
            return learned_score_value
    
    # Multi-coordinate case
    elif gt_tensor.dim() == 1 and len(gt_tensor) > 1:
        # Multiple single coordinates - process each one
        results = []
        for i in range(len(gt_tensor)):
            single_result = learned_score(
                gt_tensor[i].item(), 
                pred_tensor[i].item(), 
                score_tensor[i].item(),
                model_path,
                fallback_to_abs,
                adaptive_mode
            )
            results.append(single_result)
        return torch.tensor(results, dtype=torch.float32)
    
    # Full 4-coordinate case [N, 4] - the main case used during evaluation
    elif gt_tensor.dim() == 2 and gt_tensor.shape[1] == 4:
        # Move to device
        gt_tensor = gt_tensor.to(device)
        pred_tensor = pred_tensor.to(device)
        score_tensor = score_tensor.to(device)
        
        # Extract geometric features (13 dimensions)
        geometric_features = feature_extractor.extract_features(pred_tensor, score_tensor)
        
        # Extract uncertainty features (4 dimensions)
        uncertainty_features = uncertainty_extractor.extract_uncertainty_features(pred_tensor, score_tensor)
        
        # Combine features to get 17 dimensions total
        combined_features = torch.cat([geometric_features, uncertainty_features], dim=1)
        
        # Normalize combined features
        normalized_features = feature_extractor.normalize_features(combined_features)
        
        with torch.no_grad():
            model_output = model(normalized_features).squeeze()  # [N]
        
        if model_output.dim() == 0:
            model_output = model_output.unsqueeze(0)  # [1]
        
        if adaptive_mode:
            # Adaptive mode: model outputs are nonconformity scores
            # Compute coordinate-specific scores based on actual errors
            errors = torch.abs(gt_tensor - pred_tensor)  # [N, 4]
            
            # Model output represents difficulty/uncertainty - higher means harder prediction
            # Scale errors by model output to get adaptive nonconformity scores
            scores = errors * model_output.unsqueeze(1).expand(-1, 4)  # [N, 4]
            
            return scores.cpu()
        else:
            # Legacy mode: model outputs are widths, broadcast to all coordinates
            learned_scores_broadcast = model_output.unsqueeze(1).expand(-1, 4)  # [N, 4]
            
            return learned_scores_broadcast.cpu()
    
    else:
        raise ValueError(f"Unexpected tensor shapes: gt={gt_tensor.shape}, pred={pred_tensor.shape}")


def get_learned_score_batch(gt_batch: torch.Tensor, pred_batch: torch.Tensor, 
                           pred_score_batch: torch.Tensor, model_path: str = None):
    """
    Batch version of learned_score for efficiency.
    
    Args:
        gt_batch: Ground truth coordinates [N, 4]
        pred_batch: Predicted coordinates [N, 4]
        pred_score_batch: Prediction confidence scores [N]
        model_path: Path to trained model
        
    Returns:
        scores_batch: Learned nonconformity scores [N]
    """
    # Try loading with GPU memory check first
    try:
        # Load model once for the entire batch - will automatically check GPU memory
        model, feature_extractor, uncertainty_extractor = load_trained_scoring_model(model_path, force_cpu=False)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("GPU out of memory, retrying with CPU...")
            # Clear GPU cache and retry on CPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model, feature_extractor, uncertainty_extractor = load_trained_scoring_model(model_path, force_cpu=True)
        else:
            raise
    
    if model is None or feature_extractor is None or uncertainty_extractor is None:
        raise RuntimeError("Failed to load trained scoring model for batch processing")
    
    device = next(model.parameters()).device
    
    # Move to device
    pred_batch = pred_batch.to(device)
    pred_score_batch = pred_score_batch.to(device)
    gt_batch = gt_batch.to(device)
    
    # Extract geometric features (13 dimensions)
    geometric_features = feature_extractor.extract_features(pred_batch, pred_score_batch)
    
    # Extract uncertainty features (4 dimensions)
    uncertainty_features = uncertainty_extractor.extract_uncertainty_features(pred_batch, pred_score_batch)
    
    # Combine features to get 17 dimensions total
    combined_features = torch.cat([geometric_features, uncertainty_features], dim=1)
    
    # Normalize combined features
    normalized_features = feature_extractor.normalize_features(combined_features)
    
    # Get learned scores
    with torch.no_grad():
        learned_scores = model(normalized_features).squeeze()  # [N]
    
    # Compute scaled nonconformity scores
    abs_errors = torch.abs(gt_batch - pred_batch)  # [N, 4]
    scaled_scores = abs_errors / (learned_scores.unsqueeze(1).expand(-1, 4) + 1e-6)
    
    # Return mean across coordinates
    return scaled_scores.mean(dim=1).cpu()


def learned_score_adaptive(pred: torch.Tensor, pred_score: torch.Tensor = None, 
                          model_path: str = None, scoring_strategy: str = 'direct'):
    """
    Adaptive learned scoring function that outputs pure nonconformity scores.
    This version doesn't require ground truth during inference.
    
    Args:
        pred: Predicted coordinates [N, 4]
        pred_score: Prediction confidence scores [N]
        model_path: Path to trained model
        scoring_strategy: Strategy for score computation ('direct', 'multiplicative', 'coordinate_specific')
        
    Returns:
        scores: Nonconformity scores [N] or [N, 4] depending on strategy
    """
    if pred_score is None:
        raise ValueError("pred_score is required for learned scoring function")
    
    # Load model
    try:
        model, feature_extractor, uncertainty_extractor = load_trained_scoring_model(model_path, force_cpu=False)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("GPU out of memory, retrying with CPU...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model, feature_extractor, uncertainty_extractor = load_trained_scoring_model(model_path, force_cpu=True)
        else:
            raise
    
    if model is None:
        raise RuntimeError("Failed to load trained scoring model")
    
    model.eval()
    device = next(model.parameters()).device
    
    # Ensure tensors
    pred_tensor = torch.tensor(pred, dtype=torch.float32) if not isinstance(pred, torch.Tensor) else pred.clone()
    score_tensor = torch.tensor(pred_score, dtype=torch.float32) if not isinstance(pred_score, torch.Tensor) else pred_score.clone()
    
    # Move to device
    pred_tensor = pred_tensor.to(device)
    score_tensor = score_tensor.to(device)
    
    # Extract features
    geometric_features = feature_extractor.extract_features(pred_tensor, score_tensor)
    uncertainty_features = uncertainty_extractor.extract_uncertainty_features(pred_tensor, score_tensor)
    combined_features = torch.cat([geometric_features, uncertainty_features], dim=1)
    normalized_features = feature_extractor.normalize_features(combined_features)
    
    # Get model predictions
    with torch.no_grad():
        model_output = model(normalized_features).squeeze()  # [N] or [N, C]
    
    if model_output.dim() == 0:
        model_output = model_output.unsqueeze(0)
    
    # Apply scoring strategy
    if scoring_strategy == 'direct':
        # Direct nonconformity scores
        return model_output.cpu()
    
    elif scoring_strategy == 'multiplicative':
        # Use exponential to ensure positive scaling
        scale_factors = torch.exp(model_output)
        return scale_factors.cpu()
    
    elif scoring_strategy == 'coordinate_specific':
        # Model should output 4 values per prediction
        if model_output.shape[-1] != 4:
            # If not, expand to 4 coordinates
            model_output = model_output.unsqueeze(1).expand(-1, 4)
        return model_output.cpu()
    
    else:
        return model_output.cpu()


# Convenience function for integration with existing code
def get_available_scoring_functions():
    """Return list of available scoring function names."""
    return [
        'abs_res',           # Standard absolute residual
        'norm_res',          # Normalized residual (requires uncertainty)
        'quant_res',         # CQR scores (requires prediction intervals)
        'one_sided_res',     # One-sided residual
        'one_sided_mult_res', # One-sided multiplicative residual
        'learned_score'      # Learnable scoring function
    ]


def is_learned_model_available(model_path: str = None):
    """Check if trained model is available and loadable."""
    model, feature_extractor, uncertainty_extractor = load_trained_scoring_model(model_path)
    if model is None or feature_extractor is None or uncertainty_extractor is None:
        raise RuntimeError(f"Trained scoring model not available at path: {model_path}")
    return True
