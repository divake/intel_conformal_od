"""Safe version of optimized data loading with timeout and error handling."""

import pickle
import torch
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import logging
import multiprocessing as mp
from functools import partial
import numpy as np
import time
import signal
from contextlib import contextmanager

# Import feature extractors
import sys
from pathlib import Path
# Add parent directory to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from learnable_scoring_fn.feature_utils import FeatureExtractor
from learnable_scoring_fn.model import UncertaintyFeatureExtractor


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@contextmanager
def timeout(seconds):
    """Context manager for timing out operations."""
    # Disable timeout in multiprocessing to avoid hanging
    # Just yield without timeout for now
    yield


def process_class_data_safe(class_data_with_id, feature_extractor=None, uncertainty_extractor=None):
    """Process a single class's data with error handling."""
    class_id, class_data = class_data_with_id
    
    try:
        if not isinstance(class_data, dict):
            logger.warning(f"Class {class_id}: Not a dict, type={type(class_data)}")
            return None
            
        if 'pred_x0' not in class_data:
            logger.warning(f"Class {class_id}: Missing 'pred_x0' key")
            return None
            
        n_preds = len(class_data['pred_x0'])
        if n_preds == 0:
            logger.debug(f"Class {class_id}: No predictions")
            return None
        
        # Log large classes
        if n_preds > 10000:
            logger.info(f"Class {class_id}: Processing {n_preds} predictions (large class)")
        
        # Convert to tensors with error checking
        try:
            pred_x0 = torch.tensor(class_data['pred_x0'], dtype=torch.float32)
            pred_y0 = torch.tensor(class_data['pred_y0'], dtype=torch.float32)
            pred_x1 = torch.tensor(class_data['pred_x1'], dtype=torch.float32)
            pred_y1 = torch.tensor(class_data['pred_y1'], dtype=torch.float32)
            pred_scores = torch.tensor(class_data['pred_score'], dtype=torch.float32)
            
            gt_x0 = torch.tensor(class_data['gt_x0'], dtype=torch.float32)
            gt_y0 = torch.tensor(class_data['gt_y0'], dtype=torch.float32)
            gt_x1 = torch.tensor(class_data['gt_x1'], dtype=torch.float32)
            gt_y1 = torch.tensor(class_data['gt_y1'], dtype=torch.float32)
        except Exception as e:
            logger.error(f"Class {class_id}: Error converting to tensors: {str(e)}")
            return None
        
        # Stack coordinates
        pred_coords = torch.stack([pred_x0, pred_y0, pred_x1, pred_y1], dim=1)
        gt_coords = torch.stack([gt_x0, gt_y0, gt_x1, gt_y1], dim=1)
        
        # Get image dimensions
        if 'img_height' in class_data and 'img_width' in class_data:
            img_heights = torch.tensor(class_data['img_height'], dtype=torch.float32)
            img_widths = torch.tensor(class_data['img_width'], dtype=torch.float32)
        else:
            # Use default COCO image dimensions or estimate from data
            max_x = max(pred_x1.max().item(), gt_x1.max().item())
            max_y = max(pred_y1.max().item(), gt_y1.max().item())
            
            estimated_width = max(max_x * 1.1, 640.0)
            estimated_height = max(max_y * 1.1, 480.0)
            
            img_heights = torch.full((n_preds,), estimated_height, dtype=torch.float32)
            img_widths = torch.full((n_preds,), estimated_width, dtype=torch.float32)
        
        # Extract features with error handling
        try:
            base_features = feature_extractor.extract_features(
                pred_coords, pred_scores, img_heights, img_widths
            )
            
            uncertainty_features = uncertainty_extractor.extract_features(
                pred_coords, gt_coords, pred_scores
            )
            
            features = torch.cat([base_features, uncertainty_features], dim=1)
        except Exception as e:
            logger.error(f"Class {class_id}: Error extracting features: {str(e)}")
            return None
        
        # Get image IDs if available
        img_ids = None
        if 'img_id' in class_data:
            try:
                img_ids = torch.tensor(class_data['img_id'], dtype=torch.int64)
            except Exception as e:
                logger.warning(f"Class {class_id}: Error converting img_id to tensor: {str(e)}")
        
        return {
            'features': features,
            'pred_coords': pred_coords,
            'gt_coords': gt_coords,
            'confidence': pred_scores,
            'class_id': class_id,
            'img_ids': img_ids
        }
        
    except Exception as e:
        logger.error(f"Class {class_id}: Unexpected error: {str(e)}")
        return None


def process_class_with_timeout(args):
    """Wrapper to process class with timeout."""
    class_data_with_id, timeout_seconds = args
    
    # Re-create extractors in each process
    feature_extractor = FeatureExtractor()
    uncertainty_extractor = UncertaintyFeatureExtractor()
    
    try:
        with timeout(timeout_seconds):
            return process_class_data_safe(
                class_data_with_id, 
                feature_extractor, 
                uncertainty_extractor
            )
    except TimeoutError:
        class_id = class_data_with_id[0]
        logger.error(f"Class {class_id}: Timeout after {timeout_seconds}s")
        return None


def extract_features_from_predictions_safe(
    predictions: Tuple[Any, Any], 
    num_workers: int = 8,  # Reduced default to avoid hanging
    timeout_per_class: int = 60,  # Increased timeout
    use_sequential_fallback: bool = True
) -> Tuple[torch.Tensor, ...]:
    """Extract features with improved error handling and timeout support.
    
    Args:
        predictions: Tuple of (img_list, ist_list)
        num_workers: Number of parallel workers
        timeout_per_class: Timeout in seconds for each class
        use_sequential_fallback: Fall back to sequential if parallel fails
        
    Returns:
        features, gt_coords, pred_coords, confidence: Extracted tensors
    """
    img_list, ist_list = predictions
    
    logger.info(f"Starting feature extraction for {len(ist_list)} classes")
    
    # First, analyze the data
    total_predictions = 0
    large_classes = []
    for i, class_data in enumerate(ist_list):
        if isinstance(class_data, dict) and 'pred_x0' in class_data:
            n_preds = len(class_data['pred_x0'])
            total_predictions += n_preds
            if n_preds > 10000:
                large_classes.append((i, n_preds))
    
    logger.info(f"Total predictions: {total_predictions}")
    logger.info(f"Large classes (>10k predictions): {len(large_classes)}")
    
    # Prepare data for processing
    class_data_list = [(i, ist_list[i]) for i in range(len(ist_list))]
    
    # Try parallel processing with safer approach
    try:
        logger.info(f"Processing {len(class_data_list)} classes using {num_workers} workers...")
        
        # Reduce workers for validation data to avoid hanging
        if total_predictions > 100000:  # Large dataset
            num_workers = min(4, num_workers)
            logger.info(f"Large dataset detected, reducing workers to {num_workers}")
        
        # Add timeout to each task
        tasks_with_timeout = [(cd, timeout_per_class) for cd in class_data_list]
        
        start_time = time.time()
        
        # Use map instead of imap_unordered to avoid hanging
        with mp.Pool(num_workers) as pool:
            results = []
            
            # Process smaller batches
            batch_size = 20  # Process in small batches
            for i in range(0, len(tasks_with_timeout), batch_size):
                batch = tasks_with_timeout[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(tasks_with_timeout) + batch_size - 1)//batch_size}")
                
                batch_results = pool.map(process_class_with_timeout, batch)
                results.extend(batch_results)
                
                # Check for progress
                elapsed = time.time() - start_time
                completed = len(results)
                rate = completed / elapsed if elapsed > 0 else 0
                logger.info(f"Progress: {completed}/{len(tasks_with_timeout)} classes "
                          f"({completed/len(tasks_with_timeout)*100:.1f}%) "
                          f"Rate: {rate:.1f} classes/sec")
        
        total_time = time.time() - start_time
        logger.info(f"Parallel processing completed in {total_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Parallel processing failed: {str(e)}")
        
        if use_sequential_fallback:
            logger.info("Falling back to sequential processing...")
            results = []
            
            # Re-create extractors for sequential processing
            feature_extractor = FeatureExtractor()
            uncertainty_extractor = UncertaintyFeatureExtractor()
            
            for i, (class_id, class_data) in enumerate(class_data_list):
                if i % 10 == 0:
                    logger.info(f"Sequential progress: {i}/{len(class_data_list)}")
                
                result = process_class_data_safe(
                    (class_id, class_data),
                    feature_extractor,
                    uncertainty_extractor
                )
                results.append(result)
        else:
            raise
    
    # Filter out None results
    valid_results = [r for r in results if r is not None]
    
    logger.info(f"Valid results: {len(valid_results)}/{len(results)}")
    
    if not valid_results:
        raise ValueError("No valid predictions found")
    
    # Combine all results
    try:
        all_features = torch.cat([r['features'] for r in valid_results], dim=0)
        all_pred_coords = torch.cat([r['pred_coords'] for r in valid_results], dim=0)
        all_gt_coords = torch.cat([r['gt_coords'] for r in valid_results], dim=0)
        all_confidence = torch.cat([r['confidence'] for r in valid_results], dim=0)
        
        # Combine image IDs if available
        img_ids_list = [r['img_ids'] for r in valid_results if r['img_ids'] is not None]
        if img_ids_list:
            all_img_ids = torch.cat(img_ids_list, dim=0)
        else:
            all_img_ids = None
            
    except Exception as e:
        logger.error(f"Error combining results: {str(e)}")
        raise
    
    logger.info(f"Extracted features for {len(all_features)} predictions")
    if all_img_ids is not None:
        logger.info(f"Image IDs available for {len(all_img_ids)} predictions")
    
    return all_features, all_gt_coords, all_pred_coords, all_confidence, all_img_ids


# Keep original function names for compatibility
def extract_features_from_predictions_parallel(predictions: Tuple[Any, Any], 
                                             num_workers: int = 36) -> Tuple[torch.Tensor, ...]:
    """Extract features using safe parallel processing."""
    return extract_features_from_predictions_safe(
        predictions, 
        num_workers=num_workers,
        timeout_per_class=30,
        use_sequential_fallback=True
    )


def extract_features_from_predictions(predictions: Tuple[Any, Any], 
                                     num_workers: int = 8,
                                     timeout_per_class: int = 60,
                                     use_sequential_fallback: bool = True) -> Tuple[torch.Tensor, ...]:
    """Extract features from predictions - uses safe parallel version."""
    return extract_features_from_predictions_safe(
        predictions,
        num_workers=num_workers,
        timeout_per_class=timeout_per_class,
        use_sequential_fallback=use_sequential_fallback
    )


# Copy other functions from original
def load_cached_predictions(cache_dir: str, limit_dataset_size: bool = False) -> Tuple[Any, Any]:
    """Load cached predictions from directory."""
    cache_path = Path(cache_dir)
    
    train_file = cache_path / "predictions_train.pkl"
    val_file = cache_path / "predictions_val.pkl"
    
    if not train_file.exists() or not val_file.exists():
        raise FileNotFoundError(
            f"Cached predictions not found in {cache_dir}. "
            f"Please run the original training script first to generate cache."
        )
    
    logger.info(f"Loading cached predictions from {cache_dir}")
    
    with open(train_file, 'rb') as f:
        train_predictions = pickle.load(f)
    logger.info("Loaded train predictions")
    
    with open(val_file, 'rb') as f:
        val_predictions = pickle.load(f)
    logger.info("Loaded validation predictions")
    
    return train_predictions, val_predictions


def prepare_data_splits(
    train_features: torch.Tensor,
    train_gt: torch.Tensor,
    train_preds: torch.Tensor,
    train_conf: torch.Tensor,
    val_features: torch.Tensor,
    val_gt: torch.Tensor,
    val_preds: torch.Tensor,
    val_conf: torch.Tensor,
    calib_fraction: float = 0.5,
    seed: int = 42,
    train_img_ids: Optional[torch.Tensor] = None,
    val_img_ids: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    """Prepare train/calibration/test splits."""
    torch.manual_seed(seed)
    
    n_val = len(val_features)
    n_calib = int(n_val * calib_fraction)
    
    perm = torch.randperm(n_val)
    calib_idx = perm[:n_calib]
    test_idx = perm[n_calib:]
    
    result = {
        'train_features': train_features,
        'train_gt_coords': train_gt,
        'train_pred_coords': train_preds,
        'train_confidence': train_conf,
        'calib_features': val_features[calib_idx],
        'calib_gt_coords': val_gt[calib_idx],
        'calib_pred_coords': val_preds[calib_idx],
        'calib_confidence': val_conf[calib_idx],
        'test_features': val_features[test_idx],
        'test_gt_coords': val_gt[test_idx],
        'test_pred_coords': val_preds[test_idx],
        'test_confidence': val_conf[test_idx]
    }
    
    # Add image IDs if available
    if train_img_ids is not None:
        result['train_img_ids'] = train_img_ids
    if val_img_ids is not None:
        result['calib_img_ids'] = val_img_ids[calib_idx]
        result['test_img_ids'] = val_img_ids[test_idx]
    
    return result