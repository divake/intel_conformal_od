"""
Standalone collect_predictions function for learnable scoring function training.

This module provides a simplified interface to collect model predictions
that can be used by the learnable scoring function training pipeline.
"""

import torch
import json
import os
from tqdm import tqdm
from detectron2.data.detection_utils import annotations_to_instances

# Import from the project structure
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model import model_loader
from model import matching


def collect_predictions(cfg, data_list, alpha=0.1, label_set='class_threshold', 
                       label_alpha=0.1, risk_control=True, save_label_set=True,
                       logger=None, iou_thresh=0.5, box_matching='box', 
                       class_matching=True):
    """
    Collect model predictions for use in learnable scoring function training.
    
    This is a simplified version of the collect_predictions methods found
    in the various control classes, designed specifically for training
    the learnable scoring function.
    
    Args:
        cfg: Configuration object
        data_list: List of data dictionaries
        alpha: Alpha level for conformal prediction
        label_set: Label set type
        label_alpha: Label alpha
        risk_control: Risk control flag
        save_label_set: Save label set flag
        logger: Logger instance
        iou_thresh: IoU threshold for matching
        box_matching: Box matching method
        class_matching: Whether to use class matching
        
    Returns:
        img_list: List of image information
        ist_list: List of instance information
    """
    if logger is not None:
        logger.info("Building model for prediction collection...")
    
    # Build and load model
    cfg_model, model = model_loader.d2_build_model(cfg, logger=logger)
    model_loader.d2_load_model(cfg_model, model, logger=logger)
    
    # Set up data loader
    from data import data_loader
    dataloader = data_loader.d2_load_dataset_from_dict(
        data_list, cfg, cfg_model, logger=logger
    )
    
    if logger is not None:
        logger.info(f"Collecting predictions with IoU threshold: {iou_thresh}")
        logger.info(f"Box matching: {box_matching}, Class matching: {class_matching}")
    
    # Initialize storage
    img_list = []
    ist_list = []
    
    # Collect predictions
    model.eval()
    with torch.no_grad(), tqdm(dataloader, desc="Collecting Predictions") as loader:
        for i, img in enumerate(loader):
            # Convert annotations to instances
            gt = annotations_to_instances(
                img[0]["annotations"], (img[0]["height"], img[0]["width"])
            )
            gt_box, gt_class = gt.gt_boxes, gt.gt_classes
            
            # Get model predictions
            pred = model(img)
            pred_ist = pred[0]["instances"].to("cpu")
            pred_box = pred_ist.pred_boxes
            pred_class = pred_ist.pred_classes
            pred_score = pred_ist.scores
            pred_score_all = pred_ist.scores_all
            pred_logits_all = pred_ist.logits_all
            
            # Object matching process (predictions to ground truths)
            (
                gt_box_matched,
                pred_box_matched,
                gt_class_matched,
                pred_class_matched,
                pred_score_matched,
                pred_score_all_matched,
                pred_logits_all_matched,
                matches,
            ) = matching.matching(
                gt_box,
                pred_box,
                gt_class,
                pred_class,
                pred_score,
                pred_score_all,
                pred_logits_all,
                box_matching=box_matching,
                class_match=class_matching,
                thresh=iou_thresh,
            )
            
            # Store prediction information in a simplified format
            if matches:
                # Convert to basic data structures for JSON serialization
                # Use actual COCO image ID instead of loop index
                actual_image_id = img[0].get("image_id", i)  # Fallback to i if not available
                
                img_info = {
                    'img_id': actual_image_id,  # Changed to use actual image ID
                    'height': img[0]["height"],
                    'width': img[0]["width"],
                    'nr_matches': len(gt_class_matched)
                }
                img_list.append(img_info)
                
                # For each class, store the relevant information
                for class_id in torch.unique(gt_class_matched):
                    class_mask = (gt_class_matched == class_id)
                    
                    if class_mask.sum() > 0:
                        class_id_int = int(class_id.item())
                        
                        # Ensure ist_list has enough elements
                        while len(ist_list) <= class_id_int:
                            ist_list.append({
                                'gt_x0': [], 'gt_y0': [], 'gt_x1': [], 'gt_y1': [],
                                'pred_x0': [], 'pred_y0': [], 'pred_x1': [], 'pred_y1': [],
                                'pred_score': [], 'pred_score_all': [], 'pred_logits_all': []
                            })
                        
                        # Extract coordinate information
                        gt_coords = gt_box_matched.tensor[class_mask]
                        pred_coords = pred_box_matched.tensor[class_mask]
                        scores = pred_score_matched[class_mask]
                        scores_all = pred_score_all_matched[class_mask]
                        logits_all = pred_logits_all_matched[class_mask]
                        
                        # Store in lists
                        for j in range(len(gt_coords)):
                            ist_list[class_id_int]['gt_x0'].append(float(gt_coords[j][0]))
                            ist_list[class_id_int]['gt_y0'].append(float(gt_coords[j][1]))
                            ist_list[class_id_int]['gt_x1'].append(float(gt_coords[j][2]))
                            ist_list[class_id_int]['gt_y1'].append(float(gt_coords[j][3]))
                            
                            ist_list[class_id_int]['pred_x0'].append(float(pred_coords[j][0]))
                            ist_list[class_id_int]['pred_y0'].append(float(pred_coords[j][1]))
                            ist_list[class_id_int]['pred_x1'].append(float(pred_coords[j][2]))
                            ist_list[class_id_int]['pred_y1'].append(float(pred_coords[j][3]))
                            
                            ist_list[class_id_int]['pred_score'].append(float(scores[j]))
                            ist_list[class_id_int]['pred_score_all'].append(scores_all[j].tolist())
                            ist_list[class_id_int]['pred_logits_all'].append(logits_all[j].tolist())
            
            # Clean up memory
            del gt, pred, pred_ist
    
    if logger is not None:
        logger.info(f"Collected predictions from {len(img_list)} images")
        logger.info(f"Found data for {len(ist_list)} classes")
    
    return img_list, ist_list


def load_cached_predictions(cache_dir, data_name, logger=None):
    """
    Load cached predictions from disk.
    
    Args:
        cache_dir: Directory containing cached predictions
        data_name: Dataset name
        logger: Logger instance
        
    Returns:
        img_list: List of image information
        ist_list: List of instance information
    """
    img_list_path = os.path.join(cache_dir, f'{data_name}_img_list.json')
    ist_list_path = os.path.join(cache_dir, f'{data_name}_ist_list.json')
    
    if not os.path.exists(img_list_path) or not os.path.exists(ist_list_path):
        raise FileNotFoundError(f"Cached predictions not found in {cache_dir}")
    
    if logger is not None:
        logger.info(f"Loading cached predictions from {cache_dir}")
    
    with open(img_list_path, 'r') as f:
        img_list = json.load(f)
    with open(ist_list_path, 'r') as f:
        ist_list = json.load(f)
    
    return img_list, ist_list


def save_predictions(img_list, ist_list, cache_dir, data_name, logger=None):
    """
    Save predictions to cache directory.
    
    Args:
        img_list: List of image information
        ist_list: List of instance information
        cache_dir: Directory to save cached predictions
        data_name: Dataset name
        logger: Logger instance
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    img_list_path = os.path.join(cache_dir, f'{data_name}_img_list.json')
    ist_list_path = os.path.join(cache_dir, f'{data_name}_ist_list.json')
    
    if logger is not None:
        logger.info(f"Saving predictions to {cache_dir}")
    
    with open(img_list_path, 'w') as f:
        json.dump(img_list, f)
    with open(ist_list_path, 'w') as f:
        json.dump(ist_list, f)
    
    if logger is not None:
        logger.info("Predictions saved successfully") 