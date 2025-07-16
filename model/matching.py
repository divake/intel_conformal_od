"""
Object matching utilities for conformal prediction.
Matches predicted boxes to ground truth boxes based on IoU.
"""

import torch
from detectron2.structures import pairwise_iou


def matching(
    gt_box, pred_box, gt_class, pred_class, pred_score,
    pred_score_all=None, pred_logits_all=None,
    box_matching="box", class_match=True, thresh=0.5, return_idx=False
):
    """
    Matches predicted boxes to ground truth boxes.
    
    Args:
        gt_box: Ground truth boxes
        pred_box: Predicted boxes
        gt_class: Ground truth classes
        pred_class: Predicted classes
        pred_score: Predicted scores
        pred_score_all: All class scores (optional)
        pred_logits_all: All class logits (optional)
        box_matching: Matching method ("box" or other)
        class_match: Whether to enforce class matching
        thresh: IoU threshold for matching
        return_idx: Whether to return indices
        
    Returns:
        Matched boxes, classes, scores, and optionally indices
    """
    # Handle empty predictions
    if len(pred_box) == 0:
        empty_result = (
            gt_box.tensor.new_empty((0, 4)),
            gt_box.tensor.new_empty((0, 4)),
            gt_class.new_empty(0),
            pred_class.new_empty(0),
            pred_score.new_empty(0),
        )
        if pred_score_all is not None:
            empty_result += (pred_score_all.new_empty((0, pred_score_all.shape[1])),)
        if pred_logits_all is not None:
            empty_result += (pred_logits_all.new_empty((0, pred_logits_all.shape[1])),)
        empty_result += (False,)
        if return_idx:
            empty_result += (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
        return empty_result
    
    # Compute IoU matrix
    iou_matrix = pairwise_iou(gt_box, pred_box)
    
    # Initialize lists for matches
    matched_gt_boxes = []
    matched_pred_boxes = []
    matched_gt_classes = []
    matched_pred_classes = []
    matched_pred_scores = []
    matched_pred_scores_all = []
    matched_pred_logits_all = []
    matched_pred_idx = []
    matched_class_idx = []
    
    # Track which predictions have been matched
    pred_matched = torch.zeros(len(pred_box), dtype=torch.bool)
    
    # For each ground truth box, find best matching prediction
    for gt_idx in range(len(gt_box)):
        # Get IoUs for this GT box
        ious = iou_matrix[gt_idx]
        
        # Filter by class if required
        if class_match:
            class_mask = pred_class == gt_class[gt_idx]
            ious = ious * class_mask.float()
        
        # Filter by already matched predictions
        ious[pred_matched] = 0
        
        # Find best match
        if ious.max() >= thresh:
            pred_idx = ious.argmax()
            pred_matched[pred_idx] = True
            
            # Store matches
            matched_gt_boxes.append(gt_box[gt_idx].tensor)
            matched_pred_boxes.append(pred_box[pred_idx].tensor)
            matched_gt_classes.append(gt_class[gt_idx])
            matched_pred_classes.append(pred_class[pred_idx])
            matched_pred_scores.append(pred_score[pred_idx])
            
            if pred_score_all is not None:
                matched_pred_scores_all.append(pred_score_all[pred_idx])
            if pred_logits_all is not None:
                matched_pred_logits_all.append(pred_logits_all[pred_idx])
            
            if return_idx:
                matched_pred_idx.append(pred_idx)
                matched_class_idx.append(gt_class[gt_idx])
    
    # Convert to tensors
    if matched_gt_boxes:
        matched_gt_boxes = torch.stack(matched_gt_boxes)
        matched_pred_boxes = torch.stack(matched_pred_boxes)
        matched_gt_classes = torch.stack(matched_gt_classes)
        matched_pred_classes = torch.stack(matched_pred_classes)
        matched_pred_scores = torch.stack(matched_pred_scores)
        
        if pred_score_all is not None:
            matched_pred_scores_all = torch.stack(matched_pred_scores_all)
        if pred_logits_all is not None:
            matched_pred_logits_all = torch.stack(matched_pred_logits_all)
            
        matches = True
    else:
        # No matches found
        matched_gt_boxes = gt_box.tensor.new_empty((0, 4))
        matched_pred_boxes = pred_box.tensor.new_empty((0, 4))
        matched_gt_classes = gt_class.new_empty(0)
        matched_pred_classes = pred_class.new_empty(0)
        matched_pred_scores = pred_score.new_empty(0)
        
        if pred_score_all is not None:
            matched_pred_scores_all = pred_score_all.new_empty((0, pred_score_all.shape[1]))
        if pred_logits_all is not None:
            matched_pred_logits_all = pred_logits_all.new_empty((0, pred_logits_all.shape[1]))
            
        matches = False
    
    # Build result tuple
    result = (
        matched_gt_boxes,
        matched_pred_boxes,
        matched_gt_classes,
        matched_pred_classes,
        matched_pred_scores,
    )
    
    if pred_score_all is not None:
        result += (matched_pred_scores_all,)
    if pred_logits_all is not None:
        result += (matched_pred_logits_all,)
    
    result += (matches,)
    
    if return_idx:
        if matched_pred_idx:
            result += (torch.tensor(matched_pred_idx), torch.tensor(matched_class_idx))
        else:
            result += (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
    
    return result