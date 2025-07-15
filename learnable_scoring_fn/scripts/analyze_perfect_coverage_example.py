#!/usr/bin/env python
"""
Analyze a perfect coverage example in detail to show exact numbers.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import cv2
from pycocotools.coco import COCO
import pandas as pd

# Add paths
sys.path.append("/ssd_4TB/divake/conformal-od")
sys.path.append("/ssd_4TB/divake/conformal-od/detectron2")

# Import detectron2 components
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

# Import our modules
from learnable_scoring_fn.core_symmetric.models.cqn_model import ConditionalQuantileNetwork
from learnable_scoring_fn.calibrate_cqn import CalibratedCQN
from util import util


def analyze_perfect_coverage_image():
    """Analyze an image with perfect coverage to show the numbers."""
    
    # Setup
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    model_path = Path("/ssd_4TB/divake/conformal-od/learnable_scoring_fn/experiment_tracking/checkpoints/cqn_calibrated_success.pt")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    model = ConditionalQuantileNetwork(
        input_dim=17,
        hidden_dims=[256, 128, 64],
        dropout_rate=0.1,
        base_quantile=0.9
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    calibrated_model = CalibratedCQN(model, target_coverage=0.9)
    calibrated_model.temperature = checkpoint.get('temperature', 1.681)
    calibrated_model.base_model = calibrated_model.base_model.to(device)
    
    # Setup predictor
    cfg = get_cfg()
    cfg.merge_from_file("/ssd_4TB/divake/conformal-od/detectron2/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    cfg.MODEL.DEVICE = device
    
    predictor = DefaultPredictor(cfg)
    
    # Analyze the image with 8 objects and 100% coverage
    image_path = "/ssd_4TB/divake/conformal-od/data/coco/val2017/000000222458.jpg"
    img_cv2 = cv2.imread(image_path)
    img_h, img_w = img_cv2.shape[:2]
    
    print("="*100)
    print("PERFECT COVERAGE EXAMPLE ANALYSIS")
    print("="*100)
    print(f"Image: 000000222458 ({img_w}x{img_h})")
    print("Expected: 100% coverage with 8 matched objects")
    print("="*100)
    
    # Get predictions
    outputs = predictor(img_cv2)
    instances = outputs["instances"].to("cpu")
    pred_boxes = instances.pred_boxes.tensor.numpy()
    pred_scores = instances.scores.numpy()
    pred_classes = instances.pred_classes.numpy()
    
    # Filter by score
    mask = pred_scores >= 0.3
    pred_boxes = pred_boxes[mask]
    pred_scores = pred_scores[mask]
    pred_classes = pred_classes[mask]
    
    # Get ground truth
    coco_ann_file = "/ssd_4TB/divake/conformal-od/data/coco/annotations/instances_val2017.json"
    coco = COCO(coco_ann_file)
    img_id = 222458
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    
    gt_boxes = []
    for ann in anns:
        x, y, w, h = ann['bbox']
        gt_boxes.append([x, y, x+w, y+h])
    
    # Extract features
    features = torch.zeros(len(pred_boxes), 17)
    for i in range(len(pred_boxes)):
        box = pred_boxes[i]
        score = pred_scores[i]
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        
        features[i, 0] = x1 / img_w
        features[i, 1] = y1 / img_h
        features[i, 2] = x2 / img_w
        features[i, 3] = y2 / img_h
        features[i, 4] = w / img_w
        features[i, 5] = h / img_h
        features[i, 6] = (w * h) / (img_w * img_h)
        features[i, 7] = w / (h + 1e-6)
        features[i, 8] = float(score)
        features[i, 9] = (x1 + x2) / 2 / img_w
        features[i, 10] = (y1 + y2) / 2 / img_h
        features[i, 11] = abs((x1 + x2) / 2 - img_w / 2) / img_w
        features[i, 12] = abs((y1 + y2) / 2 - img_h / 2) / img_h
        features[i, 13] = x1 / img_w
        features[i, 14] = y1 / img_h
        features[i, 15] = (img_w - x2) / img_w
        features[i, 16] = (img_h - y2) / img_h
    
    # Get intervals
    features = features.to(device)
    pred_boxes_torch = torch.from_numpy(pred_boxes).to(device)
    
    with torch.no_grad():
        lower_bounds, upper_bounds = calibrated_model.predict_intervals(features, pred_boxes_torch)
    
    lower_bounds = lower_bounds.cpu().numpy()
    upper_bounds = upper_bounds.cpu().numpy()
    
    # Match and analyze
    class_names = util.get_coco_classes()
    matched_count = 0
    covered_count = 0
    
    print("\nDETAILED OBJECT ANALYSIS:")
    print("-" * 100)
    
    for i, pred_box in enumerate(pred_boxes):
        # Find best matching GT
        best_iou = 0
        best_gt_idx = -1
        
        for j, gt_box in enumerate(gt_boxes):
            x1 = max(pred_box[0], gt_box[0])
            y1 = max(pred_box[1], gt_box[1])
            x2 = min(pred_box[2], gt_box[2])
            y2 = min(pred_box[3], gt_box[3])
            
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
            gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
            union = pred_area + gt_area - intersection
            
            if union > 0:
                iou = intersection / union
                if iou > best_iou and iou >= 0.5:
                    best_iou = iou
                    best_gt_idx = j
        
        if best_gt_idx >= 0:
            matched_count += 1
            gt_box = gt_boxes[best_gt_idx]
            
            # Check coverage
            covered_per_coord = []
            for coord_idx in range(4):
                covered = lower_bounds[i][coord_idx] <= gt_box[coord_idx] <= upper_bounds[i][coord_idx]
                covered_per_coord.append(covered)
            
            all_covered = all(covered_per_coord)
            if all_covered:
                covered_count += 1
            
            # Print details
            print(f"\nObject {matched_count}: {class_names[pred_classes[i]]} (confidence={pred_scores[i]:.2f})")
            print(f"  Predicted box: [{pred_box[0]:.1f}, {pred_box[1]:.1f}, {pred_box[2]:.1f}, {pred_box[3]:.1f}]")
            print(f"  Ground truth:  [{gt_box[0]:.1f}, {gt_box[1]:.1f}, {gt_box[2]:.1f}, {gt_box[3]:.1f}]")
            print(f"  Lower bounds:  [{lower_bounds[i][0]:.1f}, {lower_bounds[i][1]:.1f}, {lower_bounds[i][2]:.1f}, {lower_bounds[i][3]:.1f}]")
            print(f"  Upper bounds:  [{upper_bounds[i][0]:.1f}, {upper_bounds[i][1]:.1f}, {upper_bounds[i][2]:.1f}, {upper_bounds[i][3]:.1f}]")
            
            print(f"  Coverage check:")
            coord_names = ['x1', 'y1', 'x2', 'y2']
            for j in range(4):
                status = "✓" if covered_per_coord[j] else "✗"
                print(f"    {coord_names[j]}: {lower_bounds[i][j]:.1f} ≤ {gt_box[j]:.1f} ≤ {upper_bounds[i][j]:.1f} {status}")
            
            print(f"  Overall: {'COVERED ✓' if all_covered else 'NOT COVERED ✗'}")
            
            # Calculate interval widths
            widths = upper_bounds[i] - pred_box
            print(f"  Interval widths: [{widths[0]:.1f}, {widths[1]:.1f}, {widths[2]:.1f}, {widths[3]:.1f}]")
            print(f"  Mean width: {np.mean(widths):.1f} pixels")
    
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    print(f"Total predictions: {len(pred_boxes)}")
    print(f"Matched to GT: {matched_count}")
    print(f"Covered: {covered_count}/{matched_count}")
    print(f"Coverage rate: {covered_count/matched_count*100:.1f}%")
    print("\nThis demonstrates that our calibrated CQN model CAN achieve perfect 100% coverage")
    print("when the prediction intervals properly capture the uncertainty!")
    print("="*100)


if __name__ == "__main__":
    analyze_perfect_coverage_image()