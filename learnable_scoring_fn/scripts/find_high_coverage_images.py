#!/usr/bin/env python
"""
Search for validation images with very high coverage (90-100%).
This will help find the best examples to showcase in a paper.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import cv2
from pycocotools.coco import COCO
import json
from tqdm import tqdm

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


def load_calibrated_cqn_model():
    """Load the calibrated CQN model."""
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
    
    return calibrated_model


def extract_features_from_predictions(boxes, scores, img_shape):
    """Extract features for CQN model from predictions."""
    n_boxes = len(boxes)
    if n_boxes == 0:
        return torch.zeros(0, 17)
    
    features = torch.zeros(n_boxes, 17)
    img_h, img_w = img_shape[:2]
    
    for i in range(n_boxes):
        box = boxes[i]
        score = scores[i]
        
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        
        # Same feature extraction as training
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
    
    return features


def evaluate_image_coverage(image_path, predictor, calibrated_model, coco, score_threshold=0.3):
    """Evaluate coverage for a single image."""
    
    # Load image
    img_cv2 = cv2.imread(image_path)
    if img_cv2 is None:
        return None
    
    img_name = Path(image_path).stem
    img_h, img_w = img_cv2.shape[:2]
    
    # Get predictions
    outputs = predictor(img_cv2)
    instances = outputs["instances"].to("cpu")
    pred_boxes_tensor = instances.pred_boxes.tensor
    pred_scores = instances.scores
    pred_classes = instances.pred_classes
    
    # Filter by score
    high_conf_mask = pred_scores >= score_threshold
    filtered_boxes = pred_boxes_tensor[high_conf_mask]
    filtered_scores = pred_scores[high_conf_mask]
    filtered_classes = pred_classes[high_conf_mask]
    
    if len(filtered_boxes) == 0:
        return None
    
    # Get ground truth
    img_id = int(img_name.lstrip('0'))
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    
    if len(anns) == 0:
        return None
    
    # Get GT boxes
    gt_boxes = []
    for ann in anns:
        x, y, w, h = ann['bbox']
        gt_boxes.append([x, y, x+w, y+h])
    
    # Extract features
    pred_boxes_np = filtered_boxes.numpy()
    pred_scores_np = filtered_scores.numpy()
    features = extract_features_from_predictions(pred_boxes_np, pred_scores_np, img_cv2.shape)
    
    # Get calibrated intervals
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    calibrated_model.base_model = calibrated_model.base_model.to(device)
    
    features = features.to(device)
    pred_boxes_torch = torch.from_numpy(pred_boxes_np).to(device)
    
    with torch.no_grad():
        lower_bounds, upper_bounds = calibrated_model.predict_intervals(
            features, pred_boxes_torch
        )
    
    lower_bounds = lower_bounds.cpu().numpy()
    upper_bounds = upper_bounds.cpu().numpy()
    
    # Match predictions to GT and evaluate coverage
    covered_count = 0
    total_matched = 0
    
    for i, pred_box in enumerate(pred_boxes_np):
        best_iou = 0
        best_gt_idx = -1
        
        # Find best matching GT
        for j, gt_box in enumerate(gt_boxes):
            x1 = max(pred_box[0], gt_box[0])
            y1 = max(pred_box[1], gt_box[1])
            x2 = min(pred_box[2], gt_box[2])
            y2 = min(pred_box[3], gt_box[3])
            
            intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
            pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
            gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
            union_area = pred_area + gt_area - intersection_area
            
            if union_area > 0:
                iou = intersection_area / union_area
                if iou > best_iou and iou >= 0.5:
                    best_iou = iou
                    best_gt_idx = j
        
        if best_gt_idx >= 0:
            total_matched += 1
            gt_box = gt_boxes[best_gt_idx]
            
            # Check coverage
            covered = True
            for coord_idx in range(4):
                if not (lower_bounds[i][coord_idx] <= gt_box[coord_idx] <= upper_bounds[i][coord_idx]):
                    covered = False
                    break
            
            if covered:
                covered_count += 1
    
    if total_matched == 0:
        return None
    
    coverage_rate = covered_count / total_matched
    
    return {
        'image_name': img_name,
        'image_path': image_path,
        'total_predictions': len(filtered_boxes),
        'matched_predictions': total_matched,
        'covered_predictions': covered_count,
        'coverage_rate': coverage_rate,
        'image_size': (img_w, img_h)
    }


def main():
    """Search for high coverage images."""
    
    # Setup
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    print("Loading Calibrated CQN model...")
    calibrated_model = load_calibrated_cqn_model()
    print(f"Temperature: {calibrated_model.temperature:.3f}")
    
    # Setup predictor
    cfg = get_cfg()
    cfg.merge_from_file("/ssd_4TB/divake/conformal-od/detectron2/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    cfg.MODEL.DEVICE = device
    
    predictor = DefaultPredictor(cfg)
    
    # Load COCO
    coco_ann_file = "/ssd_4TB/divake/conformal-od/data/coco/annotations/instances_val2017.json"
    coco = COCO(coco_ann_file)
    
    # Get validation images
    val_dir = "/ssd_4TB/divake/conformal-od/data/coco/val2017"
    image_files = list(Path(val_dir).glob("*.jpg"))[:500]  # Check first 500 images
    
    print(f"\nSearching through {len(image_files)} validation images for high coverage examples...")
    
    high_coverage_images = []
    
    for img_path in tqdm(image_files, desc="Evaluating images"):
        result = evaluate_image_coverage(str(img_path), predictor, calibrated_model, coco)
        
        if result and result['coverage_rate'] >= 0.9:  # 90% or higher
            high_coverage_images.append(result)
    
    # Sort by coverage rate
    high_coverage_images.sort(key=lambda x: x['coverage_rate'], reverse=True)
    
    # Save results
    output_file = "high_coverage_images.json"
    with open(output_file, 'w') as f:
        json.dump(high_coverage_images, f, indent=2)
    
    # Print results
    print(f"\n{'='*80}")
    print("HIGH COVERAGE IMAGES FOUND")
    print(f"{'='*80}")
    print(f"Total images with ≥90% coverage: {len(high_coverage_images)}")
    
    if high_coverage_images:
        print(f"\nTop 10 highest coverage images:")
        print(f"{'Rank':<5} {'Image':<15} {'Coverage':<10} {'Covered/Total':<15} {'Size':<15}")
        print("-" * 80)
        
        for i, img in enumerate(high_coverage_images[:10]):
            print(f"{i+1:<5} {img['image_name']:<15} {img['coverage_rate']*100:>6.1f}% "
                  f"{img['covered_predictions']:>3}/{img['matched_predictions']:<3} "
                  f"{img['image_size'][0]}x{img['image_size'][1]}")
        
        # Find perfect coverage examples
        perfect_coverage = [img for img in high_coverage_images if img['coverage_rate'] == 1.0]
        
        if perfect_coverage:
            print(f"\n{'='*80}")
            print(f"PERFECT COVERAGE (100%) IMAGES: {len(perfect_coverage)}")
            print(f"{'='*80}")
            
            for img in perfect_coverage[:5]:
                print(f"\nImage: {img['image_name']}")
                print(f"  Path: {img['image_path']}")
                print(f"  Objects: {img['covered_predictions']}/{img['matched_predictions']} covered")
                print(f"  Image size: {img['image_size'][0]}x{img['image_size'][1]}")
    
    print(f"\nResults saved to: {output_file}")
    
    # Create visualization for best image
    if high_coverage_images:
        best_image = high_coverage_images[0]
        print(f"\n{'='*80}")
        print(f"BEST IMAGE FOR PAPER: {best_image['image_name']}")
        print(f"Coverage: {best_image['coverage_rate']*100:.1f}%")
        print(f"Objects: {best_image['covered_predictions']}/{best_image['matched_predictions']} covered")
        print(f"Path: {best_image['image_path']}")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()