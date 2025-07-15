#!/usr/bin/env python
"""
Create visualization for high coverage images to showcase in paper.
Shows prediction boxes with intervals and ground truth.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import cv2
from pycocotools.coco import COCO

# Add paths
sys.path.append("/ssd_4TB/divake/conformal-od")
sys.path.append("/ssd_4TB/divake/conformal-od/detectron2")

# Import detectron2 components
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.structures import Boxes, Instances

# Import our modules
from learnable_scoring_fn.core_symmetric.models.cqn_model import ConditionalQuantileNetwork
from learnable_scoring_fn.calibrate_cqn import CalibratedCQN
from util import util
from plots import plot_util


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


def create_high_coverage_visualization(image_path, predictor, calibrated_model, 
                                     output_path, score_threshold=0.3):
    """Create visualization for high coverage image."""
    
    # Load image
    img_cv2 = cv2.imread(image_path)
    img_name = Path(image_path).stem
    img_h, img_w = img_cv2.shape[:2]
    
    print(f"\nCreating visualization for: {img_name}")
    print(f"Image size: {img_w}x{img_h}")
    
    # Get predictions
    outputs = predictor(img_cv2)
    instances = outputs["instances"].to("cpu")
    pred_boxes_tensor = instances.pred_boxes.tensor
    pred_scores = instances.scores
    pred_classes = instances.pred_classes
    
    # Load COCO metadata
    coco_ann_file = "/ssd_4TB/divake/conformal-od/data/coco/annotations/instances_val2017.json"
    coco = COCO(coco_ann_file)
    
    # Get ground truth
    img_id = int(img_name.lstrip('0'))
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    
    # Get class names
    class_names = util.get_coco_classes()
    
    # Filter predictions
    high_conf_mask = pred_scores >= score_threshold
    filtered_instances = Instances((img_h, img_w))
    filtered_instances.pred_boxes = Boxes(pred_boxes_tensor[high_conf_mask])
    filtered_instances.scores = pred_scores[high_conf_mask]
    filtered_instances.pred_classes = pred_classes[high_conf_mask]
    
    print(f"Predictions after filtering: {len(filtered_instances)}")
    
    # Extract features
    pred_boxes_np = filtered_instances.pred_boxes.tensor.numpy()
    pred_scores_np = filtered_instances.scores.numpy()
    features = extract_features_from_predictions(pred_boxes_np, pred_scores_np, img_cv2.shape)
    
    # Get calibrated intervals
    if len(features) > 0:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        features = features.to(device)
        pred_boxes_torch = torch.from_numpy(pred_boxes_np).to(device)
        calibrated_model.base_model = calibrated_model.base_model.to(device)
        
        lower_bounds, upper_bounds = calibrated_model.predict_intervals(
            features, pred_boxes_torch
        )
        
        # Calculate widths for visualization
        widths = (upper_bounds - pred_boxes_torch).cpu()
        quant_values = widths
    else:
        quant_values = torch.zeros((0, 4))
    
    # Get ground truth boxes
    gt_boxes = []
    for ann in anns:
        x, y, w, h = ann['bbox']
        gt_boxes.append([x, y, x+w, y+h])
    
    gt_boxes_tensor = Boxes(torch.tensor(gt_boxes, dtype=torch.float32)) if gt_boxes else Boxes(torch.zeros((0, 4)))
    
    print(f"Ground truth objects: {len(gt_boxes)}")
    
    # Prepare image dict for plot_util
    img_dict = {
        "file_name": image_path,
        "height": img_h,
        "width": img_w,
        "image": torch.from_numpy(img_cv2.transpose(2, 0, 1))
    }
    
    # Create visualization
    plot_util.d2_plot_pi(
        risk_control="std_conf",
        image=img_dict,
        gt_box=gt_boxes_tensor,
        pred=filtered_instances,
        quant=quant_values,
        channel_order="BGR",
        draw_labels=[],
        colors=["red", "green", "palegreen"],
        alpha=[1.0, 0.6, 0.4],
        lw=1.5,
        notebook=False,
        to_file=True,
        filename=output_path,
        label_gt=None,
        label_set=None
    )
    
    print(f"Visualization saved to: {output_path}")
    
    # Print class information
    if len(filtered_instances) > 0:
        print("\nDetected objects:")
        for i in range(len(filtered_instances)):
            class_name = class_names[filtered_instances.pred_classes[i]]
            score = filtered_instances.scores[i]
            print(f"  - {class_name}: {score:.2f}")


def main():
    """Create visualizations for best high coverage images."""
    
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
    
    # Best high coverage images from our search
    high_coverage_images = [
        {
            "name": "000000222458",  # 100% coverage, 8/8 objects
            "path": "/ssd_4TB/divake/conformal-od/data/coco/val2017/000000222458.jpg",
            "output": "high_coverage_100_percent_8objects.jpg"
        },
        {
            "name": "000000224222",  # 100% coverage, 4/4 objects
            "path": "/ssd_4TB/divake/conformal-od/data/coco/val2017/000000224222.jpg",
            "output": "high_coverage_100_percent_4objects.jpg"
        },
        {
            "name": "000000573626",  # 100% coverage, 3/3 objects
            "path": "/ssd_4TB/divake/conformal-od/data/coco/val2017/000000573626.jpg",
            "output": "high_coverage_100_percent_3objects.jpg"
        }
    ]
    
    print("\n" + "="*80)
    print("CREATING HIGH COVERAGE VISUALIZATIONS FOR PAPER")
    print("="*80)
    
    for img_config in high_coverage_images:
        print(f"\n{'='*60}")
        print(f"Image: {img_config['name']}")
        print(f"Expected: 100% coverage")
        
        create_high_coverage_visualization(
            img_config["path"],
            predictor,
            calibrated_model,
            img_config["output"],
            score_threshold=0.3
        )
    
    print("\n" + "="*80)
    print("HIGH COVERAGE VISUALIZATIONS COMPLETE!")
    print("="*80)
    print("These images demonstrate that our method CAN achieve perfect 100% coverage")
    print("when the model's uncertainty estimates align well with the actual errors.")
    print("="*80)


if __name__ == "__main__":
    main()