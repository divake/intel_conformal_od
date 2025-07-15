#!/usr/bin/env python3
"""
Generate cache model from Faster R-CNN checkpoint.

This script:
1. Loads the Faster R-CNN model from checkpoint
2. Runs inference on COCO dataset (train and validation)
3. Extracts features using FeatureExtractor
4. Saves cache in the expected format for learnable scoring function training

Usage:
    python generate_cache.py --checkpoint /path/to/faster_rcnn_X_101_32x8d_FPN_3x.pth
"""

import os
import sys
import argparse
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add the parent directory and detectron2 to sys.path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "detectron2"))

# Import detectron2 components
try:
    # Import detectron2 using the same pattern as your working code
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.data import MetadataCatalog, DatasetCatalog
    from detectron2.data.datasets import register_coco_instances
    from detectron2.structures import Boxes, Instances
    from detectron2.utils.logger import setup_logger
    from detectron2.data import build_detection_train_loader, build_detection_test_loader
    from detectron2.data.dataset_mapper import DatasetMapper
    from detectron2.data.transforms import ResizeShortestEdge
    from detectron2.data.build import get_detection_dataset_dicts
    from detectron2.utils.visualizer import Visualizer
    print("Detectron2 imported successfully")
except ImportError as e:
    print(f"Error importing detectron2: {e}")
    print("Please check that all dependencies are properly installed in your environment.")
    sys.exit(1)

# Import feature extractor from learnable_scoring_fn
try:
    from learnable_scoring_fn.feature_utils import FeatureExtractor
    print("Feature extractor imported successfully")
except ImportError as e:
    print(f"Error importing feature extractor: {e}")
    print("Make sure you're running from the correct directory")
    sys.exit(1)

# Setup logger
setup_logger()


class CacheGenerator:
    """Generate cache from Faster R-CNN checkpoint for learnable scoring function."""
    
    def __init__(self, checkpoint_path: str, coco_data_dir: str, output_dir: str, 
                 device: str = "auto", confidence_threshold: float = 0.1,
                 iou_threshold: float = 0.3, config_path: str = None):
        """
        Initialize cache generator.
        
        Args:
            checkpoint_path: Path to model checkpoint
            coco_data_dir: Path to COCO dataset directory
            output_dir: Directory to save cache files
            device: Device to use ("auto", "cuda", "cpu")
            confidence_threshold: Minimum confidence threshold for predictions
            iou_threshold: IoU threshold for matching predictions to ground truth
            config_path: Path to model config file (if None, auto-determined from checkpoint)
        """
        self.checkpoint_path = checkpoint_path
        self.coco_data_dir = Path(coco_data_dir)
        self.output_dir = Path(output_dir)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.config_path = config_path
        
        # Auto-detect device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.predictor = None
        
        print(f"Using device: {self.device}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"COCO data directory: {coco_data_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Confidence threshold: {confidence_threshold}")
        print(f"IoU matching threshold: {iou_threshold}")
        if config_path:
            print(f"Config path: {config_path}")
        print()
    
    def setup_model(self):
        """Setup the model from checkpoint."""
        print("Setting up model...")
        
        # Create config
        cfg = get_cfg()
        
        # Determine config path
        if self.config_path:
            config_path = self.config_path
        else:
            # Auto-determine config based on checkpoint name
            config_path = self._auto_determine_config_path()
        
        print(f"Using config: {config_path}")
        cfg.merge_from_file(config_path)
        
        # Set the checkpoint path
        cfg.MODEL.WEIGHTS = self.checkpoint_path
        
        # Set confidence threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
        
        # Set device
        cfg.MODEL.DEVICE = self.device
        
        # Create predictor
        self.predictor = DefaultPredictor(cfg)
        self.model = self.predictor.model
        
        print("Model setup completed")
        print(f"Model device: {next(self.model.parameters()).device}")

    def _auto_determine_config_path(self):
        """Auto-determine config path based on checkpoint filename."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        
        checkpoint_name = os.path.basename(self.checkpoint_path).lower()
        
        # Map common checkpoint names to config files
        config_mapping = {
            'faster_rcnn_r_50_fpn': 'faster_rcnn_R_50_FPN_3x.yaml',
            'faster_rcnn_r_101_fpn': 'faster_rcnn_R_101_FPN_3x.yaml',
            'faster_rcnn_x_101_32x8d_fpn': 'faster_rcnn_X_101_32x8d_FPN_3x.yaml',
            'mask_rcnn_r_50_fpn': 'mask_rcnn_R_50_FPN_3x.yaml',
            'mask_rcnn_r_101_fpn': 'mask_rcnn_R_101_FPN_3x.yaml',
            'retinanet_r_50_fpn': 'retinanet_R_50_FPN_3x.yaml',
            'retinanet_r_101_fpn': 'retinanet_R_101_FPN_3x.yaml',
            'sparse_rcnn_r101_300pro': 'faster_rcnn_X_101_32x8d_FPN_3x.yaml',  # Sparse R-CNN uses X101 FPN config
        }
        
        # Find matching config
        for key, config_file in config_mapping.items():
            if key in checkpoint_name:
                return os.path.join(project_root, "detectron2/configs/COCO-Detection", config_file)
        
        # Default fallback
        print(f"Warning: Could not auto-determine config for {checkpoint_name}, using R-50 FPN default")
        return os.path.join(project_root, "detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    
    def register_coco_datasets(self):
        """Register COCO datasets with detectron2."""
        print("Registering COCO datasets...")
        
        # Register train dataset
        train_json = self.coco_data_dir / "annotations" / "instances_train2017.json"
        train_images = self.coco_data_dir / "train2017"
        
        if train_json.exists() and train_images.exists():
            register_coco_instances("coco_train", {}, str(train_json), str(train_images))
            print(f"Registered train dataset: {len(DatasetCatalog.get('coco_train'))} images")
        else:
            print(f"Warning: Train dataset not found at {train_json} or {train_images}")
        
        # Register val dataset
        val_json = self.coco_data_dir / "annotations" / "instances_val2017.json"
        val_images = self.coco_data_dir / "val2017"
        
        if val_json.exists() and val_images.exists():
            register_coco_instances("coco_val", {}, str(val_json), str(val_images))
            print(f"Registered val dataset: {len(DatasetCatalog.get('coco_val'))} images")
        else:
            print(f"Warning: Val dataset not found at {val_json} or {val_images}")
    
    def run_inference_on_dataset(self, dataset_name: str, max_images: Optional[int] = None) -> Tuple[List[Dict], List[Dict]]:
        """
        Run inference on a dataset and extract predictions and labels.
        
        Args:
            dataset_name: Name of registered dataset
            max_images: Maximum number of images to process (for testing)
            
        Returns:
            predictions: List of prediction dictionaries
            labels: List of label dictionaries
        """
        print(f"Running inference on {dataset_name}...")
        
        # Get dataset
        dataset_dicts = get_detection_dataset_dicts([dataset_name])
        
        if max_images:
            dataset_dicts = dataset_dicts[:max_images]
        
        predictions = []
        labels = []
        
        # Process each image
        for idx, record in enumerate(tqdm(dataset_dicts, desc=f"Processing {dataset_name}")):
            try:
                # Load image
                image_path = record["file_name"]
                if not os.path.exists(image_path):
                    print(f"Warning: Image {image_path} not found, skipping...")
                    continue
                
                # Read image
                import cv2
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Could not read image {image_path}, skipping...")
                    continue
                
                # Run inference
                outputs = self.predictor(image)
                
                # Extract predictions
                instances = outputs["instances"]
                if len(instances) == 0:
                    continue
                
                # Get predictions on CPU
                pred_boxes = instances.pred_boxes.tensor.cpu().numpy()
                pred_classes = instances.pred_classes.cpu().numpy()
                pred_scores = instances.scores.cpu().numpy()
                
                # Create prediction dictionary
                pred_dict = {
                    'pred_coords': pred_boxes,  # [N, 4] - x0, y0, x1, y1
                    'pred_cls': pred_classes,   # [N] - class indices
                    'pred_score': pred_scores,  # [N] - confidence scores
                    'img_id': record.get('image_id', idx),
                    'height': record.get('height', image.shape[0]),
                    'width': record.get('width', image.shape[1])
                }
                
                # Extract ground truth
                gt_boxes = []
                gt_classes = []
                
                for annotation in record.get('annotations', []):
                    # Convert COCO bbox format [x, y, width, height] to [x0, y0, x1, y1]
                    bbox = annotation['bbox']
                    x0, y0, w, h = bbox
                    x1, y1 = x0 + w, y0 + h
                    
                    gt_boxes.append([x0, y0, x1, y1])
                    gt_classes.append(annotation['category_id'])
                
                # Create label dictionary
                label_dict = {
                    'gt_coords': np.array(gt_boxes) if gt_boxes else np.empty((0, 4)),
                    'gt_cls': np.array(gt_classes) if gt_classes else np.empty((0,)),
                    'img_id': record.get('image_id', idx),
                    'height': record.get('height', image.shape[0]),
                    'width': record.get('width', image.shape[1])
                }
                
                predictions.append(pred_dict)
                labels.append(label_dict)
                
            except Exception as e:
                print(f"Error processing image {idx}: {e}")
                continue
        
        print(f"Processed {len(predictions)} images from {dataset_name}")
        return predictions, labels
    
    def match_predictions_to_ground_truth(self, predictions: List[Dict], labels: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Match predictions to ground truth using IoU and create matched pairs.
        
        Args:
            predictions: List of prediction dictionaries
            labels: List of label dictionaries
            
        Returns:
            matched_predictions: List of matched prediction instances
            matched_labels: List of matched label instances
        """
        print("Matching predictions to ground truth...")
        
        def compute_iou(box1, box2):
            """Compute IoU between two boxes."""
            x1, y1, x2, y2 = box1
            x1_gt, y1_gt, x2_gt, y2_gt = box2
            
            # Intersection
            xi1, yi1 = max(x1, x1_gt), max(y1, y1_gt)
            xi2, yi2 = min(x2, x2_gt), min(y2, y2_gt)
            
            if xi2 <= xi1 or yi2 <= yi1:
                return 0
            
            inter_area = (xi2 - xi1) * (yi2 - yi1)
            
            # Union
            area1 = (x2 - x1) * (y2 - y1)
            area2 = (x2_gt - x1_gt) * (y2_gt - y1_gt)
            union_area = area1 + area2 - inter_area
            
            return inter_area / union_area if union_area > 0 else 0
        
        matched_predictions = []
        matched_labels = []
        
        for pred_dict, label_dict in zip(predictions, labels):
            if pred_dict['img_id'] != label_dict['img_id']:
                continue
            
            pred_boxes = pred_dict['pred_coords']
            gt_boxes = label_dict['gt_coords']
            
            if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                continue
            
            # For each prediction, find best matching ground truth
            for i, pred_box in enumerate(pred_boxes):
                best_iou = 0
                best_gt_idx = -1
                
                for j, gt_box in enumerate(gt_boxes):
                    iou = compute_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
                
                # If IoU is above threshold, create matched pair
                if best_iou > self.iou_threshold:
                    matched_pred = {
                        'pred_coords': pred_box,
                        'pred_cls': pred_dict['pred_cls'][i],
                        'pred_score': pred_dict['pred_score'][i],
                        'img_id': pred_dict['img_id'],
                        'height': pred_dict['height'],
                        'width': pred_dict['width']
                    }
                    
                    matched_label = {
                        'gt_coords': gt_boxes[best_gt_idx],
                        'gt_cls': label_dict['gt_cls'][best_gt_idx],
                        'img_id': label_dict['img_id'],
                        'height': label_dict['height'],
                        'width': label_dict['width'],
                        'iou': best_iou
                    }
                    
                    matched_predictions.append(matched_pred)
                    matched_labels.append(matched_label)
        
        total_predictions = sum(len(pred['pred_coords']) for pred in predictions)
        total_gt_boxes = sum(len(label['gt_coords']) for label in labels)
        
        print(f"Total predictions across all images: {total_predictions}")
        print(f"Total ground truth boxes: {total_gt_boxes}")
        print(f"Created {len(matched_predictions)} matched prediction-label pairs")
        print(f"Matching rate: {len(matched_predictions)/total_predictions*100:.1f}% of predictions matched")
        return matched_predictions, matched_labels
    
    def extract_features(self, predictions: List[Dict]) -> torch.Tensor:
        """
        Extract 17-dimensional features from predictions (compatible with trained models).
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            features: [N, 17] tensor of extracted features (13 geometric + 4 uncertainty)
        """
        print("Extracting features from predictions...")
        
        if not predictions:
            return torch.empty(0, 17)
        
        n_predictions = len(predictions)
        features = torch.zeros(n_predictions, 17, dtype=torch.float32)
        
        for i, pred in enumerate(predictions):
            box = pred['pred_coords']
            score = pred['pred_score']
            img_h = pred['height']
            img_w = pred['width']
            
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            
            # Extract 13 features matching feature_utils.py
            # 1-4: Raw coordinates
            features[i, 0] = float(x1)                                     # x0
            features[i, 1] = float(y1)                                     # y0
            features[i, 2] = float(x2)                                     # x1
            features[i, 3] = float(y2)                                     # y1
            
            # 5: Confidence score
            features[i, 4] = float(score)                                  # confidence
            
            # 6: Log area
            area = float(w * h)
            features[i, 5] = float(np.log(max(area, 1e-6)))               # log_area
            
            # 7: Aspect ratio
            features[i, 6] = float(w / (h + 1e-6))                         # aspect_ratio
            
            # 8-9: Normalized center coordinates
            features[i, 7] = float((x1 + x2) / 2 / img_w)                 # center_x_norm
            features[i, 8] = float((y1 + y2) / 2 / img_h)                 # center_y_norm
            
            # 10-11: Position relative to image center
            features[i, 9] = float(((x1 + x2) / 2 - img_w / 2) / img_w)   # rel_pos_x
            features[i, 10] = float(((y1 + y2) / 2 - img_h / 2) / img_h)  # rel_pos_y
            
            # 12: Relative size
            features[i, 11] = float(area / (img_w * img_h))                # rel_size
            
            # 13: Distance to nearest edge (minimum of all 4 edges)
            dist_left = float(x1 / img_w)
            dist_right = float((img_w - x2) / img_w)
            dist_top = float(y1 / img_h)
            dist_bottom = float((img_h - y2) / img_h)
            features[i, 12] = float(min(dist_left, dist_right, dist_top, dist_bottom))  # edge_distance
            
            # 14-17: Uncertainty features (matching UncertaintyFeatureExtractor)
            # 14: Confidence-based uncertainty
            features[i, 13] = 1.0 - float(score)                           # uncertainty_score
            
            # 15: Ensemble uncertainty proxy (scaled confidence-based)
            features[i, 14] = (1.0 - float(score)) * 10.0                  # scaled uncertainty
            
            # 16: Expected error proxy (scaled by typical error magnitude)
            features[i, 15] = (1.0 - float(score)) * 50.0                  # expected_error
            
            # 17: Difficulty score (area difficulty + aspect ratio difficulty)
            area_difficulty = float(1.0 / (area + 1.0))
            aspect_difficulty = float(abs(np.log(float(w) / (float(h) + 1e-6) + 1e-6)))
            features[i, 16] = float((area_difficulty + aspect_difficulty) / 2.0)  # difficulty_score
        
        print(f"Extracted features shape: {features.shape}")
        return features
    
    def create_rich_prediction_format(self, predictions: List[Dict], labels: List[Dict]):
        """
        Create rich prediction format with detailed geometric features and residuals.
        
        Args:
            predictions: List of prediction dictionaries
            labels: List of label dictionaries
            
        Returns:
            rich_predictions: List of lists with detailed prediction data
            rich_labels: List of dictionaries with detailed label data
        """
        print("Creating rich prediction format with detailed features...")
        
        # Group data by image_id to create the list-of-lists structure observed in cache_base_model_copy
        image_groups = defaultdict(list)
        for pred, label in zip(predictions, labels):
            img_id = pred['img_id']
            
            # Calculate detailed geometric features
            gt_x0, gt_y0, gt_x1, gt_y1 = label['gt_coords']
            pred_x0, pred_y0, pred_x1, pred_y1 = pred['pred_coords']
            
            # Centers
            gt_center_x = (gt_x0 + gt_x1) / 2.0
            gt_center_y = (gt_y0 + gt_y1) / 2.0
            pred_center_x = (pred_x0 + pred_x1) / 2.0
            pred_center_y = (pred_y0 + pred_y1) / 2.0
            
            gt_centers = [gt_center_x, gt_center_y]
            pred_centers = [pred_center_x, pred_center_y]
            
            # Areas
            gt_area = (gt_x1 - gt_x0) * (gt_y1 - gt_y0)
            pred_area = (pred_x1 - pred_x0) * (pred_y1 - pred_y0)
            
            # Residuals
            abs_res_x0 = abs(pred_x0 - gt_x0)
            abs_res_y0 = abs(pred_y0 - gt_y0)
            abs_res_x1 = abs(pred_x1 - gt_x1)
            abs_res_y1 = abs(pred_y1 - gt_y1)
            
            one_sided_res_x0 = pred_x0 - gt_x0
            one_sided_res_y0 = pred_y0 - gt_y0
            one_sided_res_x1 = pred_x1 - gt_x1
            one_sided_res_y1 = pred_y1 - gt_y1
            
            # Create rich data entry
            rich_entry = {
                'gt_x0': float(gt_x0),
                'gt_y0': float(gt_y0),
                'gt_x1': float(gt_x1),
                'gt_y1': float(gt_y1),
                'pred_x0': float(pred_x0),
                'pred_y0': float(pred_y0),
                'pred_x1': float(pred_x1),
                'pred_y1': float(pred_y1),
                'gt_centers': gt_centers,
                'pred_centers': pred_centers,
                'gt_area': float(gt_area),
                'pred_area': float(pred_area),
                'pred_score': float(pred['pred_score']),
                'pred_score_all': [float(pred['pred_score'])] * 5,  # Simulated all scores
                'pred_logits_all': [float(pred['pred_score']) * 2 - 1] * 5,  # Simulated logits
                'iou': float(label['iou']),
                'img_id': int(img_id),
                'label_score': 1.0,  # Ground truth label score
                'abs_res_x0': float(abs_res_x0),
                'abs_res_y0': float(abs_res_y0),
                'abs_res_x1': float(abs_res_x1),
                'abs_res_y1': float(abs_res_y1),
                'one_sided_res_x0': float(one_sided_res_x0),
                'one_sided_res_y0': float(one_sided_res_y0),
                'one_sided_res_x1': float(one_sided_res_x1),
                'one_sided_res_y1': float(one_sided_res_y1)
            }
            
            image_groups[img_id].append(rich_entry)
        
        # Create the expected format: list of 80 items (for each image)
        # Take a sample of unique image IDs for efficient processing
        unique_img_ids = list(image_groups.keys())[:80]  # Limit to 80 images for compatibility
        
        rich_predictions = []
        rich_labels = []
        
        for img_id in unique_img_ids:
            img_entries = image_groups[img_id]
            
            # Create list of prediction data for this image
            img_pred_list = []
            img_label_list = []
            
            for entry in img_entries:
                img_pred_list.append([
                    entry['pred_x0'], entry['pred_y0'], entry['pred_x1'], entry['pred_y1'],
                    entry['pred_score']
                ])
                img_label_list.append(entry)
            
            rich_predictions.append(img_pred_list)
            rich_labels.append(img_entries[0] if img_entries else {})  # Use first entry as representative
        
        print(f"Created rich format with {len(rich_predictions)} image groups")
        return rich_predictions, rich_labels
    
    def save_cache(self, train_predictions: List[Dict], train_labels: List[Dict],
                   val_predictions: List[Dict], val_labels: List[Dict],
                   train_features: torch.Tensor, val_features: torch.Tensor):
        """
        Save cache with rich prediction format and detailed geometric features.
        
        Args:
            train_predictions: Training predictions
            train_labels: Training labels
            val_predictions: Validation predictions
            val_labels: Validation labels
            train_features: Training features
            val_features: Validation features
        """
        print("Saving cache files with rich prediction format...")
        
        # Create rich prediction data with detailed geometric features and residuals
        rich_train_data, rich_train_labels = self.create_rich_prediction_format(train_predictions, train_labels)
        rich_val_data, rich_val_labels = self.create_rich_prediction_format(val_predictions, val_labels)
        
        # Save predictions as pickle files in tuple format
        with open(self.output_dir / "predictions_train.pkl", 'wb') as f:
            pickle.dump((rich_train_data, rich_train_labels), f)
        
        with open(self.output_dir / "predictions_val.pkl", 'wb') as f:
            pickle.dump((rich_val_data, rich_val_labels), f)
        
        # Prepare tensor data for .pt files with img_ids
        train_img_ids = torch.tensor([p['img_id'] for p in train_predictions], dtype=torch.int64)
        val_img_ids = torch.tensor([p['img_id'] for p in val_predictions], dtype=torch.int64)
        
        train_data = {
            'features': train_features,
            'gt_coords': torch.tensor([l['gt_coords'] for l in train_labels], dtype=torch.float32),
            'pred_coords': torch.tensor([p['pred_coords'] for p in train_predictions], dtype=torch.float32),
            'confidence': torch.tensor([p['pred_score'] for p in train_predictions], dtype=torch.float32),
            'img_ids': train_img_ids
        }
        
        # Create calibration/test splits for validation data
        val_size = len(val_predictions)
        calib_size = val_size // 2
        calib_indices = torch.arange(calib_size, dtype=torch.int64)
        test_indices = torch.arange(calib_size, val_size, dtype=torch.int64)
        
        val_data = {
            'features': val_features,
            'gt_coords': torch.tensor([l['gt_coords'] for l in val_labels], dtype=torch.float32),
            'pred_coords': torch.tensor([p['pred_coords'] for p in val_predictions], dtype=torch.float32),
            'confidence': torch.tensor([p['pred_score'] for p in val_predictions], dtype=torch.float32),
            'img_ids': val_img_ids,
            'calib_indices': calib_indices,
            'test_indices': test_indices
        }
        
        # Save .pt files
        torch.save(train_data, self.output_dir / "features_train.pt")
        torch.save(val_data, self.output_dir / "features_val.pt")
        
        # Save subset versions as dictionaries for compatibility
        train_subset_size = min(50000, len(train_predictions))
        val_subset_size = min(20000, len(val_predictions))
        
        train_subset_data = {
            'features': train_features[:train_subset_size],
            'gt_coords': train_data['gt_coords'][:train_subset_size],
            'pred_coords': train_data['pred_coords'][:train_subset_size],
            'confidence': train_data['confidence'][:train_subset_size]
        }
        
        val_subset_data = {
            'features': val_features[:val_subset_size],
            'gt_coords': val_data['gt_coords'][:val_subset_size],
            'pred_coords': val_data['pred_coords'][:val_subset_size],
            'confidence': val_data['confidence'][:val_subset_size]
        }
        
        torch.save(train_subset_data, self.output_dir / "features_train_no_img_ids.pt")
        torch.save(val_subset_data, self.output_dir / "features_val_no_img_ids.pt")
        
        print("Cache files saved successfully!")
        print(f"Train samples: {len(train_predictions)}")
        print(f"Val samples: {len(val_predictions)}")
        print(f"Feature dimension: {train_features.shape[1] if len(train_features) > 0 else 0}")
        print(f"Train subset size: {train_subset_size}")
        print(f"Val subset size: {val_subset_size}")
        print(f"Calibration set size: {len(calib_indices)}")
        print(f"Test set size: {len(test_indices)}")
        
        # Print file sizes
        for file_path in self.output_dir.iterdir():
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"  {file_path.name}: {size_mb:.1f} MB")
    
    def generate_cache(self, max_train_images: Optional[int] = None, max_val_images: Optional[int] = None):
        """
        Generate complete cache from checkpoint.
        
        Args:
            max_train_images: Maximum training images to process (for testing)
            max_val_images: Maximum validation images to process (for testing)
        """
        print("="*80)
        print("GENERATING CACHE FROM FASTER R-CNN CHECKPOINT")
        print("="*80)
        
        # Setup model
        self.setup_model()
        
        # Register datasets
        self.register_coco_datasets()
        
        # Process training data
        print("\n" + "="*50)
        print("PROCESSING TRAINING DATA")
        print("="*50)
        
        train_predictions, train_labels = self.run_inference_on_dataset("coco_train", max_train_images)
        train_matched_preds, train_matched_labels = self.match_predictions_to_ground_truth(train_predictions, train_labels)
        train_features = self.extract_features(train_matched_preds)
        
        # Process validation data
        print("\n" + "="*50)
        print("PROCESSING VALIDATION DATA")
        print("="*50)
        
        val_predictions, val_labels = self.run_inference_on_dataset("coco_val", max_val_images)
        val_matched_preds, val_matched_labels = self.match_predictions_to_ground_truth(val_predictions, val_labels)
        val_features = self.extract_features(val_matched_preds)
        
        # Save cache
        print("\n" + "="*50)
        print("SAVING CACHE")
        print("="*50)
        
        self.save_cache(
            train_matched_preds, train_matched_labels,
            val_matched_preds, val_matched_labels,
            train_features, val_features
        )
        
        print("\n" + "="*80)
        print("CACHE GENERATION COMPLETED!")
        print("="*80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate cache from Faster R-CNN checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to Faster R-CNN checkpoint")
    parser.add_argument("--coco-dir", type=str, required=True,
                        help="Path to COCO dataset directory")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for cache files")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--confidence-threshold", type=float, default=0.5,
                        help="Minimum confidence threshold for predictions (default: 0.5 for optimal performance)")
    parser.add_argument("--max-train-images", type=int, default=None,
                        help="Maximum training images to process (for testing)")
    parser.add_argument("--max-val-images", type=int, default=None,
                        help="Maximum validation images to process (for testing)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file {args.checkpoint} does not exist")
        return 1
    
    if not os.path.exists(args.coco_dir):
        print(f"Error: COCO directory {args.coco_dir} does not exist")
        return 1
    
    # Create cache generator
    generator = CacheGenerator(
        checkpoint_path=args.checkpoint,
        coco_data_dir=args.coco_dir,
        output_dir=args.output_dir,
        device=args.device,
        confidence_threshold=args.confidence_threshold
    )
    
    # Generate cache
    try:
        generator.generate_cache(
            max_train_images=args.max_train_images,
            max_val_images=args.max_val_images
        )
        return 0
    except Exception as e:
        print(f"Error generating cache: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 