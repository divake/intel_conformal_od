#!/usr/bin/env python3
"""
Generate cache model from YOLOv3/Gaussian YOLOv3 checkpoint.

This script:
1. Loads the YOLOv3 model from checkpoint
2. Runs inference on COCO dataset (train and validation)
3. Extracts features using FeatureExtractor
4. Saves cache in the expected format for learnable scoring function training

Usage:
    python generate_cache_yolo.py --checkpoint /path/to/gaussian_yolov3_coco.pth --config /path/to/gaussian_yolov3_eval.yaml
"""

import os
import sys
import argparse
import pickle
import torch
import numpy as np
import cv2
import yaml
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

# Import YOLOv3 components
try:
    from gaussian_yolo.yolov3 import YOLOv3
    from gaussian_yolo import utils as yolo_utils
    print("YOLOv3 imported successfully")
except ImportError as e:
    print(f"Error importing YOLOv3: {e}")
    print("Please check that YOLOv3 components are available")
    sys.exit(1)

# Import detectron2 for dataset handling
try:
    from detectron2.data.datasets import register_coco_instances
    from detectron2.data import MetadataCatalog, DatasetCatalog
    from detectron2.data.build import get_detection_dataset_dicts
    from detectron2.structures import Boxes, Instances
    from detectron2.structures.instances import Instances as D2Instances
    print("Detectron2 dataset utilities imported successfully")
except ImportError as e:
    print(f"Error importing detectron2: {e}")
    print("Please check that detectron2 is properly installed")
    sys.exit(1)

# Import feature extractor from learnable_scoring_fn
try:
    from learnable_scoring_fn.feature_utils import FeatureExtractor
    print("Feature extractor imported successfully")
except ImportError as e:
    print(f"Error importing feature extractor: {e}")
    print("Make sure you're running from the correct directory")
    sys.exit(1)


def annotations_to_instances(annotations, image_size):
    """Convert annotations to detectron2 Instances format."""
    target = D2Instances(image_size)
    
    boxes = []
    classes = []
    
    for ann in annotations:
        if 'bbox' in ann:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            classes.append(ann['category_id'])
    
    if boxes:
        target.gt_boxes = Boxes(torch.tensor(boxes, dtype=torch.float32))
        target.gt_classes = torch.tensor(classes, dtype=torch.int64)
    else:
        target.gt_boxes = Boxes(torch.zeros((0, 4), dtype=torch.float32))
        target.gt_classes = torch.tensor([], dtype=torch.int64)
    
    return target


class YOLOCacheGenerator:
    """Generate cache from YOLOv3/Gaussian YOLOv3 checkpoint for learnable scoring function."""
    
    def __init__(self, checkpoint_path: str, config_path: str, coco_data_dir: str, output_dir: str,
                 device: str = "auto", confidence_threshold: float = 0.1,
                 iou_threshold: float = 0.3, nms_threshold: float = 0.6, imgsize: int = 416):
        """
        Initialize cache generator.
        
        Args:
            checkpoint_path: Path to YOLOv3 model checkpoint
            config_path: Path to YOLOv3 config file (YAML)
            coco_data_dir: Path to COCO dataset directory
            output_dir: Directory to save cache files
            device: Device to use ("auto", "cuda", "cpu")
            confidence_threshold: Minimum confidence threshold for predictions
            iou_threshold: IoU threshold for matching predictions to ground truth
            nms_threshold: NMS threshold for YOLOv3 postprocessing
            imgsize: Input image size for YOLOv3
        """
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.coco_data_dir = Path(coco_data_dir)
        self.output_dir = Path(output_dir)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.nms_threshold = nms_threshold
        self.imgsize = imgsize
        
        # Auto-detect device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = None
        self.nr_class = 80  # COCO classes
        
        print(f"Using device: {self.device}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Config: {config_path}")
        print(f"COCO data directory: {coco_data_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Confidence threshold: {confidence_threshold}")
        print(f"IoU matching threshold: {iou_threshold}")
        print(f"NMS threshold: {nms_threshold}")
        print(f"Image size: {imgsize}")
        print()
    
    def setup_model(self):
        """Setup the YOLOv3 model from checkpoint."""
        print("Setting up YOLOv3 model...")
        
        # Load config
        with open(self.config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        model_config = cfg['MODEL']
        
        # Create model
        self.model = YOLOv3(model_config)
        
        # Load checkpoint
        print(f"Loading checkpoint {self.checkpoint_path}")
        state = torch.load(self.checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in state.keys():
            self.model.load_state_dict(state['model_state_dict'])
        else:
            self.model.load_state_dict(state)
        
        self.model.eval()
        
        if self.device == "cuda":
            self.model.cuda()
        
        print("YOLOv3 model setup completed")
        print(f"Model device: {next(self.model.parameters()).device}")
    
    def register_coco_datasets(self):
        """Register COCO datasets with detectron2."""
        print("Registering COCO datasets...")
        
        # Register training dataset
        train_json = self.coco_data_dir / "annotations/instances_train2017.json"
        train_images = self.coco_data_dir / "train2017"
        
        if "coco_train" not in DatasetCatalog:
            register_coco_instances("coco_train", {}, str(train_json), str(train_images))
        
        train_metadata = MetadataCatalog.get("coco_train")
        print(f"Registered train dataset: {len(DatasetCatalog.get('coco_train'))} images")
        
        # Register validation dataset
        val_json = self.coco_data_dir / "annotations/instances_val2017.json"
        val_images = self.coco_data_dir / "val2017"
        
        if "coco_val" not in DatasetCatalog:
            register_coco_instances("coco_val", {}, str(val_json), str(val_images))
        
        val_metadata = MetadataCatalog.get("coco_val")
        print(f"Registered val dataset: {len(DatasetCatalog.get('coco_val'))} images")
        print()
    
    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, Tuple]:
        """Preprocess image for YOLOv3 inference."""
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return None, None
        
        # Preprocess for YOLO
        img, info_img = yolo_utils.preprocess(img, self.imgsize, jitter=0)
        img = np.transpose(img / 255.0, (2, 0, 1))
        img = torch.from_numpy(img).float().unsqueeze(0)
        
        if self.device == "cuda":
            img = img.cuda()
        
        return img, info_img
    
    def run_inference_on_dataset(self, dataset_name: str, max_images: Optional[int] = None) -> Tuple[List[Dict], List[Dict]]:
        """
        Run inference on a dataset and collect predictions and ground truth.
        
        Args:
            dataset_name: Name of registered dataset
            max_images: Maximum number of images to process
            
        Returns:
            Tuple of (predictions, labels) lists
        """
        print(f"Running inference on {dataset_name}...")
        
        # Get dataset
        dataset_dicts = get_detection_dataset_dicts(dataset_name)
        
        if max_images is not None:
            dataset_dicts = dataset_dicts[:max_images]
        
        predictions = []
        labels = []
        processed_count = 0
        
        # Process each image
        for idx, record in enumerate(tqdm(dataset_dicts, desc=f"Processing {dataset_name}")):
            try:
                # Load and preprocess image
                image_path = record["file_name"]
                if not os.path.exists(image_path):
                    continue
                
                img_tensor, info_img = self.preprocess_image(image_path)
                if img_tensor is None:
                    continue
                
                # Run YOLOv3 inference
                with torch.no_grad():
                    outputs = self.model(img_tensor)
                    outputs = yolo_utils.postprocess(
                        outputs, self.nr_class, self.confidence_threshold, self.nms_threshold
                    )
                
                # Process YOLOv3 outputs
                if outputs[0] is None:
                    continue
                
                # Extract predictions
                yolo_preds = outputs[0]
                num_preds = len(yolo_preds)
                
                pred_boxes = torch.zeros((num_preds, 4))
                pred_classes = torch.zeros((num_preds,))
                pred_scores = torch.zeros((num_preds,))
                
                for i, pred in enumerate(yolo_preds):
                    x1, y1, x2, y2, obj_conf, cls_conf, cls_pred = pred[:7]
                    
                    # Convert to image coordinates
                    box = yolo_utils.yolobox2label([y1, x1, y2, x2], info_img)
                    pred_boxes[i] = torch.tensor(box)
                    pred_classes[i] = cls_pred
                    pred_scores[i] = obj_conf * cls_conf
                
                # Create prediction dictionary
                pred_dict = {
                    'pred_coords': pred_boxes.numpy(),
                    'pred_cls': pred_classes.numpy().astype(int),
                    'pred_score': pred_scores.numpy(),
                    'img_id': record.get('image_id', idx),
                    'height': record.get('height', info_img[0]),
                    'width': record.get('width', info_img[1])
                }
                
                # Extract ground truth using detectron2 format
                gt = annotations_to_instances(record["annotations"], (record["height"], record["width"]))
                gt_boxes = gt.gt_boxes.tensor.numpy()
                gt_classes = gt.gt_classes.numpy()
                
                gt_dict = {
                    'gt_coords': gt_boxes,
                    'gt_cls': gt_classes,
                    'img_id': record.get('image_id', idx),
                    'height': record.get('height', info_img[0]),
                    'width': record.get('width', info_img[1])
                }
                
                predictions.append(pred_dict)
                labels.append(gt_dict)
                processed_count += 1
                
            except Exception as e:
                print(f"Warning: Error processing image {idx}: {e}")
                continue
        
        print(f"Processed {processed_count} images from {dataset_name}")
        return predictions, labels
    
    def match_predictions_to_ground_truth(self, predictions: List[Dict], labels: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Match predictions to ground truth using IoU threshold."""
        print("Matching predictions to ground truth...")
        
        matched_predictions = []
        matched_labels = []
        total_predictions = 0
        total_gt = 0
        
        for pred_dict, label_dict in zip(predictions, labels):
            pred_boxes = pred_dict['pred_coords']
            pred_classes = pred_dict['pred_cls']
            pred_scores = pred_dict['pred_score']
            
            gt_boxes = label_dict['gt_coords']
            gt_classes = label_dict['gt_cls']
            
            total_predictions += len(pred_boxes)
            total_gt += len(gt_boxes)
            
            if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                continue
            
            # Convert to torch tensors for IoU computation
            pred_boxes_tensor = torch.tensor(pred_boxes, dtype=torch.float32)
            gt_boxes_tensor = torch.tensor(gt_boxes, dtype=torch.float32)
            
            # Compute IoU matrix
            from detectron2.structures import pairwise_iou
            pred_boxes_struct = Boxes(pred_boxes_tensor)
            gt_boxes_struct = Boxes(gt_boxes_tensor)
            iou_matrix = pairwise_iou(pred_boxes_struct, gt_boxes_struct)
            
            # Find matches
            for i, pred_box in enumerate(pred_boxes):
                pred_class = pred_classes[i]
                pred_score = pred_scores[i]
                
                # Find best matching GT box
                ious = iou_matrix[i]
                best_gt_idx = torch.argmax(ious)
                best_iou = ious[best_gt_idx].item()
                
                if best_iou >= self.iou_threshold:
                    gt_class = gt_classes[best_gt_idx]
                    gt_box = gt_boxes[best_gt_idx]
                    
                    # Class matching (optional - can be disabled)
                    if pred_class == gt_class:
                        # Create matched pair
                        matched_pred = {
                            'pred_coords': pred_box,
                            'pred_cls': pred_class,
                            'pred_score': pred_score,
                            'img_id': pred_dict['img_id'],
                            'height': pred_dict['height'],
                            'width': pred_dict['width']
                        }
                        
                        matched_label = {
                            'gt_coords': gt_box,
                            'gt_cls': gt_class,
                            'img_id': label_dict['img_id'],
                            'height': label_dict['height'],
                            'width': label_dict['width']
                        }
                        
                        matched_predictions.append(matched_pred)
                        matched_labels.append(matched_label)
        
        print(f"Total predictions across all images: {total_predictions}")
        print(f"Total ground truth boxes: {total_gt}")
        print(f"Created {len(matched_predictions)} matched prediction-label pairs")
        print(f"Matching rate: {len(matched_predictions)/total_predictions*100:.1f}% of predictions matched")
        
        return matched_predictions, matched_labels
    
    def extract_features(self, predictions: List[Dict]) -> torch.Tensor:
        """Extract features from matched predictions using FeatureExtractor."""
        print("Extracting features from predictions...")
        
        if not predictions:
            return torch.zeros((0, 17))  # Empty tensor with correct feature dimension
        
        # Initialize feature extractor
        feature_extractor = FeatureExtractor()
        
        # Create tensors from predictions
        num_preds = len(predictions)
        features = torch.zeros((num_preds, 17))  # Standard feature dimension
        
        for i, pred in enumerate(predictions):
            box = pred['pred_coords']
            score = pred['pred_score']
            img_h = pred['height']
            img_w = pred['width']
            
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            
            # Extract 17 features matching feature_utils.py
            # 1-4: Raw coordinates
            features[i, 0] = float(x1)
            features[i, 1] = float(y1)
            features[i, 2] = float(x2)
            features[i, 3] = float(y2)
            
            # 5: Confidence score
            features[i, 4] = float(score)
            
            # 6: Log area
            area = float(w * h)
            features[i, 5] = float(np.log(max(area, 1e-6)))
            
            # 7: Aspect ratio
            features[i, 6] = float(w / (h + 1e-6))
            
            # 8-9: Normalized center coordinates
            features[i, 7] = float((x1 + x2) / 2 / img_w)
            features[i, 8] = float((y1 + y2) / 2 / img_h)
            
            # 10-11: Position relative to image center
            features[i, 9] = float(((x1 + x2) / 2 - img_w / 2) / img_w)
            features[i, 10] = float(((y1 + y2) / 2 - img_h / 2) / img_h)
            
            # 12: Relative size
            features[i, 11] = float(area / (img_w * img_h))
            
            # 13: Distance to nearest edge
            dist_left = float(x1 / img_w)
            dist_right = float((img_w - x2) / img_w)
            dist_top = float(y1 / img_h)
            dist_bottom = float((img_h - y2) / img_h)
            features[i, 12] = min(dist_left, dist_right, dist_top, dist_bottom)
            
            # 14-17: Additional geometric features
            features[i, 13] = float(x1 / img_w)  # Left edge position
            features[i, 14] = float(y1 / img_h)  # Top edge position
            features[i, 15] = float(w / img_w)   # Normalized width
            features[i, 16] = float(h / img_h)   # Normalized height
        
        print(f"Extracted features shape: {features.shape}")
        return features
    
    def create_rich_prediction_format(self, predictions: List[Dict], labels: List[Dict]) -> Tuple[List, List]:
        """Create rich prediction format compatible with learnable scoring function."""
        print("Creating rich prediction format with detailed features...")
        
        # Group by image ID
        img_groups = defaultdict(lambda: {'predictions': [], 'labels': []})
        
        for pred, label in zip(predictions, labels):
            img_id = pred['img_id']
            img_groups[img_id]['predictions'].append(pred)
            img_groups[img_id]['labels'].append(label)
        
        rich_predictions = []
        rich_labels = []
        
        for img_id, group in img_groups.items():
            if group['predictions'] and group['labels']:
                rich_predictions.append(group['predictions'])
                rich_labels.append(group['labels'])
        
        print(f"Created rich format with {len(rich_predictions)} image groups")
        return rich_predictions, rich_labels
    
    def save_cache(self, train_predictions: List[Dict], train_labels: List[Dict],
                   val_predictions: List[Dict], val_labels: List[Dict],
                   train_features: torch.Tensor, val_features: torch.Tensor):
        """Save cache files in the expected format."""
        print("Saving cache files with rich prediction format...")
        
        # Create rich prediction data
        rich_train_data, rich_train_labels = self.create_rich_prediction_format(train_predictions, train_labels)
        rich_val_data, rich_val_labels = self.create_rich_prediction_format(val_predictions, val_labels)
        
        # Save predictions as pickle files
        with open(self.output_dir / "predictions_train.pkl", 'wb') as f:
            pickle.dump((rich_train_data, rich_train_labels), f)
        
        with open(self.output_dir / "predictions_val.pkl", 'wb') as f:
            pickle.dump((rich_val_data, rich_val_labels), f)
        
        # Prepare tensor data for .pt files
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
        
        # Save subset versions without img_ids for compatibility
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
        """Generate complete cache from YOLOv3 checkpoint."""
        print("="*80)
        print("GENERATING CACHE FROM YOLOV3 CHECKPOINT")
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
    parser = argparse.ArgumentParser(description="Generate cache from YOLOv3 checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to YOLOv3 checkpoint")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YOLOv3 config file (YAML)")
    parser.add_argument("--coco-dir", type=str, required=True,
                        help="Path to COCO dataset directory")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for cache files")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--confidence-threshold", type=float, default=0.5,
                        help="Minimum confidence threshold for predictions")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IoU threshold for matching predictions to ground truth")
    parser.add_argument("--nms-threshold", type=float, default=0.6,
                        help="NMS threshold for YOLOv3 postprocessing")
    parser.add_argument("--imgsize", type=int, default=416,
                        help="Input image size for YOLOv3")
    parser.add_argument("--max-train-images", type=int, default=None,
                        help="Maximum training images to process (for testing)")
    parser.add_argument("--max-val-images", type=int, default=None,
                        help="Maximum validation images to process (for testing)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file {args.checkpoint} does not exist")
        return 1
    
    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} does not exist")
        return 1
    
    if not os.path.exists(args.coco_dir):
        print(f"Error: COCO directory {args.coco_dir} does not exist")
        return 1
    
    # Create cache generator
    generator = YOLOCacheGenerator(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        coco_data_dir=args.coco_dir,
        output_dir=args.output_dir,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
        nms_threshold=args.nms_threshold,
        imgsize=args.imgsize
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