import os
import torch
import json
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Optional
from pathlib import Path


def get_coco_class_frequencies(coco_train_path: str) -> Dict[int, int]:
    """
    Analyze COCO training dataset to get class frequencies.
    
    Args:
        coco_train_path: Path to COCO training annotations
        
    Returns:
        class_frequencies: Dictionary mapping class_id to frequency count
    """
    # Load COCO annotations
    with open(os.path.join(coco_train_path, "annotations", "instances_train2017.json"), 'r') as f:
        coco_data = json.load(f)
    
    # Count class frequencies
    class_counts = Counter()
    for annotation in coco_data['annotations']:
        class_counts[annotation['category_id']] += 1
    
    return dict(class_counts)


def get_top_classes(class_frequencies: Dict[int, int], top_k: int = 6) -> List[int]:
    """
    Get the top K most frequent classes.
    
    Args:
        class_frequencies: Dictionary mapping class_id to frequency
        top_k: Number of top classes to return
        
    Returns:
        top_classes: List of top K class IDs sorted by frequency (descending)
    """
    sorted_classes = sorted(class_frequencies.items(), key=lambda x: x[1], reverse=True)
    return [class_id for class_id, _ in sorted_classes[:top_k]]


def create_stratified_subset(ist_list: List[Dict], img_list: List[List], 
                           top_classes: List[int], subset_size: int = 50000,
                           min_samples_per_class: int = 1000) -> Tuple[List[Dict], List[List]]:
    """
    Create a stratified subset of the data focusing on top classes.
    
    Args:
        ist_list: List of instance dictionaries per class
        img_list: List of image lists per class  
        top_classes: List of class IDs to focus on
        subset_size: Target total number of samples
        min_samples_per_class: Minimum samples to ensure per class
        
    Returns:
        subset_ist_list: Subset of instance data
        subset_img_list: Subset of image data
    """
    # Calculate total available samples for top classes
    total_available = sum(len(ist_list[c]['gt_x0']) for c in top_classes if c < len(ist_list))
    
    if total_available < subset_size:
        subset_size = total_available
        print(f"Warning: Only {total_available} samples available, using all of them")
    
    # Calculate samples per class (proportional to class frequency, with minimum)
    class_samples = {}
    remaining_budget = subset_size
    
    # First, allocate minimum samples per class
    for class_id in top_classes:
        if class_id < len(ist_list):
            available_samples = len(ist_list[class_id]['gt_x0'])
            allocated = min(min_samples_per_class, available_samples)
            class_samples[class_id] = allocated
            remaining_budget -= allocated
    
    # Then, distribute remaining budget proportionally
    if remaining_budget > 0:
        # Get proportional weights based on available data
        class_weights = {}
        total_weight = 0
        for class_id in top_classes:
            if class_id < len(ist_list):
                available = len(ist_list[class_id]['gt_x0']) - class_samples.get(class_id, 0)
                if available > 0:
                    class_weights[class_id] = available
                    total_weight += available
        
        # Distribute remaining budget
        for class_id, weight in class_weights.items():
            if total_weight > 0:
                additional = int((weight / total_weight) * remaining_budget)
                class_samples[class_id] += additional
    
    print(f"Stratified sampling plan:")
    for class_id, count in class_samples.items():
        total_available = len(ist_list[class_id]['gt_x0']) if class_id < len(ist_list) else 0
        print(f"  Class {class_id}: {count} samples (from {total_available} available)")
    
    # Create subset by sampling from each class
    subset_ist_list = [{'gt_x0': [], 'gt_y0': [], 'gt_x1': [], 'gt_y1': [],
                       'pred_x0': [], 'pred_y0': [], 'pred_x1': [], 'pred_y1': [],
                       'pred_score': [], 'img_id': []} for _ in range(len(ist_list))]
    subset_img_list = [[] for _ in range(len(img_list))]
    
    for class_id, target_count in class_samples.items():
        if class_id >= len(ist_list) or len(ist_list[class_id]['gt_x0']) == 0:
            continue
            
        # Get all indices for this class
        total_samples = len(ist_list[class_id]['gt_x0'])
        indices = np.random.choice(total_samples, size=min(target_count, total_samples), replace=False)
        
        # Sample data for this class
        for key in subset_ist_list[class_id].keys():
            if key in ist_list[class_id]:
                subset_ist_list[class_id][key] = [ist_list[class_id][key][i] for i in indices]
        
        # Update image list (copy relevant entries)
        subset_img_list[class_id] = img_list[class_id].copy()
    
    return subset_ist_list, subset_img_list


def prepare_training_data(ist_list: List[Dict], img_list: List[List], 
                         subset_size: int = 50000) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """
    Prepare training data by analyzing class frequencies and creating stratified subset.
    
    Args:
        ist_list: List of instance dictionaries per class
        img_list: List of image lists per class
        subset_size: Target number of training samples
        
    Returns:
        all_gt_coords: [N, 4] tensor of ground truth coordinates  
        all_pred_coords: [N, 4] tensor of predicted coordinates
        all_pred_scores: [N] tensor of prediction scores
        selected_classes: List of class IDs included in training
    """
    # Analyze class frequencies
    class_frequencies = {}
    for class_id, instances in enumerate(ist_list):
        if len(instances.get('gt_x0', [])) > 0:
            class_frequencies[class_id] = len(instances['gt_x0'])
    
    # Get top 6 classes
    top_classes = get_top_classes(class_frequencies, top_k=6)
    print(f"Selected top classes: {top_classes}")
    print(f"Class frequencies: {[(c, class_frequencies[c]) for c in top_classes]}")
    
    # Create stratified subset
    subset_ist_list, subset_img_list = create_stratified_subset(
        ist_list, img_list, top_classes, subset_size
    )
    
    # Convert to tensors
    all_gt_coords = []
    all_pred_coords = []
    all_pred_scores = []
    
    for class_id in top_classes:
        if class_id < len(subset_ist_list) and len(subset_ist_list[class_id]['gt_x0']) > 0:
            # Ground truth coordinates
            gt_coords = torch.tensor([
                subset_ist_list[class_id]['gt_x0'],
                subset_ist_list[class_id]['gt_y0'],
                subset_ist_list[class_id]['gt_x1'],
                subset_ist_list[class_id]['gt_y1']
            ]).T  # [N, 4]
            
            # Predicted coordinates
            pred_coords = torch.tensor([
                subset_ist_list[class_id]['pred_x0'],
                subset_ist_list[class_id]['pred_y0'],
                subset_ist_list[class_id]['pred_x1'],
                subset_ist_list[class_id]['pred_y1']
            ]).T  # [N, 4]
            
            # Prediction scores
            pred_scores = torch.tensor(subset_ist_list[class_id]['pred_score'])  # [N]
            
            all_gt_coords.append(gt_coords)
            all_pred_coords.append(pred_coords)
            all_pred_scores.append(pred_scores)
    
    # Concatenate all classes
    all_gt_coords = torch.cat(all_gt_coords, dim=0) if all_gt_coords else torch.empty(0, 4)
    all_pred_coords = torch.cat(all_pred_coords, dim=0) if all_pred_coords else torch.empty(0, 4)
    all_pred_scores = torch.cat(all_pred_scores, dim=0) if all_pred_scores else torch.empty(0)
    
    print(f"Final training data shape: {all_gt_coords.shape[0]} samples")
    
    return all_gt_coords, all_pred_coords, all_pred_scores, top_classes


def split_data(features: torch.Tensor, gt_coords: torch.Tensor, pred_coords: torch.Tensor,
               train_frac: float = 0.5, cal_frac: float = 0.3, val_frac: float = 0.2,
               seed: int = 42) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Split data into train/calibration/validation sets.
    
    Args:
        features: [N, F] feature tensor
        gt_coords: [N, 4] ground truth coordinates
        pred_coords: [N, 4] predicted coordinates  
        train_frac: Fraction for training
        cal_frac: Fraction for calibration
        val_frac: Fraction for validation
        seed: Random seed
        
    Returns:
        train_data: Dictionary with train data
        cal_data: Dictionary with calibration data
        val_data: Dictionary with validation data
    """
    torch.manual_seed(seed)
    n_samples = features.shape[0]
    indices = torch.randperm(n_samples)
    
    # Calculate split sizes
    train_size = int(n_samples * train_frac)
    cal_size = int(n_samples * cal_frac)
    
    # Split indices
    train_indices = indices[:train_size]
    cal_indices = indices[train_size:train_size + cal_size] 
    val_indices = indices[train_size + cal_size:]
    
    # Create data splits
    train_data = {
        'features': features[train_indices],
        'gt_coords': gt_coords[train_indices],
        'pred_coords': pred_coords[train_indices]
    }
    
    cal_data = {
        'features': features[cal_indices],
        'gt_coords': gt_coords[cal_indices],
        'pred_coords': pred_coords[cal_indices]
    }
    
    val_data = {
        'features': features[val_indices],
        'gt_coords': gt_coords[val_indices],
        'pred_coords': pred_coords[val_indices]
    }
    
    print(f"Data split: train={len(train_indices)}, cal={len(cal_indices)}, val={len(val_indices)}")
    
    return train_data, cal_data, val_data


def save_training_data(data_dict: Dict, filepath: str):
    """Save training data to file."""
    torch.save(data_dict, filepath)


def load_training_data(filepath: str) -> Dict:
    """Load training data from file."""
    return torch.load(filepath)


class COCOClassMapper:
    """Helper class to map between COCO class IDs and names."""
    
    def __init__(self):
        # COCO class ID to name mapping
        self.id_to_name = {
            1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
            6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
            11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
            16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
            21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
            27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
            34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
            39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
            43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup',
            48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana',
            53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot',
            58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair',
            63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
            70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote',
            76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
            80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book',
            85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
            89: 'hair drier', 90: 'toothbrush'
        }
        
        self.name_to_id = {v: k for k, v in self.id_to_name.items()}
    
    def get_name(self, class_id: int) -> str:
        """Get class name from ID."""
        return self.id_to_name.get(class_id, f"unknown_{class_id}")
    
    def get_id(self, class_name: str) -> int:
        """Get class ID from name."""
        return self.name_to_id.get(class_name, -1)
    
    def get_top_class_names(self, class_ids: List[int]) -> List[str]:
        """Get class names for list of IDs."""
        return [self.get_name(cid) for cid in class_ids] 