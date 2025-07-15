import torch
import numpy as np
from typing import Tuple, Dict, Optional


class FeatureExtractor:
    """Extract hand-crafted features from bounding box predictions."""
    
    def __init__(self, img_height: int = 480, img_width: int = 640):
        self.img_height = img_height
        self.img_width = img_width
        self.feature_stats = None
        
    def extract_features(self, pred_coords: torch.Tensor, pred_scores: torch.Tensor, 
                        img_heights: Optional[torch.Tensor] = None, 
                        img_widths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract hand-crafted features from predictions.
        
        Args:
            pred_coords: [N, 4] predicted coordinates (x0, y0, x1, y1)
            pred_scores: [N] confidence scores
            img_heights: [N] image heights (optional, defaults to self.img_height)
            img_widths: [N] image widths (optional, defaults to self.img_width)
            
        Returns:
            features: [N, 11] feature tensor
                     [x0, y0, x1, y1, confidence, area, aspect_ratio, 
                      center_x, center_y, rel_pos_x, rel_pos_y, rel_size, edge_dist]
        """
        device = pred_coords.device
        batch_size = pred_coords.shape[0]
        
        # Use default image sizes if not provided
        if img_heights is None:
            img_heights = torch.full((batch_size,), self.img_height, device=device)
        if img_widths is None:
            img_widths = torch.full((batch_size,), self.img_width, device=device)
        
        # Extract coordinate components
        x0, y0, x1, y1 = pred_coords[:, 0], pred_coords[:, 1], pred_coords[:, 2], pred_coords[:, 3]
        
        # 1. Basic coordinates (already included)
        coords = pred_coords  # [N, 4]
        
        # 2. Confidence scores
        confidence = pred_scores.unsqueeze(1)  # [N, 1]
        
        # 3. Box area (with log transformation)
        width = x1 - x0
        height = y1 - y0
        area = (width * height).clamp(min=1e-6)  # Avoid log(0)
        log_area = torch.log(area).unsqueeze(1)  # [N, 1]
        
        # 4. Aspect ratio
        aspect_ratio = (width / height.clamp(min=1e-6)).unsqueeze(1)  # [N, 1]
        
        # 5. Box center coordinates (normalized by image size)
        center_x = (((x0 + x1) / 2) / img_widths).unsqueeze(1)  # [N, 1]
        center_y = (((y0 + y1) / 2) / img_heights).unsqueeze(1)  # [N, 1]
        
        # 6. Position relative to image center
        img_center_x = img_widths / 2
        img_center_y = img_heights / 2
        rel_pos_x = ((((x0 + x1) / 2) - img_center_x) / img_widths).unsqueeze(1)  # [N, 1]
        rel_pos_y = ((((y0 + y1) / 2) - img_center_y) / img_heights).unsqueeze(1)  # [N, 1]
        
        # 7. Box size relative to image size
        rel_size = (area / (img_heights * img_widths)).unsqueeze(1)  # [N, 1]
        
        # 8. Distance to nearest edge (normalized)
        dist_left = x0 / img_widths
        dist_right = (img_widths - x1) / img_widths
        dist_top = y0 / img_heights
        dist_bottom = (img_heights - y1) / img_heights
        edge_dist = torch.min(torch.stack([dist_left, dist_right, dist_top, dist_bottom], dim=1), dim=1)[0].unsqueeze(1)  # [N, 1]
        
        # Concatenate all features
        features = torch.cat([
            coords,           # [N, 4] - x0, y0, x1, y1
            confidence,       # [N, 1] - confidence score
            log_area,         # [N, 1] - log(area)
            aspect_ratio,     # [N, 1] - width/height
            center_x,         # [N, 1] - normalized center x
            center_y,         # [N, 1] - normalized center y
            rel_pos_x,        # [N, 1] - position relative to image center x
            rel_pos_y,        # [N, 1] - position relative to image center y
            rel_size,         # [N, 1] - box size relative to image
            edge_dist         # [N, 1] - distance to nearest edge
        ], dim=1)
        
        return features  # [N, 13] total features
    
    def fit_normalizer(self, features: torch.Tensor):
        """
        Fit normalization statistics on training data.
        
        Args:
            features: [N, F] training features
        """
        self.feature_stats = {
            'mean': features.mean(dim=0),
            'std': features.std(dim=0).clamp(min=1e-6)
        }
    
    def normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Normalize features using fitted statistics.
        
        Args:
            features: [N, F] features to normalize
            
        Returns:
            normalized_features: [N, F] normalized features
        """
        if self.feature_stats is None:
            raise ValueError("Must call fit_normalizer() first")
        
        # Get the device of input features
        device = features.device
        
        # Apply consistent z-score normalization to all features for stability
        mean_vals = self.feature_stats['mean'].to(device)
        std_vals = self.feature_stats['std'].to(device)
        
        normalized = (features - mean_vals) / std_vals
        
        return normalized
    
    def save_stats(self, filepath: str):
        """Save normalization statistics."""
        if self.feature_stats is not None:
            torch.save(self.feature_stats, filepath)
    
    def load_stats(self, filepath: str):
        """Load normalization statistics."""
        self.feature_stats = torch.load(filepath)


def get_feature_names() -> list:
    """Return list of feature names in order."""
    return [
        'x0', 'y0', 'x1', 'y1',           # coordinates
        'confidence',                      # confidence score  
        'log_area',                        # log(box area)
        'aspect_ratio',                    # width/height
        'center_x_norm',                   # normalized center x
        'center_y_norm',                   # normalized center y
        'rel_pos_x',                       # relative position x
        'rel_pos_y',                       # relative position y
        'rel_size',                        # relative box size
        'edge_distance'                    # distance to nearest edge
    ] 