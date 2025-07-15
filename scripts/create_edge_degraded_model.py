#!/usr/bin/env python3
"""
Create a systematically degraded model for edge deployment.
This simulates deploying a smaller, quantized model on edge devices.

Degradation strategy:
1. Feature dimension reduction (17 → 8-10 features)
2. Feature quantization (FP32 → INT8 simulation)
3. Detection quality degradation (IoU, confidence)
4. Systematic measurement of degradation
"""

import numpy as np
import torch
from pathlib import Path
import pickle
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime

class EdgeModelDegrader:
    def __init__(self, target_performance=0.6, seed=42):
        """
        Initialize degrader for edge deployment simulation.
        
        Args:
            target_performance: Target performance (0.6 = 60% of original)
            seed: Random seed for reproducibility
        """
        self.target_performance = target_performance
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Degradation parameters
        self.feature_reduction_ratio = 0.5  # Reduce features by 50%
        self.quantization_bits = 8  # INT8 quantization
        self.detection_drop_rate = 0.15  # Drop 15% of detections
        self.iou_degradation = 0.1  # Reduce IoU by 10%
        self.confidence_noise = 0.05  # Add 5% noise to confidence
        
    def load_base_cache(self, cache_dir):
        """Load base model cached data."""
        cache_path = Path(cache_dir)
        
        # Load features
        train_features = torch.load(cache_path / "features_train.pt")
        val_features = torch.load(cache_path / "features_val.pt")
        
        # Load predictions
        with open(cache_path / "predictions_train.pkl", 'rb') as f:
            train_preds, train_labels = pickle.load(f)
            
        with open(cache_path / "predictions_val.pkl", 'rb') as f:
            val_preds, val_labels = pickle.load(f)
            
        print(f"Loaded base cache:")
        print(f"  Train: {train_features['features'].shape}")
        print(f"  Val: {val_features['features'].shape}")
        
        return {
            'train_features': train_features,
            'val_features': val_features,
            'train_preds': train_preds,
            'train_labels': train_labels,
            'val_preds': val_preds,
            'val_labels': val_labels
        }
    
    def reduce_features(self, features, fit_pca=True):
        """
        Simulate edge device feature extraction with reduced dimensions.
        This mimics using a smaller backbone (e.g., MobileNet vs ResNet50).
        """
        original_dim = features.shape[1]
        target_dim = int(original_dim * self.feature_reduction_ratio)
        
        print(f"\nReducing features: {original_dim} → {target_dim} dimensions")
        
        if fit_pca:
            # Fit PCA on training data
            self.scaler = StandardScaler()
            scaled_features = self.scaler.fit_transform(features.numpy())
            
            self.pca = PCA(n_components=target_dim, random_state=self.seed)
            reduced_features = self.pca.fit_transform(scaled_features)
            
            # Print explained variance
            explained_var = np.sum(self.pca.explained_variance_ratio_)
            print(f"  PCA explained variance: {explained_var:.2%}")
            
        else:
            # Transform using fitted PCA
            scaled_features = self.scaler.transform(features.numpy())
            reduced_features = self.pca.transform(scaled_features)
        
        return torch.FloatTensor(reduced_features)
    
    def quantize_features(self, features):
        """
        Simulate INT8 quantization effects on features.
        """
        # Get quantization range
        min_val = features.min()
        max_val = features.max()
        
        # Simulate INT8 quantization
        scale = (max_val - min_val) / 255.0
        quantized = torch.round((features - min_val) / scale)
        dequantized = quantized * scale + min_val
        
        # Add quantization noise
        quant_error = features - dequantized
        noise_std = quant_error.std() * 0.1
        noise = torch.randn_like(features) * noise_std
        
        return dequantized + noise
    
    def degrade_detections(self, predictions, labels):
        """
        Degrade detection quality to simulate edge model performance.
        """
        degraded_preds = []
        degraded_labels = []
        
        total_detections = 0
        kept_detections = 0
        
        for img_idx, (img_preds, img_labels) in enumerate(zip(predictions, labels)):
            if 'iou' not in img_labels or len(img_labels['iou']) == 0:
                degraded_preds.append(img_preds)
                degraded_labels.append(img_labels)
                continue
            
            # Copy original data
            new_preds = img_preds.copy()
            new_labels = img_labels.copy()
            
            num_dets = len(img_labels['iou'])
            total_detections += num_dets
            
            # 1. Drop some detections (simulate missed detections)
            keep_mask = np.random.rand(num_dets) > self.detection_drop_rate
            
            # 2. Degrade IoU for kept detections
            degraded_iou = np.array(new_labels['iou']) * (1 - self.iou_degradation)
            
            # 3. Add noise to confidence scores
            if 'pred_score' in new_labels:
                confidence_noise = np.random.normal(0, self.confidence_noise, num_dets)
                pred_scores = np.array(new_labels['pred_score'])
                new_labels['pred_score'] = np.clip(
                    pred_scores + confidence_noise, 0, 1
                ).tolist()
            
            # 4. Add coordinate noise (simulate localization errors)
            coord_noise = np.random.normal(0, 5, new_preds['pred_coords'].shape)
            new_preds['pred_coords'] = new_preds['pred_coords'] + coord_noise
            
            # Apply keep mask
            new_labels['iou'] = degraded_iou[keep_mask].tolist()
            if 'pred_score' in new_labels:
                pred_scores_array = np.array(new_labels['pred_score'])
                new_labels['pred_score'] = pred_scores_array[keep_mask].tolist()
            
            # Filter predictions
            new_preds['pred_coords'] = new_preds['pred_coords'][keep_mask]
            new_preds['pred_cls'] = new_preds['pred_cls'][keep_mask]
            if 'pred_score' in new_preds:
                new_preds['pred_score'] = new_preds['pred_score'][keep_mask]
            
            kept_detections += keep_mask.sum()
            
            degraded_preds.append(new_preds)
            degraded_labels.append(new_labels)
        
        print(f"\nDetection degradation:")
        print(f"  Total detections: {total_detections}")
        print(f"  Kept detections: {kept_detections} ({kept_detections/total_detections:.1%})")
        
        return degraded_preds, degraded_labels
    
    def measure_degradation(self, original_data, degraded_data):
        """
        Measure various aspects of degradation.
        """
        metrics = {
            'feature_metrics': {},
            'detection_metrics': {},
            'overall_metrics': {}
        }
        
        # Feature metrics
        orig_features = original_data['val_features']['features']
        deg_features = degraded_data['val_features']['features']
        
        metrics['feature_metrics']['dimension_reduction'] = {
            'original': orig_features.shape[1],
            'degraded': deg_features.shape[1],
            'reduction_ratio': 1 - deg_features.shape[1] / orig_features.shape[1]
        }
        
        # Detection metrics
        orig_labels = original_data['val_labels']
        deg_labels = degraded_data['val_labels']
        
        orig_ious = []
        deg_ious = []
        
        for orig, deg in zip(orig_labels, deg_labels):
            if 'iou' in orig:
                orig_ious.extend(orig['iou'])
            if 'iou' in deg:
                deg_ious.extend(deg['iou'])
        
        metrics['detection_metrics']['iou_degradation'] = {
            'original_mean': np.mean(orig_ious),
            'degraded_mean': np.mean(deg_ious),
            'degradation_percent': (1 - np.mean(deg_ious) / np.mean(orig_ious)) * 100
        }
        
        metrics['detection_metrics']['detection_count'] = {
            'original': len(orig_ious),
            'degraded': len(deg_ious),
            'drop_rate': (1 - len(deg_ious) / len(orig_ious)) * 100
        }
        
        # Estimate overall performance
        # Simplified mAP approximation based on IoU>0.5
        orig_ap50 = np.mean([iou > 0.5 for iou in orig_ious])
        deg_ap50 = np.mean([iou > 0.5 for iou in deg_ious])
        
        metrics['overall_metrics']['estimated_performance'] = {
            'original_ap50': orig_ap50,
            'degraded_ap50': deg_ap50,
            'performance_ratio': deg_ap50 / orig_ap50,
            'performance_drop': (1 - deg_ap50 / orig_ap50) * 100
        }
        
        return metrics
    
    def save_degraded_cache(self, degraded_data, output_dir, metrics):
        """Save degraded cache and metadata."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save features
        torch.save(degraded_data['train_features'], output_path / "features_train.pt")
        torch.save(degraded_data['val_features'], output_path / "features_val.pt")
        
        # Save predictions
        with open(output_path / "predictions_train.pkl", 'wb') as f:
            pickle.dump((degraded_data['train_preds'], degraded_data['train_labels']), f)
        
        with open(output_path / "predictions_val.pkl", 'wb') as f:
            pickle.dump((degraded_data['val_preds'], degraded_data['val_labels']), f)
        
        # Save degradation info
        degradation_info = {
            'target_performance': self.target_performance,
            'degradation_timestamp': datetime.now().isoformat(),
            'degradation_parameters': {
                'feature_reduction_ratio': self.feature_reduction_ratio,
                'quantization_bits': self.quantization_bits,
                'detection_drop_rate': self.detection_drop_rate,
                'iou_degradation': self.iou_degradation,
                'confidence_noise': self.confidence_noise
            },
            'metrics': metrics,
            'edge_deployment_simulation': {
                'simulated_model': 'MobileNet-like (reduced features)',
                'quantization': 'INT8',
                'target_device': 'Edge TPU / Mobile processor'
            }
        }
        
        with open(output_path / "degradation_info.json", 'w') as f:
            json.dump(degradation_info, f, indent=2)
        
        print(f"\nSaved degraded cache to: {output_path}")
    
    def create_edge_degraded_model(self, base_cache_dir, output_dir):
        """Main method to create edge-degraded model."""
        print("="*80)
        print("CREATING EDGE-DEGRADED MODEL")
        print(f"Target Performance: {self.target_performance*100}%")
        print("="*80)
        
        # Load base cache
        original_data = self.load_base_cache(base_cache_dir)
        
        # 1. Feature degradation (dimension reduction + quantization)
        print("\n1. FEATURE DEGRADATION")
        print("-"*40)
        
        # Reduce dimensions (fit on train)
        train_features_reduced = self.reduce_features(
            original_data['train_features']['features'], 
            fit_pca=True
        )
        val_features_reduced = self.reduce_features(
            original_data['val_features']['features'], 
            fit_pca=False
        )
        
        # Quantize features
        train_features_quantized = self.quantize_features(train_features_reduced)
        val_features_quantized = self.quantize_features(val_features_reduced)
        
        # 2. Detection degradation
        print("\n2. DETECTION DEGRADATION")
        print("-"*40)
        
        train_preds_deg, train_labels_deg = self.degrade_detections(
            original_data['train_preds'], 
            original_data['train_labels']
        )
        val_preds_deg, val_labels_deg = self.degrade_detections(
            original_data['val_preds'], 
            original_data['val_labels']
        )
        
        # Create degraded data structure
        degraded_data = {
            'train_features': {
                'features': train_features_quantized,
                'pred_coords': original_data['train_features']['pred_coords'],
                'gt_coords': original_data['train_features']['gt_coords'],
                'confidence': original_data['train_features']['confidence']
            },
            'val_features': {
                'features': val_features_quantized,
                'pred_coords': original_data['val_features']['pred_coords'],
                'gt_coords': original_data['val_features']['gt_coords'],
                'confidence': original_data['val_features']['confidence']
            },
            'train_preds': train_preds_deg,
            'train_labels': train_labels_deg,
            'val_preds': val_preds_deg,
            'val_labels': val_labels_deg
        }
        
        # 3. Measure degradation
        print("\n3. DEGRADATION METRICS")
        print("-"*40)
        metrics = self.measure_degradation(original_data, degraded_data)
        
        # Print summary
        print(f"\nFeature Degradation:")
        print(f"  Dimensions: {metrics['feature_metrics']['dimension_reduction']['original']} → "
              f"{metrics['feature_metrics']['dimension_reduction']['degraded']} "
              f"(-{metrics['feature_metrics']['dimension_reduction']['reduction_ratio']:.1%})")
        
        print(f"\nDetection Degradation:")
        print(f"  Mean IoU: {metrics['detection_metrics']['iou_degradation']['original_mean']:.3f} → "
              f"{metrics['detection_metrics']['iou_degradation']['degraded_mean']:.3f} "
              f"(-{metrics['detection_metrics']['iou_degradation']['degradation_percent']:.1f}%)")
        print(f"  Detection count: {metrics['detection_metrics']['detection_count']['original']} → "
              f"{metrics['detection_metrics']['detection_count']['degraded']} "
              f"(-{metrics['detection_metrics']['detection_count']['drop_rate']:.1f}%)")
        
        print(f"\nOverall Performance:")
        print(f"  Estimated AP@50: {metrics['overall_metrics']['estimated_performance']['original_ap50']:.3f} → "
              f"{metrics['overall_metrics']['estimated_performance']['degraded_ap50']:.3f}")
        print(f"  Performance ratio: {metrics['overall_metrics']['estimated_performance']['performance_ratio']:.1%}")
        print(f"  Performance drop: -{metrics['overall_metrics']['estimated_performance']['performance_drop']:.1f}%")
        
        # 4. Save degraded cache
        self.save_degraded_cache(degraded_data, output_dir, metrics)
        
        return degraded_data, metrics


def main():
    """Create edge-degraded model with 60% performance."""
    
    # Configuration
    base_cache_dir = "/ssd_4TB/divake/conformal-od/learnable_scoring_fn/cache_base_model"
    output_dir = "/ssd_4TB/divake/conformal-od/learnable_scoring_fn/cache_edge_60"
    target_performance = 0.6  # 60% of original performance
    
    # Create degrader
    degrader = EdgeModelDegrader(target_performance=target_performance)
    
    # Create degraded model
    degraded_data, metrics = degrader.create_edge_degraded_model(
        base_cache_dir, 
        output_dir
    )
    
    print("\n" + "="*80)
    print("EDGE-DEGRADED MODEL CREATED SUCCESSFULLY!")
    print("="*80)
    print(f"\nNext steps:")
    print(f"1. Train learnable scoring function on degraded model:")
    print(f"   python train_size_aware.py --cache_dir {output_dir}")
    print(f"\n2. Compare results:")
    print(f"   - Base model: ~42 MPIW at 89% coverage")
    print(f"   - Expected: ~50-55 MPIW at 90% coverage")
    print(f"   - Key metric: MPIW increase vs performance drop ratio")


if __name__ == "__main__":
    main()