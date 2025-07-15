#!/usr/bin/env python3
"""
Simple runner script for YOLOv3 cache generation.

This script generates cache files from YOLOv3/Gaussian YOLOv3 checkpoints.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to sys.path for imports
sys.path.append(str(Path(__file__).parent.parent))

def main():
    """Main function to run YOLOv3 cache generation with provided paths."""
    print("="*80)
    print("YOLOV3 CACHE GENERATION SCRIPT")
    print("="*80)
    
    # ================================================================================
    # CONFIGURATION SECTION - Modify these parameters for different models/experiments
    # ================================================================================
    
    # Model Configuration
    checkpoint_path = "/ssd_4TB/divake/conformal-od/checkpoints/gaussian_yolov3_coco.pth"
    config_path = "/ssd_4TB/divake/conformal-od/gaussian_yolo/gaussian_yolov3_eval.yaml"
    
    # Output Configuration  
    output_dir = "/ssd_4TB/divake/conformal-od/learnable_scoring_fn/cache_base_model_yolov3"
    
    # Dataset Limits (set to None for full dataset)
    max_train_images = None  # Set to None for full COCO train set (118k images)
    max_val_images = None    # Set to None for full COCO val set (5k images)
    
    # Threshold Configuration
    confidence_threshold = 0.5  # Confidence threshold for YOLOv3 predictions
    iou_threshold = 0.5         # IoU threshold for matching predictions to GT
    nms_threshold = 0.6         # NMS threshold for YOLOv3 postprocessing
    imgsize = 416               # Input image size for YOLOv3
    
    # Device Configuration
    device = "auto"
    
    # ================================================================================
    # END CONFIGURATION SECTION
    # ================================================================================
    
    # Common COCO dataset paths to try (prioritize user's actual path)
    possible_coco_paths = [
        "/ssd_4TB/divake/conformal-od/data/coco",  # User's actual path
        "/ssd_4TB/divake/coco",
        "/ssd_4TB/divake/datasets/coco", 
        "/ssd_4TB/divake/data/coco",
        "/datasets/coco",
        "/data/coco",
        os.path.expanduser("~/datasets/coco")
    ]
    
    # Find COCO dataset
    coco_dir = None
    for path in possible_coco_paths:
        if os.path.exists(path):
            # Check if it has the expected structure
            annotations_dir = os.path.join(path, "annotations")
            train_dir = os.path.join(path, "train2017") 
            val_dir = os.path.join(path, "val2017")
            train_json = os.path.join(annotations_dir, "instances_train2017.json")
            val_json = os.path.join(annotations_dir, "instances_val2017.json")
            
            if (os.path.exists(annotations_dir) and
                os.path.exists(train_dir) and
                os.path.exists(val_dir) and
                os.path.exists(train_json) and
                os.path.exists(val_json)):
                coco_dir = path
                print(f"✓ Found COCO dataset at: {path}")
                break
    
    if coco_dir is None:
        print("COCO dataset not found. Please specify the correct path.")
        print("Expected structure:")
        print("  coco_dir/")
        print("    annotations/")
        print("      instances_train2017.json")
        print("      instances_val2017.json")
        print("    train2017/")
        print("    val2017/")
        print()
        print("Please specify the COCO directory path:")
        coco_dir = input("COCO directory: ").strip()
        
        if not os.path.exists(coco_dir):
            print(f"Error: Directory {coco_dir} does not exist")
            return 1
    
    print(f"Using COCO directory: {coco_dir}")
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file {checkpoint_path} does not exist")
        return 1
    
    # Check if config exists
    if not os.path.exists(config_path):
        print(f"Error: Config file {config_path} does not exist")
        return 1
    
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Using config: {config_path}")
    print(f"Output directory: {output_dir}")
    
    # Import and run YOLOv3 cache generation
    try:
        from scripts.generate_cache_yolo import YOLOCacheGenerator
        
        # Create cache generator with configured parameters
        generator = YOLOCacheGenerator(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            coco_data_dir=coco_dir,
            output_dir=output_dir,
            device=device,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            nms_threshold=nms_threshold,
            imgsize=imgsize
        )
        
        print(f"\nGenerating YOLOv3 cache with configuration:")
        print(f"  Max train images: {max_train_images}")
        print(f"  Max val images: {max_val_images}")
        print(f"  Confidence threshold: {confidence_threshold}")
        print(f"  IoU threshold: {iou_threshold}")
        print(f"  NMS threshold: {nms_threshold}")
        print(f"  Image size: {imgsize}")
        print(f"  Device: {device}")
        print(f"  (Set max_*_images to None for full dataset)")
        
        # Generate cache
        generator.generate_cache(
            max_train_images=max_train_images,
            max_val_images=max_val_images
        )
        
        print("\nYOLOv3 cache generation completed successfully!")
        return 0
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please make sure all dependencies are installed:")
        print("  YOLOv3 components should be available")
        print("  detectron2 should be installed")
        print("  torch, torchvision, opencv-python should be installed")
        return 1
    except Exception as e:
        print(f"Error during cache generation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 