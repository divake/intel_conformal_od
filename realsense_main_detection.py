#!/usr/bin/env python3
"""
RealSense object detection using the main conformal detection framework
"""
import sys
import os
import time
import numpy as np
import cv2
import torch
import pyrealsense2 as rs
from pathlib import Path
import argparse
import yaml

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "detectron2"))

# Import framework modules
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog
from detectron2.structures import Boxes, Instances
import detectron2.data.transforms as T

from model import model_loader
from util import util


class MainFrameworkDetector:
    def __init__(self, config_file, checkpoint_path):
        self.config_file = config_file
        self.checkpoint_path = checkpoint_path
        
        # RealSense setup
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = None
        
        # Model setup
        self.cfg = None
        self.model = None
        self.aug = None
        self.metadata = None
        
        # Logger
        self.logger = setup_logger(name=__name__)
        
    def setup_realsense(self):
        """Configure RealSense pipeline"""
        # Configure streams
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start pipeline
        profile = self.pipeline.start(self.config)
        
        # Create align object
        self.align = rs.align(rs.stream.color)
        
        # Get device info
        device = profile.get_device()
        print(f"✅ Connected to {device.get_info(rs.camera_info.name)}")
        
        # Get depth scale
        depth_sensor = device.first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        
    def setup_model(self):
        """Setup detectron2 model using the framework"""
        print("📊 Setting up model...")
        
        # Create config
        self.cfg = get_cfg()
        
        # Load config file if provided
        if self.config_file and os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Apply some settings from config
            if 'test_score_thresh' in config_data:
                score_thresh = config_data['test_score_thresh']
            else:
                score_thresh = 0.5
        else:
            score_thresh = 0.5
        
        # Set model configuration
        if "X_101" in self.checkpoint_path:
            self.cfg.merge_from_file("detectron2/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
        else:
            self.cfg.merge_from_file("detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        
        # Override settings
        self.cfg.MODEL.WEIGHTS = self.checkpoint_path
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
        self.cfg.MODEL.DEVICE = "cpu"  # Use CPU for Intel hardware
        self.cfg.INPUT.FORMAT = "BGR"
        
        # Build and load model
        try:
            cfg_model, self.model = model_loader.d2_build_model(self.cfg, self.logger)
            model_loader.d2_load_model(cfg_model, self.model, self.logger, is_train=False)
            self.model.eval()
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            # Fallback to simple loading
            from detectron2.modeling import build_model
            from detectron2.checkpoint import DetectionCheckpointer
            
            self.model = build_model(self.cfg)
            self.model.eval()
            
            checkpointer = DetectionCheckpointer(self.model)
            checkpointer.load(self.checkpoint_path)
            print("✅ Model loaded with fallback method")
        
        # Setup augmentations
        self.aug = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST],
            self.cfg.INPUT.MAX_SIZE_TEST
        )
        
        # Get metadata
        self.metadata = MetadataCatalog.get("coco_2017_val")
        
    def preprocess_image(self, image):
        """Preprocess image for model input"""
        # Apply augmentations
        aug_input = T.AugInput(image)
        transform = self.aug(aug_input)
        image_tensor = torch.as_tensor(
            aug_input.image.transpose(2, 0, 1).astype("float32")
        )
        
        inputs = [{
            "image": image_tensor,
            "height": aug_input.height,
            "width": aug_input.width,
        }]
        
        return inputs, transform
    
    def detect_objects(self, image):
        """Run object detection"""
        # Preprocess
        inputs, transform = self.preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(inputs)[0]
        
        # Get instances
        if "instances" in outputs:
            instances = outputs["instances"].to("cpu")
            
            # Transform boxes back to original image space
            instances = transform.inverse().apply_instances(instances)
            
            return instances
        else:
            # Return empty instances
            return Instances(image.shape[:2])
    
    def get_depth_at_bbox(self, depth_image, bbox):
        """Get average depth in bounding box"""
        x1, y1, x2, y2 = bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(depth_image.shape[1], x2)
        y2 = min(depth_image.shape[0], y2)
        
        roi = depth_image[y1:y2, x1:x2]
        valid_depths = roi[roi > 0]
        
        if len(valid_depths) > 0:
            avg_depth = np.mean(valid_depths) * self.depth_scale
            min_depth = np.min(valid_depths) * self.depth_scale
            max_depth = np.max(valid_depths) * self.depth_scale
            return avg_depth, min_depth, max_depth
        return 0, 0, 0
    
    def draw_instance_predictions(self, image, instances, depth_image=None):
        """Draw predictions on image"""
        result = image.copy()
        
        if len(instances) == 0:
            return result
        
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()
        
        # Get class names
        class_names = self.metadata.thing_classes
        
        # Color map for different classes
        colors = plt.cm.rainbow(np.linspace(0, 1, len(class_names)))
        
        for i in range(len(instances)):
            x1, y1, x2, y2 = boxes[i].astype(int)
            score = scores[i]
            class_id = classes[i]
            class_name = class_names[class_id]
            
            # Get color for this class
            color = (colors[class_id][:3] * 255).astype(int).tolist()
            color = tuple(color[::-1])  # RGB to BGR
            
            # Draw box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label = f"{class_name} {score:.2f}"
            
            # Add depth if available
            if depth_image is not None:
                avg_depth, min_depth, max_depth = self.get_depth_at_bbox(depth_image, boxes[i])
                if avg_depth > 0:
                    label += f" | {avg_depth:.2f}m"
            
            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result, (x1, y1 - 20), (x1 + label_size[0] + 5, y1), color, -1)
            
            # Draw label text
            cv2.putText(result, label, (x1 + 2, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result
    
    def run(self):
        """Main detection loop"""
        self.setup_realsense()
        self.setup_model()
        
        cv2.namedWindow('Detectron2 Object Detection', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Depth View', cv2.WINDOW_NORMAL)
        
        print("\n🎥 Running detectron2 object detection...")
        print("Press 'q' to quit")
        print("Press 's' to save frame")
        print("Press 't' to adjust threshold")
        
        frame_count = 0
        fps = 0
        fps_timer = time.time()
        
        try:
            while True:
                # Get frames
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue
                
                # Convert to numpy
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                frame_count += 1
                if frame_count % 30 == 0:
                    fps = 30 / (time.time() - fps_timer)
                    fps_timer = time.time()
                
                # Detect objects
                start_time = time.time()
                instances = self.detect_objects(color_image)
                inference_time = (time.time() - start_time) * 1000
                
                # Draw results
                result_image = self.draw_instance_predictions(color_image, instances, depth_image)
                
                # Create depth colormap
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03),
                    cv2.COLORMAP_JET
                )
                
                # Draw boxes on depth too
                if len(instances) > 0:
                    boxes = instances.pred_boxes.tensor.numpy()
                    for box in boxes:
                        x1, y1, x2, y2 = box.astype(int)
                        cv2.rectangle(depth_colormap, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Info overlay
                info = [
                    f"FPS: {fps:.1f}",
                    f"Inference: {inference_time:.1f}ms",
                    f"Objects: {len(instances)}",
                    f"Threshold: {self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST:.2f}",
                    "Model: Faster R-CNN"
                ]
                
                y = 20
                for text in info:
                    cv2.putText(result_image, text, (10, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    y += 25
                
                # Display
                cv2.imshow('Detectron2 Object Detection', result_image)
                cv2.imshow('Depth View', depth_colormap)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"detection_{timestamp}.jpg", result_image)
                    cv2.imwrite(f"depth_{timestamp}.jpg", depth_colormap)
                    print(f"💾 Saved: {timestamp}")
                elif key == ord('t'):
                    # Cycle through thresholds
                    current = self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
                    if current >= 0.9:
                        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
                    else:
                        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST += 0.1
                    print(f"Threshold: {self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST:.2f}")
        
        finally:
            cv2.destroyAllWindows()
            self.pipeline.stop()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser(description="RealSense Detectron2 Object Detection")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config file")
    parser.add_argument("--checkpoint", type=str,
                       default="checkpoints/faster_rcnn_X_101_32x8d_FPN_3x.pth",
                       help="Path to model checkpoint")
    
    args = parser.parse_args()
    
    print("🎯 RealSense Object Detection with Detectron2")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    detector = MainFrameworkDetector(args.config, args.checkpoint)
    detector.run()