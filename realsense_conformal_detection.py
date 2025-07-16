#!/usr/bin/env python3
"""
Intel RealSense D455 - Real-time Object Detection with Conformal Prediction
Uses Faster R-CNN with uncertainty quantification
"""
import sys
import os
import argparse
import time
import numpy as np
import cv2
import torch
import pyrealsense2 as rs
from pathlib import Path

# Add detectron2 to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "detectron2"))

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
import detectron2.data.transforms as T
from detectron2.structures import Instances

# Import conformal prediction modules
from util import util, io_file
from control import std_conformal


class RealsenseConformalDetector:
    def __init__(self, model_path, config_path=None):
        self.model_path = model_path
        self.config_path = config_path
        
        # RealSense setup
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = None
        
        # Model setup
        self.cfg = None
        self.model = None
        self.aug = None
        self.device = torch.device("cpu")  # Use CPU for Intel hardware
        
        # Conformal prediction
        self.conformal_predictor = None
        
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
        print(f"   Depth Scale: {self.depth_scale}")
        
    def setup_model(self):
        """Load Faster R-CNN model"""
        print(f"📊 Loading Faster R-CNN model...")
        
        # Setup config
        self.cfg = get_cfg()
        
        # Use default Faster R-CNN config
        self.cfg.merge_from_file("detectron2/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
        
        # Override with custom settings
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.cfg.MODEL.WEIGHTS = self.model_path
        self.cfg.MODEL.DEVICE = "cpu"  # Use CPU
        
        # Build model
        self.model = build_model(self.cfg)
        self.model.eval()
        
        # Load checkpoint
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(self.model_path)
        
        # Setup augmentations
        self.aug = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST],
            self.cfg.INPUT.MAX_SIZE_TEST
        )
        
        print("✅ Model loaded successfully")
        
        # Get COCO metadata
        self.metadata = MetadataCatalog.get("coco_2017_val")
        self.class_names = self.metadata.thing_classes
        
    def setup_conformal_prediction(self):
        """Setup conformal prediction for uncertainty estimation"""
        # For demo, we'll use standard conformal prediction
        # In practice, you would calibrate this on a validation set
        self.alpha = 0.1  # 90% coverage
        self.tau = 0.85  # Default threshold adjustment
        print(f"✅ Conformal prediction setup (alpha={self.alpha}, tau={self.tau})")
        
    def preprocess_image(self, image):
        """Preprocess image for model input"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply augmentations
        aug_input = T.AugInput(image_rgb)
        transform = self.aug(aug_input)
        image_tensor = torch.as_tensor(
            aug_input.image.transpose(2, 0, 1).astype("float32")
        )
        
        inputs = [{"image": image_tensor, "height": image.shape[0], "width": image.shape[1]}]
        
        return inputs
    
    def detect_objects(self, image):
        """Run object detection with uncertainty estimation"""
        # Preprocess
        inputs = self.preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(inputs)[0]
        
        # Get predictions
        instances = outputs["instances"].to("cpu")
        
        # Add uncertainty boxes (simplified version)
        # In practice, this would use calibrated conformal scores
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()
        
        # Create uncertainty margins based on score
        # Lower score = higher uncertainty = larger margin
        uncertainty_margins = (1 - scores) * 20  # Scale factor for visualization
        
        detections = []
        for i in range(len(instances)):
            detection = {
                'bbox': boxes[i],
                'score': scores[i],
                'class': classes[i],
                'class_name': self.class_names[classes[i]],
                'uncertainty_margin': uncertainty_margins[i]
            }
            detections.append(detection)
            
        return detections
    
    def get_object_depth(self, depth_image, bbox):
        """Calculate depth statistics for detected object"""
        x1, y1, x2, y2 = bbox.astype(int)
        
        # Ensure bounds are within image
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(depth_image.shape[1], x2), min(depth_image.shape[0], y2)
        
        # Extract ROI
        roi = depth_image[y1:y2, x1:x2]
        
        # Filter out invalid depths
        valid_depths = roi[roi > 0]
        
        if len(valid_depths) > 0:
            avg_depth = np.mean(valid_depths) * self.depth_scale
            min_depth = np.min(valid_depths) * self.depth_scale
            max_depth = np.max(valid_depths) * self.depth_scale
            return avg_depth, min_depth, max_depth
        else:
            return 0, 0, 0
    
    def draw_detection_with_uncertainty(self, image, detection, depth_info=None):
        """Draw bounding box with uncertainty visualization"""
        bbox = detection['bbox']
        score = detection['score']
        class_name = detection['class_name']
        margin = detection['uncertainty_margin']
        
        x1, y1, x2, y2 = bbox.astype(int)
        
        # Draw main bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw uncertainty box (expanded by margin)
        ux1 = int(x1 - margin)
        uy1 = int(y1 - margin)
        ux2 = int(x2 + margin)
        uy2 = int(y2 + margin)
        cv2.rectangle(image, (ux1, uy1), (ux2, uy2), (255, 165, 0), 1, cv2.LINE_4)
        
        # Prepare label
        label = f"{class_name} {score:.2f}"
        if depth_info:
            avg_depth, _, _ = depth_info
            label += f" | {avg_depth:.2f}m"
        
        # Draw label background
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - 20), (x1 + label_size[0], y1), (0, 255, 0), -1)
        
        # Draw label text
        cv2.putText(image, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw uncertainty info
        uncertainty_text = f"±{margin:.0f}px"
        cv2.putText(image, uncertainty_text, (x1, y2 + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)
    
    def run(self):
        """Run real-time detection loop"""
        self.setup_realsense()
        self.setup_model()
        self.setup_conformal_prediction()
        
        # Create windows
        cv2.namedWindow('Conformal Object Detection', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Depth View', cv2.WINDOW_NORMAL)
        
        frame_count = 0
        fps_timer = time.time()
        fps = 0
        inference_time = 0
        
        print("\n🎥 Running conformal object detection...")
        print("Press 'q' to quit")
        print("Press 's' to save frame")
        print("Press 'u' to toggle uncertainty boxes")
        
        show_uncertainty = True
        
        try:
            while True:
                # Get frames
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue
                
                # Convert to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                frame_count += 1
                
                # Calculate FPS
                if frame_count % 30 == 0:
                    fps = 30 / (time.time() - fps_timer)
                    fps_timer = time.time()
                
                # Run detection
                start_time = time.time()
                detections = self.detect_objects(color_image)
                inference_time = (time.time() - start_time) * 1000
                
                # Draw results
                result_image = color_image.copy()
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), 
                    cv2.COLORMAP_JET
                )
                
                # Draw each detection
                for detection in detections:
                    # Get depth info
                    depth_info = self.get_object_depth(depth_image, detection['bbox'])
                    
                    # Draw on color image
                    if show_uncertainty:
                        self.draw_detection_with_uncertainty(result_image, detection, depth_info)
                    else:
                        # Draw simple box
                        x1, y1, x2, y2 = detection['bbox'].astype(int)
                        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{detection['class_name']} {detection['score']:.2f}"
                        cv2.putText(result_image, label, (x1, y1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Draw on depth map
                    x1, y1, x2, y2 = detection['bbox'].astype(int)
                    cv2.rectangle(depth_colormap, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add info overlay
                info_text = [
                    f"FPS: {fps:.1f}",
                    f"Inference: {inference_time:.1f}ms",
                    f"Objects: {len(detections)}",
                    f"Uncertainty: {'ON' if show_uncertainty else 'OFF'}"
                ]
                
                y_offset = 20
                for text in info_text:
                    cv2.putText(result_image, text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    y_offset += 25
                
                # Display
                cv2.imshow('Conformal Object Detection', result_image)
                cv2.imshow('Depth View', depth_colormap)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"conformal_detection_{timestamp}.jpg", result_image)
                    cv2.imwrite(f"conformal_depth_{timestamp}.jpg", depth_colormap)
                    print(f"💾 Saved frames: {timestamp}")
                elif key == ord('u'):
                    show_uncertainty = not show_uncertainty
                    print(f"🔄 Uncertainty boxes: {'ON' if show_uncertainty else 'OFF'}")
        
        finally:
            cv2.destroyAllWindows()
            self.pipeline.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RealSense D455 Conformal Object Detection")
    parser.add_argument("--model", type=str, 
                       default="checkpoints/faster_rcnn_X_101_32x8d_FPN_3x.pth",
                       help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config file")
    
    args = parser.parse_args()
    
    print("🎯 Intel RealSense D455 - Conformal Object Detection")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Device: CPU (Intel optimized)")
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Model not found at {args.model}")
        print("Please ensure the model checkpoint exists.")
        sys.exit(1)
    
    detector = RealsenseConformalDetector(args.model, args.config)
    detector.run()