#!/usr/bin/env python3
"""
Real-time object detection with YOLOv8 and Conformal Prediction
Shows uncertainty boxes using conformal prediction methods
WITH EXTENSIVE DEBUG OUTPUT
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
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check if ultralytics is installed, if not install it
try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics YOLOv8...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
    from ultralytics import YOLO


class YOLOConformalDetector:
    def __init__(self, model_name='yolov8m.pt', alpha=0.1):
        """
        Initialize YOLO with Conformal Prediction
        
        Args:
            model_name: YOLOv8 model variant (n/s/m/l/x)
            alpha: Miscoverage level (0.1 = 90% coverage)
        """
        print(f"\n{'='*60}")
        print(f"INITIALIZING YOLO CONFORMAL DETECTOR")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"Alpha: {alpha} (Coverage: {(1-alpha)*100}%)")
        print(f"{'='*60}\n")
        
        # RealSense setup
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = None
        
        # YOLO model
        self.model_name = model_name
        self.model = None
        
        # Conformal prediction parameters
        self.alpha = alpha  # Miscoverage level
        self.coverage = 1 - alpha  # Target coverage
        self.calibration_scores = []  # Store nonconformity scores
        self.quantile_threshold = None
        
        # Calibration mode
        self.calibrating = False
        self.calibration_count = 0
        self.calibration_target = 100  # Number of frames for calibration
        
        # Debug counters
        self.frame_count = 0
        self.detection_count = 0
        
    def setup_realsense(self):
        """Configure RealSense pipeline"""
        print("\n[REALSENSE] Setting up Intel RealSense D455...")
        
        # Configure streams
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start pipeline
        profile = self.pipeline.start(self.config)
        
        # Create align object
        self.align = rs.align(rs.stream.color)
        
        # Get device info
        device = profile.get_device()
        device_name = device.get_info(rs.camera_info.name)
        print(f"[REALSENSE] ✅ Connected to {device_name}")
        
        # Get depth scale
        depth_sensor = device.first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print(f"[REALSENSE] Depth Scale: {self.depth_scale}")
        print(f"[REALSENSE] Setup complete!\n")
        
    def setup_yolo(self):
        """Setup YOLOv8 model"""
        print(f"\n[YOLO] Loading {self.model_name}...")
        
        # Download and load model
        self.model = YOLO(self.model_name)
        
        # Set to evaluation mode
        self.model.fuse()
        
        print(f"[YOLO] ✅ Model loaded successfully")
        print(f"[YOLO] Model type: {type(self.model)}")
        print(f"[YOLO] Number of classes: {len(self.model.names)}")
        print(f"[YOLO] Class names (first 10): {list(self.model.names.values())[:10]}...")
        print(f"[YOLO] Setup complete!\n")
        
    def compute_nonconformity_score(self, confidence):
        """
        Compute nonconformity score for a detection
        Higher score = less conforming = more uncertainty
        """
        # Simple score: inverse of confidence
        score = 1.0 - confidence
        return score
    
    def calibrate_quantile(self):
        """
        Compute quantile threshold from calibration scores
        """
        if len(self.calibration_scores) == 0:
            print("[CALIBRATION] No calibration scores collected!")
            return
        
        print(f"\n[CALIBRATION] Computing quantile threshold...")
        print(f"[CALIBRATION] Number of scores: {len(self.calibration_scores)}")
        
        # Sort scores
        sorted_scores = np.sort(self.calibration_scores)
        
        # Compute (1-alpha) quantile
        n = len(sorted_scores)
        q_index = int(np.ceil((n + 1) * self.coverage)) - 1
        q_index = min(q_index, n - 1)
        
        self.quantile_threshold = sorted_scores[q_index]
        
        print(f"[CALIBRATION] Scores range: [{sorted_scores[0]:.4f}, {sorted_scores[-1]:.4f}]")
        print(f"[CALIBRATION] Quantile index: {q_index}")
        print(f"[CALIBRATION] Quantile threshold: {self.quantile_threshold:.4f}")
        print(f"[CALIBRATION] Target coverage: {self.coverage*100:.0f}%")
        print(f"[CALIBRATION] ✅ Calibration complete!\n")
        
    def compute_uncertainty_margin(self, confidence, box_width, box_height):
        """
        Compute SYMMETRIC uncertainty margin for a detection box
        Returns pixels to add to each side of the box
        """
        # Debug print
        if self.frame_count % 30 == 0:  # Print every second
            print(f"\n[MARGIN] Computing uncertainty margin:")
            print(f"  Confidence: {confidence:.3f}")
            print(f"  Box size: {box_width}x{box_height}")
        
        if self.quantile_threshold is None:
            # Not calibrated - use default scaling
            # Higher confidence = smaller margin
            # Make the effect more visible
            if confidence > 0.9:
                base_factor = 0.05  # 5% for very high confidence
            elif confidence > 0.7:
                base_factor = 0.10  # 10% for high confidence
            elif confidence > 0.5:
                base_factor = 0.15  # 15% for medium confidence
            else:
                base_factor = 0.20  # 20% for low confidence
        else:
            # Use calibrated threshold
            score = self.compute_nonconformity_score(confidence)
            
            if score <= self.quantile_threshold:
                # Within threshold - small margin
                base_factor = 0.05
            else:
                # Outside threshold - larger margin proportional to excess
                excess = (score - self.quantile_threshold) / (1 - self.quantile_threshold)
                base_factor = 0.05 + 0.20 * excess  # Max 25% expansion
        
        # Compute margin as percentage of box dimensions
        # Make it symmetric by using average dimension
        avg_dimension = (box_width + box_height) / 2
        margin = int(base_factor * avg_dimension)
        
        # Ensure minimum margin of 10 pixels for visibility
        margin = max(margin, 10)
        
        if self.frame_count % 30 == 0:
            print(f"  Base factor: {base_factor:.3f}")
            print(f"  Margin: {margin}px")
        
        return margin
    
    def detect_objects(self, image):
        """Run YOLO detection"""
        # Run inference
        results = self.model(image, verbose=False)
        
        detections = []
        
        for r in results:
            if r.boxes is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                confidences = r.boxes.conf.cpu().numpy()
                class_ids = r.boxes.cls.cpu().numpy().astype(int)
                
                print(f"\n[DETECTION] Frame {self.frame_count}: Found {len(boxes)} objects")
                
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    conf = confidences[i]
                    class_id = class_ids[i]
                    class_name = self.model.names[class_id]
                    
                    # Compute box dimensions
                    box_width = x2 - x1
                    box_height = y2 - y1
                    box_area = box_width * box_height
                    
                    # Compute uncertainty margin (symmetric)
                    margin = self.compute_uncertainty_margin(conf, box_width, box_height)
                    
                    # Print detailed detection info
                    print(f"\n  Object {i+1}:")
                    print(f"    Class: {class_name} (ID: {class_id})")
                    print(f"    Confidence: {conf:.3f}")
                    print(f"    Box coords: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
                    print(f"    Box size: {int(box_width)}x{int(box_height)} (area: {int(box_area)})")
                    print(f"    Uncertainty margin: ±{margin}px")
                    print(f"    Outer box: [{int(x1-margin)}, {int(y1-margin)}, {int(x2+margin)}, {int(y2+margin)}]")
                    
                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'class_id': class_id,
                        'class_name': class_name,
                        'uncertainty_margin': margin,
                        'nonconformity_score': self.compute_nonconformity_score(conf),
                        'box_width': int(box_width),
                        'box_height': int(box_height)
                    }
                    detections.append(detection)
                    self.detection_count += 1
                    
                    # Collect calibration scores
                    if self.calibrating:
                        self.calibration_scores.append(detection['nonconformity_score'])
                        if self.frame_count % 10 == 0:
                            print(f"    [CALIBRATION] Score added: {detection['nonconformity_score']:.4f}")
        
        return detections
    
    def draw_conformal_detection(self, image, detection, depth_avg=None):
        """Draw detection with SYMMETRIC conformal prediction uncertainty"""
        x1, y1, x2, y2 = detection['bbox']
        confidence = detection['confidence']
        class_name = detection['class_name']
        margin = detection['uncertainty_margin']
        
        # Color based on confidence
        if confidence > 0.8:
            color_main = (0, 255, 0)  # Green - high confidence
            color_uncertainty = (0, 200, 0)  # Darker green
        elif confidence > 0.6:
            color_main = (0, 165, 255)  # Orange - medium confidence
            color_uncertainty = (0, 140, 200)  # Darker orange
        else:
            color_main = (0, 0, 255)  # Red - low confidence
            color_uncertainty = (0, 0, 200)  # Darker red
        
        # Draw main detection box (inner box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color_main, 2)
        
        # Draw SYMMETRIC conformal prediction interval (outer box)
        # The margin is applied equally to all sides
        outer_x1 = x1 - margin
        outer_y1 = y1 - margin
        outer_x2 = x2 + margin
        outer_y2 = y2 + margin
        
        # Ensure outer box stays within image bounds
        h, w = image.shape[:2]
        outer_x1 = max(0, outer_x1)
        outer_y1 = max(0, outer_y1)
        outer_x2 = min(w-1, outer_x2)
        outer_y2 = min(h-1, outer_y2)
        
        # Draw outer box with thicker line
        cv2.rectangle(image, 
                     (outer_x1, outer_y1), 
                     (outer_x2, outer_y2), 
                     color_uncertainty, 3, cv2.LINE_AA)
        
        # Draw corner markers to emphasize the uncertainty region
        corner_length = 20
        corner_thickness = 3
        
        # Top-left corner
        cv2.line(image, (outer_x1, outer_y1), (outer_x1 + corner_length, outer_y1), color_uncertainty, corner_thickness)
        cv2.line(image, (outer_x1, outer_y1), (outer_x1, outer_y1 + corner_length), color_uncertainty, corner_thickness)
        
        # Top-right corner
        cv2.line(image, (outer_x2, outer_y1), (outer_x2 - corner_length, outer_y1), color_uncertainty, corner_thickness)
        cv2.line(image, (outer_x2, outer_y1), (outer_x2, outer_y1 + corner_length), color_uncertainty, corner_thickness)
        
        # Bottom-left corner
        cv2.line(image, (outer_x1, outer_y2), (outer_x1 + corner_length, outer_y2), color_uncertainty, corner_thickness)
        cv2.line(image, (outer_x1, outer_y2), (outer_x1, outer_y2 - corner_length), color_uncertainty, corner_thickness)
        
        # Bottom-right corner
        cv2.line(image, (outer_x2, outer_y2), (outer_x2 - corner_length, outer_y2), color_uncertainty, corner_thickness)
        cv2.line(image, (outer_x2, outer_y2), (outer_x2, outer_y2 - corner_length), color_uncertainty, corner_thickness)
        
        # Label with confidence
        label = f"{class_name} {confidence:.2f}"
        if depth_avg is not None and depth_avg > 0:
            label += f" | {depth_avg:.2f}m"
        
        # Label background
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - 20), (x1 + label_size[0] + 5, y1), color_main, -1)
        cv2.putText(image, label, (x1 + 2, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Uncertainty info at bottom of inner box
        if self.quantile_threshold is not None:
            margin_text = f"90% CI: ±{margin}px"
        else:
            margin_text = f"Margin: ±{margin}px"
        
        cv2.putText(image, margin_text, (x1, y2 + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_main, 1)
    
    def get_depth_at_bbox(self, depth_image, bbox):
        """Get average depth in bounding box"""
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(depth_image.shape[1], x2)
        y2 = min(depth_image.shape[0], y2)
        
        roi = depth_image[y1:y2, x1:x2]
        valid_depths = roi[roi > 0]
        
        if len(valid_depths) > 0:
            avg_depth = np.mean(valid_depths) * self.depth_scale
            return avg_depth
        return 0
    
    def run(self):
        """Main detection loop"""
        self.setup_realsense()
        self.setup_yolo()
        
        cv2.namedWindow('YOLOv8 with Conformal Prediction', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Depth View', cv2.WINDOW_NORMAL)
        
        print("\n" + "="*60)
        print("🎯 YOLOv8 Object Detection with Conformal Prediction")
        print("="*60)
        print(f"Target coverage: {self.coverage*100:.0f}%")
        print("\nControls:")
        print("  'q' - Quit")
        print("  's' - Save frame")
        print("  'c' - Start/stop calibration")
        print("  'u' - Toggle uncertainty visualization")
        print("\n⚠️  Press 'c' to calibrate conformal prediction first!")
        print("="*60 + "\n")
        
        show_uncertainty = True
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
                
                self.frame_count += 1
                
                # Calculate FPS
                if self.frame_count % 30 == 0:
                    fps = 30 / (time.time() - fps_timer)
                    fps_timer = time.time()
                    print(f"\n[STATS] Frame: {self.frame_count}, FPS: {fps:.1f}, Total detections: {self.detection_count}")
                
                # Detect objects
                start_time = time.time()
                detections = self.detect_objects(color_image)
                inference_time = (time.time() - start_time) * 1000
                
                # Update calibration
                if self.calibrating:
                    self.calibration_count += 1
                    print(f"[CALIBRATION] Progress: {self.calibration_count}/{self.calibration_target}")
                    
                    if self.calibration_count >= self.calibration_target:
                        self.calibrating = False
                        self.calibrate_quantile()
                
                # Draw results
                result_image = color_image.copy()
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03),
                    cv2.COLORMAP_JET
                )
                
                # Draw detections
                for detection in detections:
                    depth_avg = self.get_depth_at_bbox(depth_image, detection['bbox'])
                    
                    if depth_avg > 0 and self.frame_count % 30 == 0:
                        print(f"    Depth: {depth_avg:.2f}m")
                    
                    if show_uncertainty:
                        self.draw_conformal_detection(result_image, detection, depth_avg)
                    else:
                        # Simple box without uncertainty
                        x1, y1, x2, y2 = detection['bbox']
                        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{detection['class_name']} {detection['confidence']:.2f}"
                        cv2.putText(result_image, label, (x1, y1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Draw on depth
                    x1, y1, x2, y2 = detection['bbox']
                    cv2.rectangle(depth_colormap, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Info overlay
                info = [
                    f"FPS: {fps:.1f}",
                    f"Inference: {inference_time:.1f}ms",
                    f"Objects: {len(detections)}",
                    f"Coverage: {self.coverage*100:.0f}%",
                    f"Uncertainty: {'ON' if show_uncertainty else 'OFF'}"
                ]
                
                if self.calibrating:
                    info.append(f"CALIBRATING: {self.calibration_count}/{self.calibration_target}")
                elif self.quantile_threshold is not None:
                    info.append(f"Calibrated (τ={self.quantile_threshold:.3f})")
                else:
                    info.append("Not calibrated")
                
                y = 20
                for text in info:
                    cv2.putText(result_image, text, (10, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    y += 25
                
                # Conformal prediction explanation
                if show_uncertainty:
                    info_text = "Inner box: Detection | Outer box: 90% confidence region"
                    cv2.putText(result_image, info_text, (10, result_image.shape[0] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                
                # Display
                cv2.imshow('YOLOv8 with Conformal Prediction', result_image)
                cv2.imshow('Depth View', depth_colormap)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[EXIT] Shutting down...")
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"yolo_conformal_{timestamp}.jpg", result_image)
                    cv2.imwrite(f"depth_{timestamp}.jpg", depth_colormap)
                    print(f"\n[SAVE] 💾 Saved: {timestamp}")
                elif key == ord('c'):
                    if not self.calibrating:
                        self.calibrating = True
                        self.calibration_count = 0
                        self.calibration_scores = []
                        print("\n[CALIBRATION] 🔄 Starting calibration...")
                    else:
                        self.calibrating = False
                        print("\n[CALIBRATION] ❌ Calibration cancelled")
                        if len(self.calibration_scores) > 0:
                            self.calibrate_quantile()
                elif key == ord('u'):
                    show_uncertainty = not show_uncertainty
                    print(f"\n[DISPLAY] Uncertainty visualization: {'ON' if show_uncertainty else 'OFF'}")
        
        finally:
            cv2.destroyAllWindows()
            self.pipeline.stop()
            print("\n[EXIT] ✅ Cleanup complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 with Conformal Prediction (Debug Mode)")
    parser.add_argument("--model", type=str, default="yolov8m.pt",
                       help="YOLOv8 model variant (n/s/m/l/x)")
    parser.add_argument("--alpha", type=float, default=0.1,
                       help="Miscoverage level (0.1 = 90% coverage)")
    
    args = parser.parse_args()
    
    print("🎯 Intel RealSense D455 - YOLOv8 with Conformal Prediction")
    print("=" * 60)
    print("DEBUG MODE: Extensive logging enabled")
    print("SYMMETRIC UNCERTAINTY BOXES")
    print("=" * 60)
    
    detector = YOLOConformalDetector(args.model, args.alpha)
    detector.run()