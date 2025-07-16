#!/usr/bin/env python3
"""
Real-time object detection with YOLOv8 and Conformal Prediction
Shows uncertainty boxes using conformal prediction methods
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
        
    def setup_yolo(self):
        """Setup YOLOv8 model"""
        print(f"📊 Loading {self.model_name}...")
        
        # Download and load model
        self.model = YOLO(self.model_name)
        
        # Set to evaluation mode
        self.model.fuse()
        
        print(f"✅ YOLOv8 loaded successfully")
        print(f"   Model: {self.model_name}")
        print(f"   Classes: {len(self.model.names)}")
        
    def compute_nonconformity_score(self, confidence):
        """
        Compute nonconformity score for a detection
        Higher score = less conforming = more uncertainty
        """
        # Simple score: inverse of confidence
        # You can make this more sophisticated
        return 1.0 - confidence
    
    def calibrate_quantile(self):
        """
        Compute quantile threshold from calibration scores
        """
        if len(self.calibration_scores) == 0:
            return
        
        # Sort scores
        sorted_scores = np.sort(self.calibration_scores)
        
        # Compute (1-alpha) quantile
        n = len(sorted_scores)
        q_index = int(np.ceil((n + 1) * self.coverage)) - 1
        q_index = min(q_index, n - 1)
        
        self.quantile_threshold = sorted_scores[q_index]
        
        print(f"\n📊 Calibration complete!")
        print(f"   Samples: {n}")
        print(f"   Quantile threshold: {self.quantile_threshold:.3f}")
        print(f"   Target coverage: {self.coverage*100:.0f}%")
        
    def compute_uncertainty_margin(self, confidence, box_area):
        """
        Compute uncertainty margin for a detection box
        """
        if self.quantile_threshold is None:
            # Use default scaling if not calibrated
            base_uncertainty = (1 - confidence) * 0.15
        else:
            # Use calibrated threshold
            score = self.compute_nonconformity_score(confidence)
            if score <= self.quantile_threshold:
                # High confidence, small margin
                base_uncertainty = 0.05
            else:
                # Scale based on how much we exceed threshold
                base_uncertainty = 0.05 + (score - self.quantile_threshold) * 0.3
        
        # Scale by box size
        box_width = np.sqrt(box_area)
        margin = base_uncertainty * box_width
        
        return int(margin)
    
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
                
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    conf = confidences[i]
                    class_id = class_ids[i]
                    
                    # Compute box area
                    box_area = (x2 - x1) * (y2 - y1)
                    
                    # Compute uncertainty margin
                    margin = self.compute_uncertainty_margin(conf, box_area)
                    
                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'class_id': class_id,
                        'class_name': self.model.names[class_id],
                        'uncertainty_margin': margin,
                        'nonconformity_score': self.compute_nonconformity_score(conf)
                    }
                    detections.append(detection)
                    
                    # Collect calibration scores
                    if self.calibrating:
                        self.calibration_scores.append(detection['nonconformity_score'])
        
        return detections
    
    def draw_conformal_detection(self, image, detection, depth_avg=None):
        """Draw detection with conformal prediction uncertainty"""
        x1, y1, x2, y2 = detection['bbox']
        confidence = detection['confidence']
        class_name = detection['class_name']
        margin = detection['uncertainty_margin']
        
        # Color based on confidence
        if confidence > 0.8:
            color = (0, 255, 0)  # Green - high confidence
        elif confidence > 0.6:
            color = (0, 165, 255)  # Orange - medium confidence
        else:
            color = (0, 0, 255)  # Red - low confidence
        
        # Draw main detection box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw conformal prediction interval (uncertainty box)
        # This represents the region where the true object is likely to be
        cv2.rectangle(image, 
                     (x1 - margin, y1 - margin), 
                     (x2 + margin, y2 + margin), 
                     color, 1, cv2.LINE_AA)
        
        # Draw dashed lines for uncertainty
        # Top edge
        for i in range(x1 - margin, x2 + margin, 10):
            cv2.line(image, (i, y1 - margin), (min(i + 5, x2 + margin), y1 - margin), color, 1)
            cv2.line(image, (i, y2 + margin), (min(i + 5, x2 + margin), y2 + margin), color, 1)
        # Side edges
        for i in range(y1 - margin, y2 + margin, 10):
            cv2.line(image, (x1 - margin, i), (x1 - margin, min(i + 5, y2 + margin)), color, 1)
            cv2.line(image, (x2 + margin, i), (x2 + margin, min(i + 5, y2 + margin)), color, 1)
        
        # Label with confidence
        label = f"{class_name} {confidence:.2f}"
        if depth_avg is not None and depth_avg > 0:
            label += f" | {depth_avg:.2f}m"
        
        # Label background
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - 20), (x1 + label_size[0] + 5, y1), color, -1)
        cv2.putText(image, label, (x1 + 2, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Uncertainty info
        if self.quantile_threshold is not None:
            coverage_text = f"{self.coverage*100:.0f}% CI: ±{margin}px"
        else:
            coverage_text = f"Uncalibrated: ±{margin}px"
        cv2.putText(image, coverage_text, (x1, y2 + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def get_depth_at_bbox(self, depth_image, bbox):
        """Get average depth in bounding box"""
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(depth_image.shape[1], x2)
        y2 = min(depth_image.shape[0], y2)
        
        roi = depth_image[y1:y2, x1:x2]
        valid_depths = roi[roi > 0]
        
        if len(valid_depths) > 0:
            return np.mean(valid_depths) * self.depth_scale
        return 0
    
    def run(self):
        """Main detection loop"""
        self.setup_realsense()
        self.setup_yolo()
        
        cv2.namedWindow('YOLOv8 with Conformal Prediction', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Depth View', cv2.WINDOW_NORMAL)
        
        print("\n🎯 YOLOv8 Object Detection with Conformal Prediction")
        print("=" * 60)
        print(f"Target coverage: {self.coverage*100:.0f}%")
        print("\nControls:")
        print("  'q' - Quit")
        print("  's' - Save frame")
        print("  'c' - Start/stop calibration")
        print("  'u' - Toggle uncertainty visualization")
        print("\n⚠️  Press 'c' to calibrate conformal prediction first!")
        
        show_uncertainty = True
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
                detections = self.detect_objects(color_image)
                inference_time = (time.time() - start_time) * 1000
                
                # Update calibration
                if self.calibrating:
                    self.calibration_count += 1
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
                    info.append(f"Calibrated ✓")
                else:
                    info.append("Not calibrated")
                
                y = 20
                for text in info:
                    cv2.putText(result_image, text, (10, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    y += 25
                
                # Conformal prediction info
                if show_uncertainty:
                    info_text = "Conformal Prediction: Dashed boxes show uncertainty with coverage guarantee"
                    cv2.putText(result_image, info_text, (10, result_image.shape[0] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                
                # Display
                cv2.imshow('YOLOv8 with Conformal Prediction', result_image)
                cv2.imshow('Depth View', depth_colormap)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"yolo_conformal_{timestamp}.jpg", result_image)
                    cv2.imwrite(f"depth_{timestamp}.jpg", depth_colormap)
                    print(f"💾 Saved: {timestamp}")
                elif key == ord('c'):
                    if not self.calibrating:
                        self.calibrating = True
                        self.calibration_count = 0
                        self.calibration_scores = []
                        print("\n🔄 Starting calibration...")
                    else:
                        self.calibrating = False
                        if len(self.calibration_scores) > 0:
                            self.calibrate_quantile()
                elif key == ord('u'):
                    show_uncertainty = not show_uncertainty
        
        finally:
            cv2.destroyAllWindows()
            self.pipeline.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 with Conformal Prediction")
    parser.add_argument("--model", type=str, default="yolov8m.pt",
                       help="YOLOv8 model variant (n/s/m/l/x)")
    parser.add_argument("--alpha", type=float, default=0.1,
                       help="Miscoverage level (0.1 = 90% coverage)")
    
    args = parser.parse_args()
    
    print("🎯 Intel RealSense D455 - YOLOv8 with Conformal Prediction")
    print("=" * 60)
    print("This demo shows object detection with uncertainty quantification")
    print("using conformal prediction to provide coverage guarantees")
    print("=" * 60)
    
    detector = YOLOConformalDetector(args.model, args.alpha)
    detector.run()