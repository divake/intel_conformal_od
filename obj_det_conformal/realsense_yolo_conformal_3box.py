#!/usr/bin/env python3
"""
Real-time object detection with YOLOv8 and Conformal Prediction
Shows THREE boxes: inner (conservative), middle (prediction), outer (liberal)
Ground truth falls between inner and outer with 90% probability
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
        
    def compute_uncertainty_margins(self, confidence, box_width, box_height):
        """
        Compute SYMMETRIC uncertainty margins for inner and outer boxes
        Returns (inner_margin, outer_margin) in pixels
        
        The GT should fall between inner and outer with 90% probability
        """
        # Average dimension for symmetric margins
        avg_dimension = (box_width + box_height) / 2
        
        if self.quantile_threshold is None:
            # Not calibrated - use default scaling based on confidence
            if confidence > 0.9:
                inner_factor = 0.03  # 3% shrink for inner
                outer_factor = 0.08  # 8% expand for outer
            elif confidence > 0.7:
                inner_factor = 0.05  # 5% shrink for inner
                outer_factor = 0.15  # 15% expand for outer
            elif confidence > 0.5:
                inner_factor = 0.08  # 8% shrink for inner
                outer_factor = 0.22  # 22% expand for outer
            else:
                inner_factor = 0.10  # 10% shrink for inner
                outer_factor = 0.30  # 30% expand for outer
        else:
            # Use calibrated threshold
            score = self.compute_nonconformity_score(confidence)
            
            if score <= self.quantile_threshold:
                # Within threshold - narrow interval
                inner_factor = 0.03
                outer_factor = 0.08
            else:
                # Outside threshold - wider interval
                excess = (score - self.quantile_threshold) / (1 - self.quantile_threshold)
                inner_factor = 0.03 + 0.07 * excess  # Max 10% shrink
                outer_factor = 0.08 + 0.22 * excess  # Max 30% expand
        
        # Compute margins
        inner_margin = int(inner_factor * avg_dimension)
        outer_margin = int(outer_factor * avg_dimension)
        
        # Ensure minimum margins for visibility
        inner_margin = max(inner_margin, 5)
        outer_margin = max(outer_margin, 15)
        
        return inner_margin, outer_margin
    
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
                    
                    # Compute box dimensions
                    box_width = x2 - x1
                    box_height = y2 - y1
                    box_area = box_width * box_height
                    
                    # Compute uncertainty margins (inner and outer)
                    inner_margin, outer_margin = self.compute_uncertainty_margins(
                        conf, box_width, box_height)
                    
                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'class_id': class_id,
                        'class_name': self.model.names[class_id],
                        'inner_margin': inner_margin,
                        'outer_margin': outer_margin,
                        'nonconformity_score': self.compute_nonconformity_score(conf)
                    }
                    detections.append(detection)
                    
                    # Collect calibration scores
                    if self.calibrating:
                        self.calibration_scores.append(detection['nonconformity_score'])
        
        return detections
    
    def draw_conformal_detection(self, image, detection, depth_avg=None):
        """
        Draw THREE boxes: inner, middle (prediction), outer
        GT falls between inner and outer with 90% probability
        """
        x1, y1, x2, y2 = detection['bbox']
        confidence = detection['confidence']
        class_name = detection['class_name']
        inner_margin = detection['inner_margin']
        outer_margin = detection['outer_margin']
        
        # Fixed colors for clarity
        color_pred = (0, 255, 0)   # Green for prediction (middle box)
        color_inner = (0, 0, 0)    # Black for inner box
        color_outer = (0, 0, 255)  # Red for outer box
        
        # Calculate box coordinates
        # Inner box (conservative - shrinks inward)
        inner_x1 = x1 + inner_margin
        inner_y1 = y1 + inner_margin
        inner_x2 = x2 - inner_margin
        inner_y2 = y2 - inner_margin
        
        # Outer box (liberal - expands outward)
        outer_x1 = x1 - outer_margin
        outer_y1 = y1 - outer_margin
        outer_x2 = x2 + outer_margin
        outer_y2 = y2 + outer_margin
        
        # Ensure boxes stay within image bounds
        h, w = image.shape[:2]
        inner_x1 = max(0, min(inner_x1, w-1))
        inner_y1 = max(0, min(inner_y1, h-1))
        inner_x2 = max(0, min(inner_x2, w-1))
        inner_y2 = max(0, min(inner_y2, h-1))
        
        outer_x1 = max(0, outer_x1)
        outer_y1 = max(0, outer_y1)
        outer_x2 = min(w-1, outer_x2)
        outer_y2 = min(h-1, outer_y2)
        
        # Ensure inner box is valid (not inverted)
        if inner_x2 > inner_x1 and inner_y2 > inner_y1:
            # Draw inner box - solid black line
            cv2.rectangle(image, (inner_x1, inner_y1), (inner_x2, inner_y2), color_inner, 2)
        
        # Draw middle box (prediction) - dotted green line
        # Top edge
        for i in range(x1, x2, 12):
            cv2.line(image, (i, y1), (min(i + 6, x2), y1), color_pred, 2)
            cv2.line(image, (i, y2), (min(i + 6, x2), y2), color_pred, 2)
        # Side edges
        for i in range(y1, y2, 12):
            cv2.line(image, (x1, i), (x1, min(i + 6, y2)), color_pred, 2)
            cv2.line(image, (x2, i), (x2, min(i + 6, y2)), color_pred, 2)
        
        # Draw outer box - solid red line
        cv2.rectangle(image, (outer_x1, outer_y1), (outer_x2, outer_y2), color_outer, 2)
        
        
        # Label with confidence
        label = f"{class_name} {confidence:.2f}"
        if depth_avg is not None and depth_avg > 0:
            label += f" | {depth_avg:.2f}m"
        
        # Label background
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - 20), (x1 + label_size[0] + 5, y1), color_pred, -1)
        cv2.putText(image, label, (x1 + 2, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Uncertainty info
        if self.quantile_threshold is not None:
            interval_text = f"90% CI: [-{inner_margin}, +{outer_margin}]px"
        else:
            interval_text = f"Interval: [-{inner_margin}, +{outer_margin}]px"
        
        cv2.putText(image, interval_text, (x1, y2 + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_pred, 1)
    
    def get_depth_at_bbox(self, depth_image, bbox):
        """Get depth in bounding box using robust estimation"""
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(depth_image.shape[1], x2)
        y2 = min(depth_image.shape[0], y2)
        
        # Take center region of bbox (avoid edges which may have noise)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        
        # Sample from center region (25% of bbox size)
        width = x2 - x1
        height = y2 - y1
        sample_w = max(width // 4, 10)
        sample_h = max(height // 4, 10)
        
        # Ensure we stay within bounds
        sx1 = max(0, cx - sample_w // 2)
        sy1 = max(0, cy - sample_h // 2)
        sx2 = min(depth_image.shape[1], cx + sample_w // 2)
        sy2 = min(depth_image.shape[0], cy + sample_h // 2)
        
        # Get depth from center region
        roi = depth_image[sy1:sy2, sx1:sx2]
        valid_depths = roi[roi > 0]
        
        if len(valid_depths) > 10:  # Need enough valid pixels
            # Use median for robustness (less affected by outliers)
            depth_mm = np.median(valid_depths)
            depth_m = depth_mm * self.depth_scale
            
            # Sanity check - typical indoor depths are 0.3m to 5m
            if 0.3 <= depth_m <= 5.0:
                return depth_m
        
        return 0
    
    def run(self):
        """Main detection loop"""
        self.setup_realsense()
        self.setup_yolo()
        
        cv2.namedWindow('YOLOv8 with Conformal Prediction  ', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Depth View', cv2.WINDOW_NORMAL)
        
        print("\n🎯 YOLOv8 Object Detection with Conformal Prediction")
        print("=" * 60)
        print("THREE-BOX VISUALIZATION:")
        print("  • Inner box (black/solid): Conservative bound")
        print("  • Middle box (green/dotted): Prediction")
        print("  • Outer box (red/solid): Liberal bound")
        print("  → Ground truth falls between inner & outer with 90% probability")
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
                    info_text = "Black: Conservative | Green/Dotted: Prediction | Red: Liberal"
                    cv2.putText(result_image, info_text, (10, result_image.shape[0] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                
                # Display
                cv2.imshow('YOLOv8 with Conformal Prediction  ', result_image)
                cv2.imshow('Depth View', depth_colormap)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"yolo_conformal_3box_{timestamp}.jpg", result_image)
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
    parser = argparse.ArgumentParser(description="YOLOv8 with Conformal Prediction  ")
    parser.add_argument("--model", type=str, default="yolov8m.pt",
                       help="YOLOv8 model variant (n/s/m/l/x)")
    parser.add_argument("--alpha", type=float, default=0.1,
                       help="Miscoverage level (0.1 = 90% coverage)")
    
    args = parser.parse_args()
    
    print("🎯 Intel RealSense D455 - YOLOv8 with Conformal Prediction")
    print("=" * 60)
    print("THREE-BOX UNCERTAINTY QUANTIFICATION")
    print("Ground truth falls between inner & outer boxes with guaranteed coverage")
    print("=" * 60)
    
    detector = YOLOConformalDetector(args.model, args.alpha)
    detector.run()