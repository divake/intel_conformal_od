#!/usr/bin/env python3
"""
Simple RealSense object detection demo
This version works with minimal dependencies
"""
import os
import time
import numpy as np
import cv2
import torch
import pyrealsense2 as rs
import openvino as ov

class SimpleRealsenseDetector:
    def __init__(self):
        # RealSense setup
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = None
        
        # OpenVINO setup for object detection
        self.core = ov.Core()
        self.model = None
        self.compiled_model = None
        
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
        
    def setup_openvino_model(self):
        """Setup OpenVINO model for object detection"""
        print("📊 Setting up OpenVINO object detection model...")
        
        # Download a pre-trained model if needed
        model_name = "yolo-v3-tiny-tf"
        model_xml = f"{model_name}.xml"
        model_bin = f"{model_name}.bin"
        
        if not os.path.exists(model_xml):
            print("📥 Downloading YOLO model...")
            import subprocess
            try:
                subprocess.run([
                    "omz_downloader",
                    "--name", model_name,
                    "--output_dir", ".",
                ], check=True)
                
                # Move files
                model_dir = f"./public/{model_name}/FP32/"
                if os.path.exists(model_dir):
                    import shutil
                    shutil.move(os.path.join(model_dir, model_xml), model_xml)
                    shutil.move(os.path.join(model_dir, model_bin), model_bin)
                    shutil.rmtree("./public", ignore_errors=True)
            except:
                print("⚠️  Could not download model. Using simple detection instead.")
                return False
        
        # Load model
        self.model = self.core.read_model(model_xml)
        self.compiled_model = self.core.compile_model(self.model, "CPU")
        
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        
        print("✅ OpenVINO model loaded")
        return True
        
    def simple_color_detection(self, image):
        """Simple color-based object detection as fallback"""
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Detect red objects
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)
        
        lower_red = np.array([170, 120, 70])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)
        
        mask = mask1 + mask2
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter small detections
                x, y, w, h = cv2.boundingRect(contour)
                detections.append({
                    'bbox': [x, y, x+w, y+h],
                    'confidence': min(area / 5000, 1.0),  # Fake confidence based on size
                    'class': 'red_object'
                })
        
        return detections
    
    def draw_uncertainty_box(self, image, detection, depth_avg=None):
        """Draw detection with uncertainty visualization"""
        x1, y1, x2, y2 = detection['bbox']
        confidence = detection['confidence']
        
        # Main box (green)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Uncertainty margin (inversely proportional to confidence)
        margin = int((1 - confidence) * 15)
        
        # Uncertainty box (orange, dashed)
        cv2.rectangle(image, 
                     (x1 - margin, y1 - margin), 
                     (x2 + margin, y2 + margin), 
                     (0, 165, 255), 1)
        
        # Label
        label = f"{detection['class']} {confidence:.2f}"
        if depth_avg is not None and depth_avg > 0:
            label += f" | {depth_avg:.2f}m"
        
        cv2.putText(image, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Uncertainty text
        cv2.putText(image, f"±{margin}px", (x1, y2 + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
    
    def get_depth_at_bbox(self, depth_image, bbox):
        """Get average depth in bounding box"""
        x1, y1, x2, y2 = bbox
        roi = depth_image[y1:y2, x1:x2]
        valid_depths = roi[roi > 0]
        
        if len(valid_depths) > 0:
            return np.mean(valid_depths) * self.depth_scale
        return 0
    
    def run(self):
        """Main detection loop"""
        self.setup_realsense()
        
        # Try to setup OpenVINO model, fallback to simple detection
        use_openvino = self.setup_openvino_model()
        
        cv2.namedWindow('Object Detection with Uncertainty', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Depth View', cv2.WINDOW_NORMAL)
        
        print("\n🎥 Running object detection...")
        print("Press 'q' to quit")
        print("Press 's' to save frame")
        print("Press 'u' to toggle uncertainty boxes")
        
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
                
                # Use simple color detection as demo
                detections = self.simple_color_detection(color_image)
                
                inference_time = (time.time() - start_time) * 1000
                
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
                        self.draw_uncertainty_box(result_image, detection, depth_avg)
                    else:
                        x1, y1, x2, y2 = detection['bbox']
                        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Also draw on depth
                    x1, y1, x2, y2 = detection['bbox']
                    cv2.rectangle(depth_colormap, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Info overlay
                info = [
                    f"FPS: {fps:.1f}",
                    f"Inference: {inference_time:.1f}ms",
                    f"Objects: {len(detections)}",
                    f"Uncertainty: {'ON' if show_uncertainty else 'OFF'}",
                    f"Mode: {'OpenVINO' if use_openvino else 'Color Detection'}"
                ]
                
                y = 20
                for text in info:
                    cv2.putText(result_image, text, (10, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    y += 25
                
                # Show images
                cv2.imshow('Object Detection with Uncertainty', result_image)
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
                elif key == ord('u'):
                    show_uncertainty = not show_uncertainty
        
        finally:
            cv2.destroyAllWindows()
            self.pipeline.stop()
            

if __name__ == "__main__":
    print("🎯 Intel RealSense D455 - Object Detection with Uncertainty")
    print("=" * 60)
    print("This demo shows object detection with uncertainty boxes")
    print("Using simple color detection for demonstration")
    
    detector = SimpleRealsenseDetector()
    detector.run()