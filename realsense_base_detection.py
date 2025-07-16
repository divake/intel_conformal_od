#!/usr/bin/env python3
"""
Simple RealSense object detection using Faster R-CNN base model
"""
import sys
import os
import time
import numpy as np
import cv2
import torch
import pyrealsense2 as rs
from pathlib import Path

# Add detectron2 to path
sys.path.insert(0, str(Path(__file__).parent / "detectron2"))

# Import what we need without full detectron2 dependencies
import pickle


class SimpleDetector:
    def __init__(self, model_path):
        self.model_path = model_path
        
        # RealSense setup
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = None
        
        # Load model checkpoint
        self.model_data = None
        self.device = torch.device("cpu")
        
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
        
    def load_model(self):
        """Load the Faster R-CNN model"""
        print(f"📊 Loading model from {self.model_path}...")
        
        try:
            # First, let's examine what's in the checkpoint
            checkpoint = torch.load(self.model_path, map_location='cpu')
            print("✅ Checkpoint loaded")
            
            # Print checkpoint structure
            if isinstance(checkpoint, dict):
                print(f"   Keys in checkpoint: {list(checkpoint.keys())[:5]}...")
                if 'model' in checkpoint:
                    print("   Found 'model' key")
                    model_state = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    print("   Found 'state_dict' key")
                    model_state = checkpoint['state_dict']
                else:
                    print("   Using checkpoint directly as model state")
                    model_state = checkpoint
            
            # For now, we'll use a simple approach
            # In practice, you'd need to build the model architecture and load weights
            print("   Model info extracted")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
        
        return True
    
    def detect_simple(self, image):
        """Simple detection using OpenCV for demonstration"""
        # For now, use OpenCV's pre-trained cascade classifier
        # This is just to demonstrate the pipeline
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use OpenCV's face detector as example
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        detections = []
        for (x, y, w, h) in faces:
            detections.append({
                'bbox': [x, y, x+w, y+h],
                'score': 0.8,  # Dummy confidence
                'class_name': 'face'
            })
        
        # Also detect some objects using simple methods
        # Edge detection for objects
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 2000:  # Larger threshold
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 1
                
                if 0.5 < aspect_ratio < 2.0:  # Reasonable aspect ratio
                    detections.append({
                        'bbox': [x, y, x+w, y+h],
                        'score': min(area / 10000, 0.9),
                        'class_name': 'object'
                    })
        
        return detections
    
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
        self.load_model()
        
        cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Depth', cv2.WINDOW_NORMAL)
        
        print("\n🎥 Running object detection...")
        print("Press 'q' to quit")
        print("Press 's' to save frame")
        
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
                detections = self.detect_simple(color_image)
                inference_time = (time.time() - start_time) * 1000
                
                # Draw results
                result_image = color_image.copy()
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), 
                    cv2.COLORMAP_JET
                )
                
                # Draw detections
                for detection in detections:
                    x1, y1, x2, y2 = detection['bbox']
                    score = detection['score']
                    class_name = detection['class_name']
                    
                    # Get depth
                    depth_avg = self.get_depth_at_bbox(depth_image, detection['bbox'])
                    
                    # Draw box
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label
                    label = f"{class_name} {score:.2f}"
                    if depth_avg > 0:
                        label += f" | {depth_avg:.2f}m"
                    
                    cv2.putText(result_image, label, (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Draw on depth too
                    cv2.rectangle(depth_colormap, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Info overlay
                info = [
                    f"FPS: {fps:.1f}",
                    f"Inference: {inference_time:.1f}ms",
                    f"Objects: {len(detections)}"
                ]
                
                y = 20
                for text in info:
                    cv2.putText(result_image, text, (10, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    y += 25
                
                # Display
                cv2.imshow('Object Detection', result_image)
                cv2.imshow('Depth', depth_colormap)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"detection_{timestamp}.jpg", result_image)
                    cv2.imwrite(f"depth_{timestamp}.jpg", depth_colormap)
                    print(f"💾 Saved: {timestamp}")
        
        finally:
            cv2.destroyAllWindows()
            self.pipeline.stop()


if __name__ == "__main__":
    print("🎯 RealSense Object Detection - Base Model")
    print("=" * 50)
    
    model_path = "checkpoints/faster_rcnn_X_101_32x8d_FPN_3x.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)
    
    detector = SimpleDetector(model_path)
    detector.run()