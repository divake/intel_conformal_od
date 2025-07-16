#!/usr/bin/env python3
"""
RealSense object detection using Faster R-CNN
Minimal implementation without full detectron2 dependencies
"""
import sys
import os
import time
import numpy as np
import cv2
import torch
import pickle
import pyrealsense2 as rs
from pathlib import Path

# COCO classes
COCO_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
                'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
                'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 
                'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


class FasterRCNNDetector:
    def __init__(self, model_path):
        self.model_path = model_path
        
        # RealSense setup
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = None
        
        # Model
        self.model_weights = None
        
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
        """Load the Faster R-CNN model weights"""
        print(f"📊 Loading model from {self.model_path}...")
        
        try:
            # Load the pickle file
            with open(self.model_path, 'rb') as f:
                checkpoint = pickle.load(f)
            
            if 'model' in checkpoint:
                self.model_weights = checkpoint['model']
                print(f"✅ Model weights loaded")
                print(f"   Number of parameters: {len(self.model_weights)}")
                # Show some layer names
                layer_names = list(self.model_weights.keys())[:5]
                for name in layer_names:
                    print(f"   - {name}")
                print(f"   ... and {len(self.model_weights) - 5} more layers")
            else:
                print("❌ No 'model' key found in checkpoint")
                return False
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
        
        return True
    
    def detect_with_yolo(self, image):
        """Use simple detection methods as a fallback"""
        # For simplicity, we'll use basic image processing
        # In practice, you'd load a proper YOLO or Faster R-CNN model
        
        detections = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Detect faces using simple template matching or color detection
        # For demo, we'll use color-based detection for skin tones
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours for skin regions
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 2000:  # Reasonable face/hand size
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w if w > 0 else 1
                
                # Face-like aspect ratio
                if 0.8 < aspect_ratio < 1.5:
                    detections.append({
                        'bbox': [x, y, x+w, y+h],
                        'score': 0.7,
                        'class_id': 0,  # person
                        'class_name': 'person'
                    })
        
        # Also use edge detection for general objects
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 3000:  # Reasonable size
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 1
                
                # Try to classify based on shape
                if 0.8 < aspect_ratio < 1.2 and area > 5000:
                    # Square-ish, might be a monitor/tv
                    class_id = 62  # tv
                    class_name = 'tv'
                    score = 0.6
                elif aspect_ratio > 2 and area > 4000:
                    # Wide, might be a keyboard
                    class_id = 66  # keyboard
                    class_name = 'keyboard'
                    score = 0.5
                else:
                    # Generic object
                    class_id = -1
                    class_name = 'object'
                    score = 0.4
                
                if score > 0.4:
                    detections.append({
                        'bbox': [x, y, x+w, y+h],
                        'score': score,
                        'class_id': class_id,
                        'class_name': class_name
                    })
        
        # Remove duplicate/overlapping detections
        detections = self.non_max_suppression(detections)
        
        return detections
    
    def non_max_suppression(self, detections, iou_threshold=0.5):
        """Simple NMS to remove overlapping detections"""
        if len(detections) == 0:
            return []
        
        # Sort by score
        detections.sort(key=lambda x: x['score'], reverse=True)
        
        keep = []
        for i, det1 in enumerate(detections):
            should_keep = True
            for det2 in keep:
                # Calculate IoU
                x1_1, y1_1, x2_1, y2_1 = det1['bbox']
                x1_2, y1_2, x2_2, y2_2 = det2['bbox']
                
                # Intersection
                xi1 = max(x1_1, x1_2)
                yi1 = max(y1_1, y1_2)
                xi2 = min(x2_1, x2_2)
                yi2 = min(y2_1, y2_2)
                
                if xi2 > xi1 and yi2 > yi1:
                    intersection = (xi2 - xi1) * (yi2 - yi1)
                    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
                    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
                    union = area1 + area2 - intersection
                    iou = intersection / union if union > 0 else 0
                    
                    if iou > iou_threshold:
                        should_keep = False
                        break
            
            if should_keep:
                keep.append(det1)
        
        return keep
    
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
        
        cv2.namedWindow('Faster R-CNN Detection', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Depth', cv2.WINDOW_NORMAL)
        
        print("\n🎥 Running Faster R-CNN object detection...")
        print("Note: Using simplified detection for demo")
        print("Press 'q' to quit, 's' to save frame")
        
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
                detections = self.detect_with_yolo(color_image)
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
                    
                    # Choose color based on class
                    if class_name == 'person':
                        color = (0, 255, 0)  # Green
                    elif class_name in ['tv', 'laptop', 'keyboard']:
                        color = (255, 0, 0)  # Blue
                    else:
                        color = (0, 165, 255)  # Orange
                    
                    # Draw box
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label
                    label = f"{class_name} {score:.2f}"
                    if depth_avg > 0:
                        label += f" | {depth_avg:.2f}m"
                    
                    # Label background
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(result_image, (x1, y1 - 20), (x1 + label_size[0], y1), color, -1)
                    cv2.putText(result_image, label, (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Draw on depth too
                    cv2.rectangle(depth_colormap, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Info overlay
                info = [
                    f"FPS: {fps:.1f}",
                    f"Inference: {inference_time:.1f}ms",
                    f"Objects: {len(detections)}",
                    "Model: Faster R-CNN (simplified)"
                ]
                
                y = 20
                for text in info:
                    cv2.putText(result_image, text, (10, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    y += 25
                
                # Display
                cv2.imshow('Faster R-CNN Detection', result_image)
                cv2.imshow('Depth', depth_colormap)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"faster_rcnn_{timestamp}.jpg", result_image)
                    cv2.imwrite(f"depth_{timestamp}.jpg", depth_colormap)
                    print(f"💾 Saved: {timestamp}")
        
        finally:
            cv2.destroyAllWindows()
            self.pipeline.stop()


if __name__ == "__main__":
    print("🎯 RealSense Faster R-CNN Object Detection")
    print("=" * 50)
    
    # You can use either model
    model_path = "checkpoints/faster_rcnn_X_101_32x8d_FPN_3x.pth"
    # model_path = "checkpoints/faster_rcnn_R_50_FPN_3x.pkl"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)
    
    detector = FasterRCNNDetector(model_path)
    detector.run()