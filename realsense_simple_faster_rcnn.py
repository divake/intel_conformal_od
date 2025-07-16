#!/usr/bin/env python3
"""
Simple RealSense object detection using OpenVINO with a pre-trained model
This avoids detectron2 dependencies
"""
import sys
import os
import time
import numpy as np
import cv2
import torch
import pyrealsense2 as rs
import openvino as ov
from pathlib import Path
import pickle


class SimpleOpenVINODetector:
    def __init__(self):
        # RealSense setup
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = None
        
        # OpenVINO setup
        self.core = ov.Core()
        self.model = None
        self.compiled_model = None
        
        # COCO classes
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
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
        """Setup OpenVINO object detection model"""
        print("📊 Setting up OpenVINO object detection model...")
        
        # Model name for SSD MobileNet V2
        model_name = "ssd_mobilenet_v2_coco"
        model_xml = f"{model_name}.xml"
        model_bin = f"{model_name}.bin"
        
        # Check if model exists, if not download it
        if not os.path.exists(model_xml):
            print("📥 Downloading SSD MobileNet V2 model...")
            try:
                import subprocess
                subprocess.run([
                    "omz_downloader",
                    "--name", model_name,
                    "--output_dir", ".",
                ], check=True)
                
                # Move files to current directory
                model_dir = f"./public/{model_name}/FP32/"
                if os.path.exists(model_dir):
                    import shutil
                    shutil.move(os.path.join(model_dir, model_xml), model_xml)
                    shutil.move(os.path.join(model_dir, model_bin), model_bin)
                    shutil.rmtree("./public", ignore_errors=True)
                print("✅ Model downloaded")
            except Exception as e:
                print(f"⚠️  Could not download model: {e}")
                print("Using fallback detection...")
                return False
        
        try:
            # Read model
            self.model = self.core.read_model(model_xml)
            
            # Compile model for CPU
            self.compiled_model = self.core.compile_model(self.model, "CPU")
            
            # Get input and output layers
            self.input_layer = self.compiled_model.input(0)
            self.output_layer = self.compiled_model.output(0)
            
            print("✅ OpenVINO model loaded")
            print(f"   Input shape: {self.input_layer.shape}")
            print(f"   Output shape: {self.output_layer.shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def detect_openvino(self, image):
        """Run object detection using OpenVINO"""
        # Get input shape
        n, c, h, w = self.input_layer.shape
        
        # Preprocess image
        input_image = cv2.resize(image, (w, h))
        input_image = input_image.transpose(2, 0, 1)  # HWC to CHW
        input_image = input_image.reshape(n, c, h, w)
        
        # Run inference
        results = self.compiled_model([input_image])[self.output_layer]
        
        # Parse detections
        detections = []
        
        # Results format: [1, 1, num_detections, 7]
        # Each detection: [image_id, class_id, confidence, x_min, y_min, x_max, y_max]
        for detection in results[0][0]:
            confidence = detection[2]
            
            if confidence > 0.5:  # Confidence threshold
                class_id = int(detection[1])
                
                # Get coordinates (normalized)
                x_min = int(detection[3] * image.shape[1])
                y_min = int(detection[4] * image.shape[0])
                x_max = int(detection[5] * image.shape[1])
                y_max = int(detection[6] * image.shape[0])
                
                # Ensure valid bounds
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(image.shape[1], x_max)
                y_max = min(image.shape[0], y_max)
                
                if x_max > x_min and y_max > y_min and class_id < len(self.class_names):
                    detections.append({
                        'bbox': [x_min, y_min, x_max, y_max],
                        'score': confidence,
                        'class_id': class_id,
                        'class_name': self.class_names[class_id] if class_id > 0 else 'background'
                    })
        
        return detections
    
    def detect_fallback(self, image):
        """Fallback detection using simple methods"""
        detections = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use simple blob detection
        # Setup SimpleBlobDetector parameters
        params = cv2.SimpleBlobDetector_Params()
        
        # Change thresholds
        params.minThreshold = 10
        params.maxThreshold = 200
        
        # Filter by area
        params.filterByArea = True
        params.minArea = 1500
        params.maxArea = 30000
        
        # Filter by circularity
        params.filterByCircularity = False
        
        # Filter by convexity
        params.filterByConvexity = False
        
        # Filter by inertia
        params.filterByInertia = False
        
        # Create detector
        detector = cv2.SimpleBlobDetector_create(params)
        
        # Detect blobs
        keypoints = detector.detect(gray)
        
        # Convert keypoints to detections
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            size = int(kp.size)
            
            x1 = max(0, x - size)
            y1 = max(0, y - size)
            x2 = min(image.shape[1], x + size)
            y2 = min(image.shape[0], y + size)
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'score': 0.5,
                'class_id': -1,
                'class_name': 'object'
            })
        
        # Also use edge detection for rectangles
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 2000:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Classify based on shape
                if len(approx) == 4:  # Rectangle-like
                    aspect_ratio = w / h if h > 0 else 1
                    if 0.5 < aspect_ratio < 2.0:
                        detections.append({
                            'bbox': [x, y, x+w, y+h],
                            'score': 0.6,
                            'class_id': 62,  # tv/monitor
                            'class_name': 'tv'
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
    
    def draw_detections(self, image, detections, depth_image=None):
        """Draw detection boxes with labels"""
        result = image.copy()
        
        # Color palette for different classes
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(self.class_names), 3), dtype=np.uint8)
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            score = det['score']
            class_id = det['class_id']
            class_name = det['class_name']
            
            # Get color for this class
            if class_id >= 0 and class_id < len(colors):
                color = tuple(map(int, colors[class_id]))
            else:
                color = (0, 255, 0)
            
            # Draw box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label = f"{class_name} {score:.2f}"
            
            # Add depth if available
            if depth_image is not None:
                depth = self.get_depth_at_bbox(depth_image, det['bbox'])
                if depth > 0:
                    label += f" | {depth:.2f}m"
            
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
        use_openvino = self.setup_openvino_model()
        
        cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Depth', cv2.WINDOW_NORMAL)
        
        print("\n🎥 Running object detection...")
        print("Press 'q' to quit")
        print("Press 's' to save frame")
        print(f"Mode: {'OpenVINO SSD' if use_openvino else 'Fallback'}")
        
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
                
                if use_openvino:
                    detections = self.detect_openvino(color_image)
                else:
                    detections = self.detect_fallback(color_image)
                
                inference_time = (time.time() - start_time) * 1000
                
                # Draw results
                result_image = self.draw_detections(color_image, detections, depth_image)
                
                # Create depth colormap
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03),
                    cv2.COLORMAP_JET
                )
                
                # Draw boxes on depth
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    cv2.rectangle(depth_colormap, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Info overlay
                info = [
                    f"FPS: {fps:.1f}",
                    f"Inference: {inference_time:.1f}ms",
                    f"Objects: {len(detections)}",
                    f"Backend: {'OpenVINO' if use_openvino else 'Fallback'}"
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
    print("🎯 Intel RealSense D455 - Object Detection")
    print("=" * 60)
    print("Using OpenVINO with SSD MobileNet V2")
    print("Detects 80 COCO object classes")
    
    detector = SimpleOpenVINODetector()
    detector.run()