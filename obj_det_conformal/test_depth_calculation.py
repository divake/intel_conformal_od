#!/usr/bin/env python3
"""
Debug depth calculation with RealSense
"""
import numpy as np
import cv2
import pyrealsense2 as rs

# Setup RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Get depth scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"Depth scale: {depth_scale}")

# Create align object
align = rs.align(rs.stream.color)

try:
    for i in range(100):
        # Wait for frames
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue
            
        # Convert to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Get center region depth
        h, w = depth_image.shape
        cx, cy = w//2, h//2
        
        # Sample a 50x50 region at center
        roi = depth_image[cy-25:cy+25, cx-25:cx+25]
        valid_depths = roi[roi > 0]
        
        if len(valid_depths) > 0:
            # Different statistics
            avg_depth_mm = np.mean(valid_depths)
            median_depth_mm = np.median(valid_depths)
            min_depth_mm = np.min(valid_depths)
            max_depth_mm = np.max(valid_depths)
            
            # Convert to meters
            avg_depth_m = avg_depth_mm * depth_scale
            median_depth_m = median_depth_mm * depth_scale
            
            print(f"\nFrame {i}:")
            print(f"  Raw depth values: min={min_depth_mm}, max={max_depth_mm}, avg={avg_depth_mm:.0f}")
            print(f"  Depth in meters: avg={avg_depth_m:.2f}m, median={median_depth_m:.2f}m")
            print(f"  Valid pixels: {len(valid_depths)}/{50*50}")
            
            # Draw crosshair on color image
            cv2.line(color_image, (cx-20, cy), (cx+20, cy), (0, 255, 0), 2)
            cv2.line(color_image, (cx, cy-20), (cx, cy+20), (0, 255, 0), 2)
            cv2.rectangle(color_image, (cx-25, cy-25), (cx+25, cy+25), (0, 255, 0), 1)
            
            # Add depth text
            cv2.putText(color_image, f"{median_depth_m:.2f}m", (cx-30, cy-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        # Display
        cv2.imshow('Color', color_image)
        
        # Create depth colormap
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow('Depth', depth_colormap)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
finally:
    pipeline.stop()
    cv2.destroyAllWindows()