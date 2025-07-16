#!/usr/bin/env python3
"""
Create high-quality GIF from demo video for GitHub README
"""
import subprocess
import os
import sys
from pathlib import Path

def create_gif(input_video, output_gif="demo.gif", fps=10, width=800, start_time=0, duration=30):
    """
    Convert video to GIF with optimized settings for GitHub
    
    Args:
        input_video: Path to input video file
        output_gif: Output GIF filename
        fps: Frames per second (10-15 recommended for smaller size)
        width: Width of GIF (height auto-calculated)
        start_time: Start time in seconds
        duration: Duration in seconds (30s recommended for GitHub)
    """
    
    if not os.path.exists(input_video):
        print(f"❌ Video not found: {input_video}")
        return False
    
    print(f"🎬 Converting video to GIF...")
    print(f"   Input: {input_video}")
    print(f"   Output: {output_gif}")
    print(f"   FPS: {fps}")
    print(f"   Width: {width}px")
    print(f"   Duration: {duration}s (from {start_time}s)")
    
    # Create palette for better quality
    palette_file = "palette.png"
    
    # Step 1: Generate palette for better colors
    palette_cmd = [
        'ffmpeg', '-ss', str(start_time), '-t', str(duration),
        '-i', input_video,
        '-vf', f'fps={fps},scale={width}:-1:flags=lanczos,palettegen=stats_mode=diff',
        '-y', palette_file
    ]
    
    print("\n📊 Generating color palette...")
    try:
        subprocess.run(palette_cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to generate palette: {e}")
        return False
    
    # Step 2: Create GIF using palette
    gif_cmd = [
        'ffmpeg', '-ss', str(start_time), '-t', str(duration),
        '-i', input_video,
        '-i', palette_file,
        '-lavfi', f'fps={fps},scale={width}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle',
        '-y', output_gif
    ]
    
    print("🎨 Creating GIF with optimized palette...")
    try:
        subprocess.run(gif_cmd, check=True, capture_output=True)
        
        # Check file size
        size_mb = os.path.getsize(output_gif) / (1024 * 1024)
        print(f"\n✅ GIF created successfully!")
        print(f"   Size: {size_mb:.1f} MB")
        
        # Clean up palette
        if os.path.exists(palette_file):
            os.remove(palette_file)
        
        if size_mb > 10:
            print(f"\n⚠️  GIF is larger than 10MB. Consider:")
            print(f"   - Reducing duration (current: {duration}s)")
            print(f"   - Lowering FPS (current: {fps})")
            print(f"   - Reducing width (current: {width}px)")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create GIF: {e}")
        return False
    except FileNotFoundError:
        print("❌ ffmpeg not found. Please install it:")
        print("   sudo apt install ffmpeg")
        return False

def create_multiple_gifs(input_video):
    """Create multiple GIFs showcasing different aspects"""
    
    # Create main demo GIF (overview)
    print("\n🎯 Creating main demo GIF...")
    create_gif(input_video, "demo_overview.gif", fps=10, width=800, start_time=5, duration=20)
    
    # Create calibration demo
    print("\n📊 Creating calibration demo GIF...")
    create_gif(input_video, "demo_calibration.gif", fps=10, width=800, start_time=10, duration=15)
    
    # Create uncertainty visualization demo
    print("\n📐 Creating uncertainty visualization GIF...")
    create_gif(input_video, "demo_uncertainty.gif", fps=10, width=800, start_time=30, duration=15)
    
    print("\n🎬 All GIFs created! Add them to README.md:")
    print("""
## 📸 Demo

### Overview
![Conformal Object Detection Demo](demo_overview.gif)

### Calibration Process
![Calibration Demo](demo_calibration.gif)

### Uncertainty Visualization
![Uncertainty Visualization](demo_uncertainty.gif)
""")

if __name__ == "__main__":
    video_file = "Obj_det_Conformal_prediction.webm"
    
    if len(sys.argv) > 1:
        video_file = sys.argv[1]
    
    if not os.path.exists(video_file):
        print(f"❌ Video file not found: {video_file}")
        print("\nUsage: python create_demo_gif.py [video_file]")
        print(f"Default: {video_file}")
        sys.exit(1)
    
    # Create single optimized GIF
    print("🎬 Intel Conformal Object Detection - GIF Creator")
    print("=" * 50)
    
    # First create a single optimized GIF
    success = create_gif(
        video_file, 
        "demo.gif", 
        fps=10,      # Reduced for smaller size
        width=800,   # Standard width
        start_time=2,  # Skip initial setup
        duration=20    # Shorter duration
    )
    
    if success:
        print("\n💡 To create multiple GIFs for different sections:")
        print("   Uncomment the create_multiple_gifs() call in the script")
        
    # Uncomment to create multiple GIFs
    # create_multiple_gifs(video_file)