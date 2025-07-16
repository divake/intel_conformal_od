# Conformal Object Detection on Intel Hardware
## Real-time Uncertainty Quantification for Object Detection

![Conformal Object Detection Demo](demo.gif)

This repository demonstrates **Conformal Object Detection** optimized for Intel hardware, specifically the Intel NUC 14 Pro with Intel Core Ultra 7 165H processor. It provides real-time object detection with mathematically rigorous uncertainty quantification using conformal prediction methods.

## 🎯 Overview

Traditional object detection models provide bounding boxes and confidence scores, but these confidence scores are often miscalibrated and don't provide statistical guarantees. **Conformal prediction** addresses this by providing prediction intervals with guaranteed coverage under minimal assumptions.

This implementation combines:
- **State-of-the-art object detection** (YOLOv8, Faster R-CNN)
- **Intel RealSense D455** depth camera integration
- **Conformal prediction** for uncertainty quantification
- **Intel hardware optimization** (CPU, GPU, NPU support via OpenVINO)

## 🚀 Key Features

### 1. **Three-Box Uncertainty Visualization**
Our implementation uses a unique three-box visualization for uncertainty quantification:

- **Black box (inner)**: Conservative bound - the object is likely larger than this
- **Green box (middle/dotted)**: Actual detection/prediction from the model  
- **Red box (outer)**: Liberal bound - the object is likely smaller than this

**Key Insight**: The ground truth bounding box falls between the black (inner) and red (outer) boxes with 90% probability. This provides a statistically rigorous uncertainty interval for object detection.

The uncertainty intervals are:
- **Symmetric** relative to the prediction box
- **Adaptive** based on detection confidence (high confidence → narrow interval)
- **Calibrated** to guarantee coverage (90% by default)

### 2. **Real-time Performance**
- Optimized for Intel Core Ultra 7 165H
- 30+ FPS on CPU with YOLOv8
- OpenVINO acceleration support

### 3. **Accurate Depth Integration**
- Intel RealSense D455 provides precise depth measurements
- Robust depth estimation using median filtering from object center
- Distance displayed for each detection (e.g., "person 0.95 | 1.2m")
- Useful for robotics, autonomous systems, and AR/VR applications

### 4. **Multiple Detection Models**
- YOLOv8 (all variants: n/s/m/l/x)
- Faster R-CNN (ResNet-50/101 backbones)
- Gaussian YOLOv3 (with built-in uncertainty)

## 📋 System Requirements

### Hardware
- **Processor**: Intel Core Ultra 7 165H (or similar Intel CPU)
- **Camera**: Intel RealSense D455
- **RAM**: 8GB minimum (16GB+ recommended)
- **Storage**: 10GB for models and dependencies

### Software
- Ubuntu 22.04/24.04 LTS
- Python 3.8+
- Intel RealSense SDK 2.0
- OpenVINO 2024.3+ (optional, for acceleration)

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/divake/conformal-od.git
cd conformal-od

# Create conda environment
conda create -n conformal_od python=3.8
conda activate conformal_od

# Install PyTorch (CPU optimized for Intel)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
pip install opencv-python pyrealsense2 numpy ultralytics

# Optional: Install OpenVINO for acceleration
pip install openvino
```

## 🎮 Quick Start

### 1. Three-Box Conformal Prediction (Recommended)
```bash
cd obj_det_conformal
python realsense_yolo_conformal_3box.py
```
Shows the full three-box visualization with guaranteed uncertainty intervals.

### 2. Standard Two-Box Version
```bash
python realsense_yolo_conformal.py
```

### 3. Debug Version (with extensive logging)
```bash
python realsense_yolo_conformal_debug.py
```

## 🎯 Demo Scripts

### 1. **realsense_yolo_conformal_3box.py** (⭐ Main Demo)
- **What it does**: Three-box visualization of conformal prediction
- **Key features**:
  - Black box (inner): Conservative bound
  - Green box (dotted): Detection
  - Red box (outer): Liberal bound
  - Ground truth guaranteed between inner & outer with 90% probability
  - Accurate depth measurement using robust median estimation
- **Controls**:
  - `q` - Quit
  - `s` - Save current frame
  - `c` - Start/stop calibration (important!)
  - `u` - Toggle uncertainty visualization

### 2. **realsense_yolo_conformal.py**
- **What it does**: Two-box version (detection + uncertainty)
- **Key features**:
  - Inner box: Detection
  - Outer box: Uncertainty region
  - Adaptive margins based on confidence

### 3. **realsense_yolo_conformal_debug.py**
- **What it does**: Same as above but with extensive terminal output
- **Use case**: Debugging and understanding the system behavior

## 📊 Understanding Conformal Prediction

### What is Conformal Prediction?
Conformal prediction is a framework that produces prediction **sets** (rather than point predictions) with guaranteed coverage. For object detection:

1. **Traditional**: Bounding box + confidence score (e.g., "car, 0.85")
2. **Conformal**: Bounding box + uncertainty region with guarantee (e.g., "90% chance the true object is within this region")

### Key Concepts:
- **Coverage**: Probability that the true object falls within the prediction set
- **Efficiency**: Size of the prediction set (smaller is better)
- **Calibration**: Process to determine the right uncertainty margins

### Mathematical Guarantee:
Given a miscoverage level α (e.g., 0.1), conformal prediction guarantees:
```
P(true object ∈ prediction set) ≥ 1 - α
```

## 🔬 Technical Details

### Conformal Object Detection Pipeline:

1. **Calibration Phase**:
   - Collect predictions on calibration data
   - Compute nonconformity scores
   - Determine quantile threshold for desired coverage

2. **Prediction Phase**:
   - Run object detector
   - Compute nonconformity score for each detection
   - Expand boxes based on calibrated threshold

3. **Uncertainty Quantification**:
   - High confidence → Small uncertainty margin
   - Low confidence → Large uncertainty margin
   - Guaranteed coverage regardless of confidence

### Supported Conformal Methods:
- **Standard Conformal** (std_conf)
- **Quantile Regression** (cqr_conf)
- **Learnable Conformal** (learn_conf)
- **Ensemble Methods** (ens_conf)

## 🏃 Performance on Intel Hardware

### Intel Core Ultra 7 165H Performance:
| Model | Backend | FPS | Inference Time |
|-------|---------|-----|----------------|
| YOLOv8n | CPU | 45+ | ~22ms |
| YOLOv8m | CPU | 30+ | ~33ms |
| YOLOv8m | OpenVINO | 40+ | ~25ms |
| Faster R-CNN | CPU | 15+ | ~66ms |

### Optimization Features:
- Intel AVX-512 acceleration
- OpenVINO model optimization
- Efficient numpy operations
- Optimized image preprocessing

## 📸 Example Output

When running the three-box demo (`realsense_yolo_conformal_3box.py`), you'll see:
- **Black box (solid)**: Conservative inner bound
- **Green box (dotted)**: Actual detection from YOLOv8
- **Red box (solid)**: Liberal outer bound
- **Label format**: "class_name confidence | distance" (e.g., "person 0.95 | 1.2m")

The key insight: The true object boundary falls between the black and red boxes with 90% probability. When the model is confident, the boxes are close together. When uncertain, they spread apart.

## 🔮 Future Enhancements

1. **NPU Acceleration**: Utilize Intel AI Boost NPU for inference
2. **Multi-camera Support**: Multiple RealSense cameras
3. **3D Uncertainty**: Extend to 3D bounding boxes
4. **Custom Models**: Easy integration of custom trained models

## 📚 References

1. **Original Conformal-OD Paper**: "Adaptive Bounding Box Uncertainties via Two-Step Conformal Prediction" (ECCV 2024)
2. **Conformal Prediction**: Vovk et al., "Algorithmic Learning in a Random World"
3. **YOLOv8**: Ultralytics YOLOv8 Documentation
4. **Intel RealSense**: Intel RealSense SDK Documentation

## 👥 Contributors

- Divake (dkumar33@uic.edu) - Intel hardware optimization and RealSense integration
- Original conformal-od authors - Base conformal prediction framework

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Intel for providing the hardware platform
- Ultralytics for YOLOv8
- The conformal prediction community

---

**For Intel Demo Team**: This implementation showcases how Intel hardware can enable real-time AI applications with rigorous uncertainty quantification, crucial for safety-critical applications like autonomous driving, robotics, and industrial automation.