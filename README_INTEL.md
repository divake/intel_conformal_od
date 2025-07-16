# Conformal Object Detection on Intel Hardware
## Real-time Uncertainty Quantification for Object Detection

This repository demonstrates **Conformal Object Detection** optimized for Intel hardware, specifically the Intel NUC 14 Pro with Intel Core Ultra 7 165H processor. It provides real-time object detection with mathematically rigorous uncertainty quantification using conformal prediction methods.

## 🎯 Overview

Traditional object detection models provide bounding boxes and confidence scores, but these confidence scores are often miscalibrated and don't provide statistical guarantees. **Conformal prediction** addresses this by providing prediction intervals with guaranteed coverage under minimal assumptions.

This implementation combines:
- **State-of-the-art object detection** (YOLOv8, Faster R-CNN)
- **Intel RealSense D455** depth camera integration
- **Conformal prediction** for uncertainty quantification
- **Intel hardware optimization** (CPU, GPU, NPU support via OpenVINO)

## 🚀 Key Features

### 1. **Uncertainty Visualization**
- **Inner box**: Traditional object detection bounding box
- **Outer box**: Conformal prediction interval showing uncertainty region
- **Coverage guarantee**: e.g., 90% probability that true object lies within the outer box

### 2. **Real-time Performance**
- Optimized for Intel Core Ultra 7 165H
- 30+ FPS on CPU with YOLOv8
- OpenVINO acceleration support

### 3. **Depth Integration**
- Intel RealSense D455 provides depth information
- 3D object localization
- Distance measurement for each detection

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

### 1. Basic YOLOv8 with Conformal Prediction
```bash
cd obj_det_conformal
python realsense_yolo_conformal.py
```

### 2. Simple Object Detection Demo
```bash
python realsense_detection_simple.py
```

### 3. Full Conformal Prediction Demo
```bash
python realsense_conformal_demo.py
```

## 🎯 Demo Scripts

### 1. **realsense_yolo_conformal.py**
- **What it does**: Real-time YOLOv8 detection with conformal prediction
- **Key features**:
  - Automatic model download (YOLOv8m by default)
  - Calibration mode (press 'c' to calibrate)
  - Uncertainty visualization with coverage guarantees
  - Depth measurement for each object
- **Controls**:
  - `q` - Quit
  - `s` - Save current frame
  - `c` - Start/stop calibration
  - `u` - Toggle uncertainty visualization

### 2. **realsense_conformal_demo.py**
- **What it does**: Demonstrates conformal prediction concepts
- **Key features**:
  - Adjustable coverage levels (press 'a')
  - Visual explanation of uncertainty
  - Multiple detection backends

### 3. **realsense_detection_simple.py**
- **What it does**: Simple baseline without conformal prediction
- **Use case**: Compare with/without uncertainty quantification

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

When running the demos, you'll see:
- **Green boxes**: High confidence detections (>80%)
- **Orange boxes**: Medium confidence (60-80%)
- **Red boxes**: Low confidence (<60%)
- **Dashed outer boxes**: Conformal prediction intervals
- **Depth labels**: Distance to each object in meters

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

- Divake (divek1805@gmail.com) - Intel hardware optimization and RealSense integration
- Original conformal-od authors - Base conformal prediction framework

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Intel for providing the hardware platform
- Ultralytics for YOLOv8
- The conformal prediction community

---

**For Intel Demo Team**: This implementation showcases how Intel hardware can enable real-time AI applications with rigorous uncertainty quantification, crucial for safety-critical applications like autonomous driving, robotics, and industrial automation.